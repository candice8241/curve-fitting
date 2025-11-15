#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XRD Azimuthal Integration Script
================================
Author: candicewang928@gmail.com
Created: Nov 15, 2025

This script performs azimuthal integration on XRD diffraction ring data stored in HDF5 format.
It uses pyFAI (Fast Azimuthal Integration) library to integrate 2D diffraction patterns into 1D profiles.

Features:
- Reads HDF5 files containing 2D diffraction patterns
- Uses PONI calibration files for detector geometry
- Supports mask files to exclude bad pixels/regions
- Batch processing of multiple files
- Configurable output directory
- Outputs integrated data in multiple formats (xy, chi, dat)

Requirements:
- pyFAI
- h5py
- numpy
- fabio (for some file formats)
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import h5py
    import numpy as np
    import pyFAI
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
except ImportError as e:
    print(f"Error: Missing required library. {e}")
    print("Please install required packages:")
    print("  pip install pyFAI h5py numpy fabio")
    sys.exit(1)


class XRDAzimuthalIntegrator:
    """Class to handle azimuthal integration of XRD diffraction data."""

    def __init__(self, poni_file: str, mask_file: Optional[str] = None):
        """
        Initialize the azimuthal integrator.

        Parameters:
        -----------
        poni_file : str
            Path to the PONI calibration file
        mask_file : str, optional
            Path to the mask file (can be .npy, .edf, or .tif)
        """
        self.poni_file = poni_file
        self.mask_file = mask_file
        self.ai = None
        self.mask = None

        self._load_calibration()
        if mask_file:
            self._load_mask()

    def _load_calibration(self):
        """Load the PONI calibration file."""
        if not os.path.exists(self.poni_file):
            raise FileNotFoundError(f"PONI file not found: {self.poni_file}")

        print(f"Loading calibration from: {self.poni_file}")
        self.ai = pyFAI.load(self.poni_file)
        print(f"  Detector: {self.ai.detector.name}")
        print(f"  Distance: {self.ai.dist * 1000:.2f} mm")
        print(f"  Wavelength: {self.ai.wavelength * 1e10:.4f} Å")

    def _load_mask(self):
        """Load the mask file."""
        if not os.path.exists(self.mask_file):
            print(f"Warning: Mask file not found: {self.mask_file}")
            return

        print(f"Loading mask from: {self.mask_file}")

        # Try to load mask based on file extension
        ext = os.path.splitext(self.mask_file)[1].lower()

        if ext == '.npy':
            self.mask = np.load(self.mask_file)
        elif ext in ['.edf', '.tif', '.tiff']:
            try:
                import fabio
                img = fabio.open(self.mask_file)
                self.mask = img.data
            except ImportError:
                print("Warning: fabio not installed. Cannot read mask file.")
                return
        elif ext in ['.h5', '.hdf5']:
            with h5py.File(self.mask_file, 'r') as f:
                # Try common dataset names
                for key in ['mask', 'data', 'entry/data/data']:
                    if key in f:
                        self.mask = f[key][:]
                        break
                else:
                    # Use first dataset found
                    keys = list(f.keys())
                    if keys:
                        self.mask = f[keys[0]][:]
        else:
            print(f"Warning: Unsupported mask file format: {ext}")
            return

        print(f"  Mask shape: {self.mask.shape}")
        print(f"  Masked pixels: {np.sum(self.mask)}")

    def integrate_file(self, h5_file: str, output_dir: str,
                      npt: int = 2048,
                      unit: str = "q_A^-1",
                      output_format: str = "xy") -> str:
        """
        Integrate a single HDF5 file.

        Parameters:
        -----------
        h5_file : str
            Path to the HDF5 file containing 2D diffraction data
        output_dir : str
            Directory to save the integrated data
        npt : int
            Number of points in the integrated pattern (default: 2048)
        unit : str
            Unit for the radial axis (default: "q_A^-1")
            Options: "q_A^-1", "q_nm^-1", "2th_deg", "2th_rad", "r_mm"
        output_format : str
            Output file format (default: "xy")
            Options: "xy", "chi", "dat"

        Returns:
        --------
        str : Path to the output file
        """
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

        print(f"\nProcessing: {os.path.basename(h5_file)}")

        # Read the 2D diffraction data from HDF5
        data = self._read_h5_data(h5_file)

        # Perform azimuthal integration
        print(f"  Integrating with {npt} points, unit={unit}")
        result = self.ai.integrate1d(
            data,
            npt=npt,
            mask=self.mask,
            unit=unit,
            method="splitpixel",  # Use split-pixel integration for accuracy
            error_model="poisson"  # Poisson error model for counting statistics
        )

        # Extract the integrated data
        q, intensity = result

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_integrated.{output_format}")

        # Save the integrated data
        self._save_data(q, intensity, output_file, unit, output_format)

        print(f"  Saved: {output_file}")
        return output_file

    def _read_h5_data(self, h5_file: str) -> np.ndarray:
        """
        Read 2D diffraction data from HDF5 file.

        Parameters:
        -----------
        h5_file : str
            Path to the HDF5 file

        Returns:
        --------
        np.ndarray : 2D diffraction pattern
        """
        with h5py.File(h5_file, 'r') as f:
            # Try common dataset paths for diffraction data
            common_paths = [
                'entry/data/data',
                'entry/instrument/detector/data',
                'data',
                'image',
                'diffraction'
            ]

            data = None
            for path in common_paths:
                if path in f:
                    data = f[path][...]
                    if data.ndim == 3:  # If 3D, take first frame
                        data = data[0]
                    break

            # If not found in common paths, search for first 2D dataset
            if data is None:
                def find_2d_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if obj.ndim == 2 or (obj.ndim == 3 and obj.shape[0] == 1):
                            return name
                    return None

                # Search for first 2D dataset
                for key in f.keys():
                    result = f[key].visititems(find_2d_dataset)
                    if result:
                        data = f[result][...]
                        if data.ndim == 3:
                            data = data[0]
                        break

                # Last resort: use first dataset found
                if data is None:
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][...]
                        if data.ndim == 3:
                            data = data[0]

            if data is None:
                raise ValueError(f"No suitable dataset found in {h5_file}")

            print(f"  Data shape: {data.shape}, dtype: {data.dtype}")
            print(f"  Intensity range: [{np.min(data):.1f}, {np.max(data):.1f}]")

            return data

    def _save_data(self, q: np.ndarray, intensity: np.ndarray,
                   output_file: str, unit: str, output_format: str):
        """
        Save the integrated data to file.

        Parameters:
        -----------
        q : np.ndarray
            Radial coordinates
        intensity : np.ndarray
            Integrated intensity
        output_file : str
            Path to output file
        unit : str
            Unit of the radial axis
        output_format : str
            Output format
        """
        if output_format == "xy":
            # Simple two-column format
            header = f"# Azimuthal integration\n# Unit: {unit}\n# Column 1: {unit}\n# Column 2: Intensity"
            np.savetxt(output_file, np.column_stack([q, intensity]),
                      header=header, fmt='%.6e')

        elif output_format == "chi":
            # GSAS-II chi format
            with open(output_file, 'w') as f:
                f.write(f"2-Theta Angle (Degrees)\n")
                f.write(f"Intensity\n")
                for q_val, int_val in zip(q, intensity):
                    f.write(f"{q_val:.6f} {int_val:.6f}\n")

        elif output_format == "dat":
            # Three-column format with errors
            errors = np.sqrt(np.maximum(intensity, 1))  # Poisson errors
            header = f"# Azimuthal integration\n# Unit: {unit}\n# Column 1: {unit}\n# Column 2: Intensity\n# Column 3: Error"
            np.savetxt(output_file, np.column_stack([q, intensity, errors]),
                      header=header, fmt='%.6e')

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def batch_process(self, h5_files: List[str], output_dir: str, **kwargs) -> List[str]:
        """
        Process multiple HDF5 files.

        Parameters:
        -----------
        h5_files : List[str]
            List of HDF5 file paths
        output_dir : str
            Directory to save the integrated data
        **kwargs : additional arguments passed to integrate_file

        Returns:
        --------
        List[str] : List of output file paths
        """
        output_files = []
        total = len(h5_files)

        print(f"\n{'='*60}")
        print(f"Batch processing {total} file(s)")
        print(f"{'='*60}")

        for i, h5_file in enumerate(h5_files, 1):
            print(f"\n[{i}/{total}]", end=" ")
            try:
                output_file = self.integrate_file(h5_file, output_dir, **kwargs)
                output_files.append(output_file)
            except Exception as e:
                print(f"  Error processing {h5_file}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Completed: {len(output_files)}/{total} files processed successfully")
        print(f"{'='*60}\n")

        return output_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="XRD Azimuthal Integration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python xrd_azimuthal_integration.py input.h5 -p calibration.poni -o output/

  # Process multiple files with mask
  python xrd_azimuthal_integration.py data/*.h5 -p cal.poni -m mask.npy -o results/

  # Specify integration parameters
  python xrd_azimuthal_integration.py input.h5 -p cal.poni -o output/ \\
    --npt 4096 --unit 2th_deg --format dat
        """
    )

    parser.add_argument('input_files', nargs='+',
                       help='HDF5 file(s) to process (supports wildcards)')
    parser.add_argument('-p', '--poni', required=True,
                       help='PONI calibration file')
    parser.add_argument('-m', '--mask', default=None,
                       help='Mask file (optional, .npy/.edf/.tif/.h5)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for integrated data')
    parser.add_argument('--npt', type=int, default=2048,
                       help='Number of points in integration (default: 2048)')
    parser.add_argument('--unit', default='q_A^-1',
                       choices=['q_A^-1', 'q_nm^-1', '2th_deg', '2th_rad', 'r_mm'],
                       help='Unit for radial axis (default: q_A^-1)')
    parser.add_argument('--format', default='xy',
                       choices=['xy', 'chi', 'dat'],
                       help='Output file format (default: xy)')

    args = parser.parse_args()

    # Expand wildcards in input files
    h5_files = []
    for pattern in args.input_files:
        matched = glob.glob(pattern)
        if matched:
            h5_files.extend(matched)
        else:
            # If no wildcard match, add as-is (might be a direct path)
            if os.path.exists(pattern):
                h5_files.append(pattern)

    if not h5_files:
        print("Error: No input files found!")
        sys.exit(1)

    # Filter to only HDF5 files
    h5_files = [f for f in h5_files if f.endswith(('.h5', '.hdf5', '.H5', '.HDF5'))]

    if not h5_files:
        print("Error: No HDF5 files found in input!")
        sys.exit(1)

    # Initialize integrator
    try:
        integrator = XRDAzimuthalIntegrator(args.poni, args.mask)
    except Exception as e:
        print(f"Error initializing integrator: {e}")
        sys.exit(1)

    # Process files
    try:
        integrator.batch_process(
            h5_files,
            args.output,
            npt=args.npt,
            unit=args.unit,
            output_format=args.format
        )
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
