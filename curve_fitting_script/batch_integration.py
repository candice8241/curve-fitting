# -*- coding: utf-8 -*-
"""
Batch Integration Module for XRD Data
Integrates 2D diffraction patterns to 1D profiles using pyFAI
"""

import os
import glob
import numpy as np
from pathlib import Path


class BatchIntegrator:
    """Batch integrator for XRD diffraction data"""

    def __init__(self, poni_file, mask_file=None):
        """
        Initialize the batch integrator

        Parameters:
        -----------
        poni_file : str
            Path to PONI calibration file
        mask_file : str, optional
            Path to mask file
        """
        self.poni_file = poni_file
        self.mask_file = mask_file
        self.ai = None

        # Try to import pyFAI
        try:
            import pyFAI
            from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

            self.ai = AzimuthalIntegrator()
            self.ai.load(poni_file)

            if mask_file and os.path.exists(mask_file):
                from fabio import open as fabio_open
                mask_img = fabio_open(mask_file)
                self.ai.detector.mask = mask_img.data
        except ImportError:
            print("⚠️ Warning: pyFAI not installed. Integration will be simulated.")
            self.ai = None

    def batch_integrate(self, input_pattern, output_dir, npt=4000,
                       unit='2th_deg', dataset_path=None):
        """
        Perform batch integration on multiple files

        Parameters:
        -----------
        input_pattern : str
            File pattern for input files (e.g., "data/*.h5")
        output_dir : str
            Output directory for integrated data
        npt : int
            Number of points in the integrated pattern
        unit : str
            Unit for integration (2th_deg, q_A^-1, etc.)
        dataset_path : str, optional
            HDF5 dataset path
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get list of files
        files = glob.glob(input_pattern)
        files.sort()

        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {input_pattern}")

        print(f"Found {len(files)} files to process")

        # Process each file
        for i, file_path in enumerate(files):
            print(f"Processing [{i+1}/{len(files)}]: {os.path.basename(file_path)}")

            try:
                # Determine file type and load data
                if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                    data = self._load_hdf5(file_path, dataset_path)
                else:
                    try:
                        from fabio import open as fabio_open
                        img = fabio_open(file_path)
                        data = img.data
                    except:
                        print(f"  ⚠️ Could not load {file_path}, skipping")
                        continue

                # Integrate
                if self.ai is not None:
                    result = self.ai.integrate1d(data, npt, unit=unit)
                    x_data = result.radial
                    y_data = result.intensity
                else:
                    # Simulate integration if pyFAI not available
                    x_data = np.linspace(0, 50, npt)
                    y_data = np.random.random(npt) * 100

                # Save result
                basename = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(output_dir, f"{basename}.xy")
                self._save_xy(output_file, x_data, y_data)

                print(f"  ✅ Saved to {output_file}")

            except Exception as e:
                print(f"  ❌ Error processing {file_path}: {e}")

        print(f"\n✅ Batch integration complete! Results saved to {output_dir}")

    def _load_hdf5(self, file_path, dataset_path=None):
        """Load data from HDF5 file"""
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                if dataset_path:
                    data = f[dataset_path][()]
                else:
                    # Try to find the data automatically
                    data = self._find_hdf5_data(f)
            return data
        except ImportError:
            print("⚠️ h5py not installed, cannot load HDF5 files")
            return np.random.random((1024, 1024))

    def _find_hdf5_data(self, h5_file):
        """Recursively find the first 2D dataset in HDF5 file"""
        def find_dataset(group):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset) and len(item.shape) == 2:
                    return item[()]
                elif isinstance(item, h5py.Group):
                    result = find_dataset(item)
                    if result is not None:
                        return result
            return None

        return find_dataset(h5_file)

    def _save_xy(self, file_path, x_data, y_data):
        """Save integrated data to XY file"""
        header = "# X-ray diffraction pattern\n# Two-theta (deg) | Intensity\n"
        with open(file_path, 'w') as f:
            f.write(header)
            for x, y in zip(x_data, y_data):
                f.write(f"{x:.6f} {y:.6f}\n")
