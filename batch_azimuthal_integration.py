# -*- coding: utf-8 -*-
"""
Azimuthal Integration Module for Radial XRD Analysis
Handles selective azimuthal angle integration for diffraction rings

Created on 2025-11-15
@author: felicity
"""
import hdf5plugin
import numpy as np
import h5py
import os
from pathlib import Path
import glob
import pandas as pd
import pyFAI
import fabio


class XRDAzimuthalIntegrator:
    """
    Performs azimuthal integration on XRD data with selective angle ranges.

    Azimuthal angle definition:
    - 0° = Right horizontal direction (3 o'clock position)
    - Angles increase counter-clockwise
    - 90° = Top (12 o'clock)
    - 180° = Left (9 o'clock)
    - 270° = Bottom (6 o'clock)
    """

    def __init__(self, poni_path, mask_path=None):
        """
        Initialize the Azimuthal Integrator

        Parameters:
        -----------
        poni_path : str
            Path to the PONI calibration file
        mask_path : str, optional
            Path to the mask file (.edf or .npy)
        """
        self.poni_path = poni_path
        self.mask_path = mask_path

        # Load pyFAI azimuthal integrator
        self.ai = pyFAI.load(poni_path)

        # Load mask if provided
        self.mask = None
        if mask_path and os.path.exists(mask_path):
            if mask_path.endswith('.edf'):
                self.mask = fabio.open(mask_path).data
            elif mask_path.endswith('.npy'):
                self.mask = np.load(mask_path)
            else:
                print(f"Warning: Unsupported mask format. Supported: .edf, .npy")

    def integrate_azimuthal_range(self, data, azimuth_start, azimuth_end,
                                   npt=4000, unit='2th_deg'):
        """
        Integrate XRD data over a specific azimuthal angle range

        Parameters:
        -----------
        data : ndarray
            2D diffraction pattern
        azimuth_start : float
            Starting azimuthal angle in degrees (0-360)
        azimuth_end : float
            Ending azimuthal angle in degrees (0-360)
        npt : int
            Number of points in the integrated pattern
        unit : str
            Unit for integration ('2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm')

        Returns:
        --------
        x : ndarray
            X-axis values (2theta, q, or r)
        I : ndarray
            Integrated intensity
        """
        # Convert angles to radians for pyFAI (pyFAI uses radians)
        azim_start_rad = np.deg2rad(azimuth_start)
        azim_end_rad = np.deg2rad(azimuth_end)

        # Handle wrap-around case (e.g., 350° to 10°)
        if azimuth_end < azimuth_start:
            # Split into two ranges and combine
            # Range 1: azimuth_start to 360°
            result1 = self.ai.integrate1d(
                data,
                npt,
                unit=unit,
                mask=self.mask,
                azimuth_range=(azim_start_rad, np.deg2rad(360)),
                correctSolidAngle=True,
                polarization_factor=None,
                method='csr'
            )

            # Range 2: 0° to azimuth_end
            result2 = self.ai.integrate1d(
                data,
                npt,
                unit=unit,
                mask=self.mask,
                azimuth_range=(0, azim_end_rad),
                correctSolidAngle=True,
                polarization_factor=None,
                method='csr'
            )

            # Combine results (average the intensities)
            x = result1.radial
            I = (result1.intensity + result2.intensity) / 2.0
        else:
            # Normal case: single range
            result = self.ai.integrate1d(
                data,
                npt,
                unit=unit,
                mask=self.mask,
                azimuth_range=(azim_start_rad, azim_end_rad),
                correctSolidAngle=True,
                polarization_factor=None,
                method='csr'
            )
            x = result.radial
            I = result.intensity

        return x, I

    def integrate_multiple_sectors(self, data, sector_list, npt=4000, unit='2th_deg'):
        """
        Integrate multiple azimuthal sectors from the same diffraction pattern

        Parameters:
        -----------
        data : ndarray
            2D diffraction pattern
        sector_list : list of tuples
            List of (start_angle, end_angle, label) for each sector
            Example: [(0, 90, 'Sector_1'), (90, 180, 'Sector_2')]
        npt : int
            Number of points in the integrated pattern
        unit : str
            Unit for integration

        Returns:
        --------
        results : dict
            Dictionary with sector labels as keys and (x, I) tuples as values
        """
        results = {}

        for start, end, label in sector_list:
            x, I = self.integrate_azimuthal_range(data, start, end, npt, unit)
            results[label] = (x, I)

        return results

    def batch_integrate_h5(self, input_pattern, output_dir, azimuth_start,
                          azimuth_end, npt=4000, unit='2th_deg',
                          dataset_path='entry/data/data', sector_label=None):
        """
        Batch process multiple H5 files with azimuthal integration

        Parameters:
        -----------
        input_pattern : str
            File pattern for input H5 files (e.g., '/path/to/data/*.h5')
        output_dir : str
            Directory to save output CSV files
        azimuth_start : float
            Starting azimuthal angle in degrees
        azimuth_end : float
            Ending azimuthal angle in degrees
        npt : int
            Number of points in the integrated pattern
        unit : str
            Unit for integration
        dataset_path : str
            Path to dataset within H5 file
        sector_label : str, optional
            Label for this sector (used in filename)

        Returns:
        --------
        output_files : list
            List of generated output file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all matching H5 files
        h5_files = sorted(glob.glob(input_pattern))

        if len(h5_files) == 0:
            raise ValueError(f"No files found matching pattern: {input_pattern}")

        print(f"Found {len(h5_files)} H5 files to process")
        print(f"Azimuthal range: {azimuth_start}° to {azimuth_end}°")

        output_files = []

        for i, h5_file in enumerate(h5_files):
            print(f"Processing [{i+1}/{len(h5_files)}]: {os.path.basename(h5_file)}")

            try:
                # Read H5 file
                with h5py.File(h5_file, 'r') as f:
                    data = f[dataset_path][()]

                # Perform azimuthal integration
                x, I = self.integrate_azimuthal_range(
                    data, azimuth_start, azimuth_end, npt, unit
                )

                # Generate output filename
                base_name = os.path.splitext(os.path.basename(h5_file))[0]
                if sector_label:
                    output_name = f"{base_name}_{sector_label}_azim_{azimuth_start}_{azimuth_end}.csv"
                else:
                    output_name = f"{base_name}_azim_{azimuth_start}_{azimuth_end}.csv"

                output_path = os.path.join(output_dir, output_name)

                # Save to CSV
                df = pd.DataFrame({
                    unit: x,
                    'Intensity': I
                })
                df.to_csv(output_path, index=False)

                output_files.append(output_path)
                print(f"  ✓ Saved: {output_name}")

            except Exception as e:
                print(f"  ✗ Error processing {os.path.basename(h5_file)}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"Processed: {len(output_files)}/{len(h5_files)} files")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        return output_files

    def batch_integrate_multiple_sectors(self, input_pattern, output_dir,
                                         sector_list, npt=4000, unit='2th_deg',
                                         dataset_path='entry/data/data'):
        """
        Batch process multiple H5 files with multiple azimuthal sectors

        Parameters:
        -----------
        input_pattern : str
            File pattern for input H5 files
        output_dir : str
            Directory to save output CSV files
        sector_list : list of tuples
            List of (start_angle, end_angle, label) for each sector
        npt : int
            Number of points
        unit : str
            Unit for integration
        dataset_path : str
            Path to dataset within H5 file

        Returns:
        --------
        output_files : list
            List of generated output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        h5_files = sorted(glob.glob(input_pattern))

        if len(h5_files) == 0:
            raise ValueError(f"No files found matching pattern: {input_pattern}")

        print(f"Found {len(h5_files)} H5 files to process")
        print(f"Sectors to integrate:")
        for start, end, label in sector_list:
            print(f"  - {label}: {start}° to {end}°")

        output_files = []

        for i, h5_file in enumerate(h5_files):
            print(f"\nProcessing [{i+1}/{len(h5_files)}]: {os.path.basename(h5_file)}")

            try:
                # Read H5 file
                with h5py.File(h5_file, 'r') as f:
                    data = f[dataset_path][()]

                # Integrate all sectors
                results = self.integrate_multiple_sectors(data, sector_list, npt, unit)

                # Save each sector
                base_name = os.path.splitext(os.path.basename(h5_file))[0]

                for label, (x, I) in results.items():
                    output_name = f"{base_name}_{label}.csv"
                    output_path = os.path.join(output_dir, output_name)

                    df = pd.DataFrame({
                        unit: x,
                        'Intensity': I
                    })
                    df.to_csv(output_path, index=False)

                    output_files.append(output_path)
                    print(f"  ✓ Saved: {output_name}")

            except Exception as e:
                print(f"  ✗ Error processing {os.path.basename(h5_file)}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"Processed: {len(h5_files)} files")
        print(f"Generated: {len(output_files)} output files")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        return output_files


# ==================== Preset Sector Configurations ====================

def get_preset_sectors(preset_name):
    """
    Get preset sector configurations

    Parameters:
    -----------
    preset_name : str
        Name of the preset ('quadrants', 'octants', 'hemispheres', 'horizontal_vertical')

    Returns:
    --------
    sector_list : list of tuples
        List of (start_angle, end_angle, label)
    """
    presets = {
        'quadrants': [
            (0, 90, 'Q1_Right'),
            (90, 180, 'Q2_Top'),
            (180, 270, 'Q3_Left'),
            (270, 360, 'Q4_Bottom')
        ],
        'octants': [
            (0, 45, 'Oct1'),
            (45, 90, 'Oct2'),
            (90, 135, 'Oct3'),
            (135, 180, 'Oct4'),
            (180, 225, 'Oct5'),
            (225, 270, 'Oct6'),
            (270, 315, 'Oct7'),
            (315, 360, 'Oct8')
        ],
        'hemispheres': [
            (0, 180, 'Hemisphere_Right'),
            (180, 360, 'Hemisphere_Left')
        ],
        'horizontal_vertical': [
            (315, 45, 'Horizontal_Right'),
            (135, 225, 'Horizontal_Left'),
            (45, 135, 'Vertical_Top'),
            (225, 315, 'Vertical_Bottom')
        ]
    }

    return presets.get(preset_name, [])


# ==================== Example Usage ====================

if __name__ == "__main__":
    """
    Example usage of the XRDAzimuthalIntegrator
    """

    # Initialize integrator
    poni_file = r"D:\HEPS\ID31\test\using.poni"
    mask_file = r"D:\HEPS\ID31\test\use.edf"

    integrator = XRDAzimuthalIntegrator(poni_file, mask_file)

    # Example 1: Single sector integration
    input_pattern = r"D:\HEPS\ID31\test\input_dir\*.h5"
    output_dir = r"D:\HEPS\ID31\test\OUTPUT"

    integrator.batch_integrate_h5(
        input_pattern=input_pattern,
        output_dir=output_dir,
        azimuth_start=0,
        azimuth_end=90,
        npt=4000,
        unit='2th_deg',
        sector_label='Sector_0_90'
    )

    """
    # Example 2: Multiple sectors (quadrants)
    sector_list = get_preset_sectors('quadrants')

    integrator.batch_integrate_multiple_sectors(
        input_pattern=input_pattern,
        output_dir=output_dir,
        sector_list=sector_list,
        npt=4000,
        unit='2th_deg'
    )

    # Example 3: Custom sector
    custom_sectors = [
        (30, 60, 'Custom_30_60'),
        (120, 150, 'Custom_120_150')
    ]

    integrator.batch_integrate_multiple_sectors(
        input_pattern=input_pattern,
        output_dir=output_dir,
        sector_list=custom_sectors,
        npt=4000,
        unit='2th_deg'
    )
    """
