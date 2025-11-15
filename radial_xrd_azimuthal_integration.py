#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Radial XRD Azimuthal Integration Module
Performs azimuthal integration for radial diffraction analysis without GUI
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd


def azimuthal_integration(
    poni_file,
    input_pattern,
    output_dir,
    sectors,
    dataset_path="entry/data/data",
    mask_file=None,
    npt=4000,
    unit='2th_deg',
    save_csv=True,
    verbose=True
):
    """
    Perform azimuthal integration for radial XRD data

    Args:
        poni_file (str): Path to PONI calibration file
        input_pattern (str): Glob pattern for input H5 files (e.g., "data/*.h5")
        output_dir (str): Directory to save output files
        sectors (list): List of tuples (start_angle, end_angle, label)
                       e.g., [(0, 90, "Q1"), (90, 180, "Q2")]
        dataset_path (str): HDF5 dataset path (default: "entry/data/data")
        mask_file (str, optional): Path to mask file (.edf or .npy)
        npt (int): Number of points for integration (default: 4000)
        unit (str): Unit for integration (default: '2th_deg')
                   Options: '2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'
        save_csv (bool): Whether to save merged CSV files (default: True)
        verbose (bool): Print progress messages (default: True)

    Returns:
        dict: Dictionary mapping sector labels to output file paths
    """
    try:
        import pyFAI
        import h5py
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    except ImportError as e:
        raise ImportError(
            "pyFAI and h5py are required. Install with:\n"
            "  pip install pyFAI h5py"
        ) from e

    # Validate inputs
    if not os.path.exists(poni_file):
        raise FileNotFoundError(f"PONI file not found: {poni_file}")

    if not sectors:
        raise ValueError("At least one sector must be specified")

    # Load calibration
    if verbose:
        print(f"📁 Loading calibration from: {poni_file}")
    ai = AzimuthalIntegrator()
    ai.load(poni_file)

    # Load mask if provided
    mask = None
    if mask_file:
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask file not found: {mask_file}")

        if verbose:
            print(f"🎭 Loading mask from: {mask_file}")

        if mask_file.endswith('.edf'):
            import fabio
            mask = fabio.open(mask_file).data
        elif mask_file.endswith('.npy'):
            mask = np.load(mask_file)
        else:
            raise ValueError(f"Unsupported mask format: {mask_file}")

    # Get input files
    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        raise ValueError(f"No files found matching pattern: {input_pattern}")

    if verbose:
        print(f"📊 Found {len(input_files)} input files")
        print(f"📐 Processing {len(sectors)} sector(s)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store all output file paths
    all_outputs = {}

    # Process each sector
    for sector_idx, (azim_start, azim_end, sector_label) in enumerate(sectors):
        if verbose:
            print(f"\n🔄 Processing sector {sector_idx + 1}/{len(sectors)}: {sector_label}")
            print(f"   Azimuthal range: {azim_start}° to {azim_end}°")

        # Storage for CSV output
        csv_data = {}
        sector_output_files = []

        # Process each file
        for idx, h5_file in enumerate(input_files):
            filename = os.path.basename(h5_file)

            if verbose and (idx + 1) % max(1, len(input_files) // 10) == 0:
                print(f"   [{idx + 1}/{len(input_files)}] Processing {filename}")

            try:
                # Read data from H5
                with h5py.File(h5_file, 'r') as f:
                    if dataset_path not in f:
                        print(f"   ⚠️  Warning: Dataset '{dataset_path}' not found in {filename}, skipping")
                        continue
                    data = f[dataset_path][()]

                # Perform azimuthal integration with sector
                result = ai.integrate1d(
                    data,
                    npt=npt,
                    unit=unit,
                    mask=mask,
                    azimuth_range=(azim_start, azim_end)
                )

                # Save individual .xy file
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_{sector_label}.xy"
                output_path = os.path.join(output_dir, output_filename)

                np.savetxt(
                    output_path,
                    np.column_stack([result.radial, result.intensity]),
                    header=f"{unit}  Intensity",
                    comments='# '
                )
                sector_output_files.append(output_path)

                # Store for CSV
                csv_data[filename] = {
                    'x': result.radial,
                    'y': result.intensity
                }

            except Exception as e:
                print(f"   ❌ Error processing {filename}: {str(e)}")
                continue

        # Generate CSV if enabled
        if save_csv and csv_data:
            csv_filename = f"azimuthal_integration_{sector_label}.csv"
            csv_path = os.path.join(output_dir, csv_filename)

            # Get reference x-axis (assuming all have same x)
            first_key = list(csv_data.keys())[0]
            x_values = csv_data[first_key]['x']

            # Build dataframe
            df_dict = {unit: x_values}
            for filename, data in csv_data.items():
                base_name = os.path.splitext(filename)[0]
                df_dict[base_name] = data['y']

            df = pd.DataFrame(df_dict)
            df.to_csv(csv_path, index=False)
            sector_output_files.append(csv_path)

            if verbose:
                print(f"   💾 CSV saved: {csv_filename}")

        all_outputs[sector_label] = sector_output_files

        if verbose:
            print(f"   ✅ Generated {len(sector_output_files)} files for {sector_label}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"✨ Integration complete!")
        total_files = sum(len(files) for files in all_outputs.values())
        print(f"📊 Generated {total_files} files total")
        print(f"📁 Output directory: {output_dir}")
        print(f"{'='*60}")

    return all_outputs


def get_preset_sectors(preset_name):
    """
    Get predefined sector configurations

    Args:
        preset_name (str): Name of preset
                          Options: 'quadrants', 'octants', 'hemispheres', 'horizontal_vertical'

    Returns:
        list: List of (start, end, label) tuples
    """
    presets = {
        'quadrants': [
            (0, 90, "Q1_0-90"),
            (90, 180, "Q2_90-180"),
            (180, 270, "Q3_180-270"),
            (270, 360, "Q4_270-360")
        ],
        'octants': [
            (0, 45, "Oct1_0-45"),
            (45, 90, "Oct2_45-90"),
            (90, 135, "Oct3_90-135"),
            (135, 180, "Oct4_135-180"),
            (180, 225, "Oct5_180-225"),
            (225, 270, "Oct6_225-270"),
            (270, 315, "Oct7_270-315"),
            (315, 360, "Oct8_315-360")
        ],
        'hemispheres': [
            (0, 180, "Right_Hemisphere"),
            (180, 360, "Left_Hemisphere")
        ],
        'horizontal_vertical': [
            (0, 90, "Right"),
            (90, 180, "Top"),
            (180, 270, "Left"),
            (270, 360, "Bottom")
        ]
    }

    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: {preset_name}\n"
            f"Available presets: {', '.join(presets.keys())}"
        )

    return presets[preset_name]


def main():
    """
    Main function for command-line usage

    Example usage:
        # Single sector
        python radial_xrd_azimuthal_integration.py \\
            --poni calibration.poni \\
            --input "data/*.h5" \\
            --output results/ \\
            --sector 0 90 "Sector1"

        # Multiple sectors using preset
        python radial_xrd_azimuthal_integration.py \\
            --poni calibration.poni \\
            --input "data/*.h5" \\
            --output results/ \\
            --preset quadrants

        # Custom multiple sectors
        python radial_xrd_azimuthal_integration.py \\
            --poni calibration.poni \\
            --input "data/*.h5" \\
            --output results/ \\
            --sector 0 90 "Q1" \\
            --sector 90 180 "Q2" \\
            --sector 180 270 "Q3" \\
            --sector 270 360 "Q4"
    """
    parser = argparse.ArgumentParser(
        description='Azimuthal Integration for Radial XRD Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sector integration
  %(prog)s --poni cal.poni --input "data/*.h5" --output results/ --sector 0 90 "Q1"

  # Quadrants preset
  %(prog)s --poni cal.poni --input "data/*.h5" --output results/ --preset quadrants

  # Custom sectors with mask
  %(prog)s --poni cal.poni --input "data/*.h5" --output results/ \\
           --mask mask.edf --sector 0 90 "Q1" --sector 90 180 "Q2"

Azimuthal Angle Reference:
  0° = Right (→)  |  90° = Top (↑)  |  180° = Left (←)  |  270° = Bottom (↓)
  Counter-clockwise rotation from right horizontal
        """
    )

    # Required arguments
    parser.add_argument('--poni', required=True,
                       help='Path to PONI calibration file')
    parser.add_argument('--input', required=True,
                       help='Glob pattern for input H5 files (e.g., "data/*.h5")')
    parser.add_argument('--output', required=True,
                       help='Output directory for results')

    # Sector definition (mutually exclusive with preset)
    sector_group = parser.add_mutually_exclusive_group(required=True)
    sector_group.add_argument('--sector', action='append', nargs=3,
                             metavar=('START', 'END', 'LABEL'),
                             help='Define sector: start_angle end_angle label (can be repeated)')
    sector_group.add_argument('--preset', choices=['quadrants', 'octants', 'hemispheres', 'horizontal_vertical'],
                             help='Use predefined sector preset')

    # Optional arguments
    parser.add_argument('--mask', default=None,
                       help='Path to mask file (.edf or .npy)')
    parser.add_argument('--dataset', default='entry/data/data',
                       help='HDF5 dataset path (default: entry/data/data)')
    parser.add_argument('--npt', type=int, default=4000,
                       help='Number of points for integration (default: 4000)')
    parser.add_argument('--unit', default='2th_deg',
                       choices=['2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'],
                       help='Unit for integration (default: 2th_deg)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV output (only save .xy files)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    # Build sectors list
    if args.sector:
        # Custom sectors from command line
        sectors = []
        for sector_def in args.sector:
            try:
                start = float(sector_def[0])
                end = float(sector_def[1])
                label = sector_def[2]
                sectors.append((start, end, label))
            except (ValueError, IndexError) as e:
                parser.error(f"Invalid sector definition: {sector_def}\n{str(e)}")
    else:
        # Use preset
        sectors = get_preset_sectors(args.preset)

    # Run integration
    try:
        azimuthal_integration(
            poni_file=args.poni,
            input_pattern=args.input,
            output_dir=args.output,
            sectors=sectors,
            dataset_path=args.dataset,
            mask_file=args.mask,
            npt=args.npt,
            unit=args.unit,
            save_csv=not args.no_csv,
            verbose=not args.quiet
        )
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {str(e)}")
        if not args.quiet:
            print("\nDetails:")
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
