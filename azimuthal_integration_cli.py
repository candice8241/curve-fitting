# -*- coding: utf-8 -*-
"""
Azimuthal Integration CLI (Command Line Interface)
No GUI version for batch processing XRD data
Compatible with pyFAI 2025.3.0
"""

import os
import glob
import numpy as np
import pandas as pd


def integrate_sector(ai, data, mask, azim_start, azim_end, npt, unit):
    """
    Perform azimuthal integration for a single sector using pyFAI 2025.3.0

    Args:
        ai: pyFAI AzimuthalIntegrator instance
        data: 2D diffraction image data
        mask: Mask array (or None)
        azim_start: Start azimuth angle in degrees
        azim_end: End azimuth angle in degrees
        npt: Number of points for integration
        unit: Unit for radial axis (e.g., '2th_deg', 'q_A^-1')

    Returns:
        result: Integration result with radial and intensity attributes
    """
    # For pyFAI 2025.3.0, use the updated API
    result = ai.integrate1d(
        data,
        npt=npt,
        unit=unit,
        mask=mask,
        azimuth_range=(azim_start, azim_end),
        method='csr',  # Use CSR (Compressed Sparse Row) method for better compatibility
        correctSolidAngle=True  # Explicitly set solid angle correction
    )

    return result


def process_azimuthal_integration(poni_file, input_pattern, output_dir,
                                   dataset_path="entry/data/data",
                                   mask_file=None,
                                   sectors=None,
                                   npt=4000,
                                   unit='2th_deg',
                                   save_csv=True):
    """
    Main function to perform azimuthal integration on multiple files

    Args:
        poni_file: Path to PONI calibration file
        input_pattern: Glob pattern for input H5 files (e.g., "/path/to/*.h5")
        output_dir: Directory to save output files
        dataset_path: HDF5 dataset path (default: "entry/data/data")
        mask_file: Optional path to mask file (.edf or .npy)
        sectors: List of tuples (start_angle, end_angle, label).
                 If None, uses default single sector 0-90 degrees
        npt: Number of points for integration (default: 4000)
        unit: Unit for radial axis (default: '2th_deg')
        save_csv: Whether to save merged CSV files (default: True)

    Returns:
        Dictionary with sector labels as keys and output file lists as values
    """
    try:
        import pyFAI
        import h5py
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator as pyFAI_AI
    except ImportError:
        raise ImportError("pyFAI is required. Install it with: pip install pyFAI")

    print("="*70)
    print("🎯 Azimuthal Integration CLI")
    print("="*70)
    print(f"📦 pyFAI version: {pyFAI.version}")
    print()

    # Load calibration
    print(f"📁 Loading PONI file: {os.path.basename(poni_file)}")
    ai = pyFAI_AI()
    ai.load(poni_file)

    # Load mask if provided
    mask = None
    if mask_file:
        print(f"🎭 Loading mask file: {os.path.basename(mask_file)}")
        if mask_file.endswith('.edf'):
            import fabio
            mask = fabio.open(mask_file).data
        elif mask_file.endswith('.npy'):
            mask = np.load(mask_file)

    # Get input files
    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        raise ValueError(f"❌ No files found matching pattern: {input_pattern}")

    print(f"📂 Found {len(input_files)} input files")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print()

    # Default sectors if not provided
    if sectors is None:
        sectors = [(0.0, 90.0, "Sector_0-90")]

    print(f"📊 Processing {len(sectors)} sector(s):")
    for start, end, label in sectors:
        print(f"   - {label}: {start}° to {end}°")
    print()

    # Dictionary to store all outputs
    all_outputs = {}

    # Process each sector
    for azim_start, azim_end, sector_label in sectors:
        print(f"{'='*70}")
        print(f"🔄 Processing sector: {sector_label} ({azim_start}° - {azim_end}°)")
        print(f"{'='*70}")

        # Storage for CSV output
        csv_data = {}
        output_files = []

        # Process each file
        for idx, h5_file in enumerate(input_files):
            filename = os.path.basename(h5_file)
            print(f"   [{idx+1}/{len(input_files)}] {filename}")

            # Read data from H5
            with h5py.File(h5_file, 'r') as f:
                data = f[dataset_path][()]

            # Perform azimuthal integration
            result = integrate_sector(ai, data, mask, azim_start, azim_end, npt, unit)

            # Save individual .xy file
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_{sector_label}.xy"
            output_path = os.path.join(output_dir, output_filename)

            np.savetxt(output_path, np.column_stack([result.radial, result.intensity]),
                      header=f"{unit}  Intensity", comments='#')
            output_files.append(output_path)

            # Store for CSV
            csv_data[filename] = {
                'x': result.radial,
                'y': result.intensity
            }

        # Generate CSV if enabled
        if save_csv and csv_data:
            csv_filename = f"azimuthal_integration_{sector_label}.csv"
            csv_path = os.path.join(output_dir, csv_filename)

            # Get reference x-axis (assuming all have same x)
            first_key = list(csv_data.keys())[0]
            x_values = csv_data[first_key]['x']

            # Build dataframe
            df_dict = {unit: x_values}

            for filename, data_dict in csv_data.items():
                # Use filename as column header
                base_name = os.path.splitext(filename)[0]
                df_dict[base_name] = data_dict['y']

            df = pd.DataFrame(df_dict)
            df.to_csv(csv_path, index=False)
            output_files.append(csv_path)
            print(f"   💾 CSV saved: {csv_filename}")

        all_outputs[sector_label] = output_files
        print(f"   ✅ Sector {sector_label} complete! Generated {len(output_files)} files")
        print()

    print("="*70)
    print("✨ All integrations completed successfully!")
    print("="*70)

    return all_outputs


def main():
    """
    Main function - Configure your paths and parameters here
    """

    # ==================== 配置区域 / CONFIGURATION ====================

    # 1. PONI 校准文件路径 / PONI calibration file path
    poni_file = r"D:\HEPS\ID31\calibration\sample.poni"

    # 2. 输入文件模式 / Input file pattern (glob pattern for H5 files)
    input_pattern = r"D:\HEPS\ID31\data\*.h5"

    # 3. 输出目录 / Output directory
    output_dir = r"D:\HEPS\ID31\output\azimuthal_integration"

    # 4. HDF5数据集路径 / HDF5 dataset path
    dataset_path = "entry/data/data"

    # 5. 掩膜文件（可选）/ Mask file (optional, set to None if not used)
    mask_file = None  # or r"D:\HEPS\ID31\mask\mask.edf"

    # 6. 积分参数 / Integration parameters
    npt = 4000  # Number of points
    unit = '2th_deg'  # Options: '2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'

    # 7. 是否保存CSV / Save CSV files
    save_csv = True

    # 8. 定义扇区 / Define sectors (list of tuples: (start_angle, end_angle, label))
    #
    # 示例 / Examples:
    #
    # 单扇区 / Single sector:
    # sectors = [(0, 90, "Sector_0-90")]
    #
    # 四象限 / Four quadrants:
    # sectors = [
    #     (0, 90, "Q1_0-90"),
    #     (90, 180, "Q2_90-180"),
    #     (180, 270, "Q3_180-270"),
    #     (270, 360, "Q4_270-360")
    # ]
    #
    # 八分区 / Eight octants:
    # sectors = [
    #     (0, 45, "Oct1_0-45"),
    #     (45, 90, "Oct2_45-90"),
    #     (90, 135, "Oct3_90-135"),
    #     (135, 180, "Oct4_135-180"),
    #     (180, 225, "Oct5_180-225"),
    #     (225, 270, "Oct6_225-270"),
    #     (270, 315, "Oct7_270-315"),
    #     (315, 360, "Oct8_315-360")
    # ]
    #
    # 自定义 / Custom:
    sectors = [
        (0, 90, "Right"),
        (90, 180, "Top"),
        (180, 270, "Left"),
        (270, 360, "Bottom")
    ]

    # ==================== 运行积分 / RUN INTEGRATION ====================

    try:
        results = process_azimuthal_integration(
            poni_file=poni_file,
            input_pattern=input_pattern,
            output_dir=output_dir,
            dataset_path=dataset_path,
            mask_file=mask_file,
            sectors=sectors,
            npt=npt,
            unit=unit,
            save_csv=save_csv
        )

        # 打印结果摘要 / Print summary
        print("\n📋 Summary of generated files:")
        for sector_label, file_list in results.items():
            print(f"\n   {sector_label}:")
            for file_path in file_list[:3]:  # Show first 3 files
                print(f"      - {os.path.basename(file_path)}")
            if len(file_list) > 3:
                print(f"      ... and {len(file_list) - 3} more files")

        print(f"\n✅ All files saved to: {output_dir}")

    except Exception as e:
        import traceback
        print("\n❌ Error occurred:")
        print(str(e))
        print("\nTraceback:")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
