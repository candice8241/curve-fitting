#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Azimuthal Integration Example Script
快速开始示例 - Radial XRD 方位角积分

这个脚本演示了如何使用 AzimuthalIntegrator 进行方位角积分
"""

from azimuthal_integration import AzimuthalIntegrator, get_preset_sectors
import os

def example_single_sector():
    """
    示例 1: 单扇区积分
    对一系列 H5 文件进行单个扇区的方位角积分
    """
    print("\n" + "="*70)
    print("示例 1: 单扇区积分 (0° to 90°)")
    print("="*70 + "\n")

    # ===== 配置参数 =====
    poni_file = "path/to/your/calibration.poni"  # 替换为你的 PONI 文件路径
    mask_file = "path/to/your/mask.edf"          # 替换为你的 mask 文件路径（可选）
    input_pattern = "/path/to/data/*.h5"         # 替换为你的 H5 文件路径
    output_dir = "/path/to/output/single_sector" # 替换为输出目录

    # 方位角参数
    azimuth_start = 0      # 起始角度（右侧水平方向）
    azimuth_end = 90       # 结束角度（顶部）
    sector_label = "Right_Quadrant"  # 扇区标签
    npt = 4000            # 积分点数
    unit = '2th_deg'      # 单位

    # ===== 初始化积分器 =====
    print(f"📁 PONI file: {poni_file}")
    print(f"🎭 Mask file: {mask_file}")
    print(f"📐 Azimuthal range: {azimuth_start}° to {azimuth_end}°")
    print(f"🏷️  Sector label: {sector_label}\n")

    integrator = AzimuthalIntegrator(
        poni_path=poni_file,
        mask_path=mask_file  # 如果不需要 mask，设置为 None
    )

    # ===== 运行批量积分 =====
    output_files = integrator.batch_integrate_h5(
        input_pattern=input_pattern,
        output_dir=output_dir,
        azimuth_start=azimuth_start,
        azimuth_end=azimuth_end,
        npt=npt,
        unit=unit,
        dataset_path='entry/data/data',
        sector_label=sector_label
    )

    print(f"\n✅ 完成！生成了 {len(output_files)} 个文件")
    print(f"📁 输出目录: {output_dir}\n")


def example_quadrants():
    """
    示例 2: 四象限积分
    将衍射环分为四个象限进行积分
    """
    print("\n" + "="*70)
    print("示例 2: 四象限积分")
    print("="*70 + "\n")

    # ===== 配置参数 =====
    poni_file = "path/to/your/calibration.poni"
    mask_file = None  # 这个示例不使用 mask
    input_pattern = "/path/to/data/*.h5"
    output_dir = "/path/to/output/quadrants"
    npt = 4000
    unit = '2th_deg'

    # ===== 获取四象限预设 =====
    sector_list = get_preset_sectors('quadrants')
    # sector_list = [
    #     (0, 90, 'Q1_Right'),
    #     (90, 180, 'Q2_Top'),
    #     (180, 270, 'Q3_Left'),
    #     (270, 360, 'Q4_Bottom')
    # ]

    print(f"📁 PONI file: {poni_file}")
    print(f"📊 扇区配置: 四象限")
    for start, end, label in sector_list:
        print(f"   - {label}: {start}° to {end}°")
    print()

    # ===== 初始化积分器 =====
    integrator = AzimuthalIntegrator(
        poni_path=poni_file,
        mask_path=mask_file
    )

    # ===== 运行多扇区批量积分 =====
    output_files = integrator.batch_integrate_multiple_sectors(
        input_pattern=input_pattern,
        output_dir=output_dir,
        sector_list=sector_list,
        npt=npt,
        unit=unit,
        dataset_path='entry/data/data'
    )

    print(f"\n✅ 完成！生成了 {len(output_files)} 个文件")
    print(f"📁 输出目录: {output_dir}\n")


def example_custom_sectors():
    """
    示例 3: 自定义扇区积分
    根据需要定义任意角度范围的扇区
    """
    print("\n" + "="*70)
    print("示例 3: 自定义扇区积分")
    print("="*70 + "\n")

    # ===== 配置参数 =====
    poni_file = "path/to/your/calibration.poni"
    mask_file = "path/to/your/mask.edf"
    input_pattern = "/path/to/data/*.h5"
    output_dir = "/path/to/output/custom"
    npt = 4000
    unit = 'q_A^-1'  # 使用 q 单位

    # ===== 定义自定义扇区 =====
    # 格式: (起始角度, 结束角度, 标签)
    custom_sectors = [
        (0, 30, 'Sector_A'),       # 右侧 30°
        (90, 120, 'Sector_B'),     # 顶部 30°
        (180, 210, 'Sector_C'),    # 左侧 30°
        (270, 300, 'Sector_D'),    # 底部 30°
        (315, 45, 'Diagonal_1'),   # 对角线 1（跨越 0°）
        (135, 225, 'Diagonal_2')   # 对角线 2
    ]

    print(f"📁 PONI file: {poni_file}")
    print(f"🎭 Mask file: {mask_file}")
    print(f"📊 自定义扇区:")
    for start, end, label in custom_sectors:
        print(f"   - {label}: {start}° to {end}°")
    print()

    # ===== 初始化积分器 =====
    integrator = AzimuthalIntegrator(
        poni_path=poni_file,
        mask_path=mask_file
    )

    # ===== 运行多扇区批量积分 =====
    output_files = integrator.batch_integrate_multiple_sectors(
        input_pattern=input_pattern,
        output_dir=output_dir,
        sector_list=custom_sectors,
        npt=npt,
        unit=unit,
        dataset_path='entry/data/data'
    )

    print(f"\n✅ 完成！生成了 {len(output_files)} 个文件")
    print(f"📁 输出目录: {output_dir}\n")


def example_all_presets():
    """
    示例 4: 所有预设配置
    展示所有可用的预设扇区配置
    """
    print("\n" + "="*70)
    print("示例 4: 所有预设配置")
    print("="*70 + "\n")

    presets = ['quadrants', 'octants', 'hemispheres', 'horizontal_vertical']

    for preset_name in presets:
        sector_list = get_preset_sectors(preset_name)
        print(f"\n📊 Preset: {preset_name}")
        print(f"   扇区数量: {len(sector_list)}")
        for start, end, label in sector_list:
            print(f"   - {label}: {start}° to {end}°")


def example_single_file():
    """
    示例 5: 单个文件积分
    对单个 H5 文件进行积分（不使用批量处理）
    """
    print("\n" + "="*70)
    print("示例 5: 单个文件积分")
    print("="*70 + "\n")

    import h5py
    import pandas as pd

    # ===== 配置参数 =====
    poni_file = "path/to/your/calibration.poni"
    mask_file = None
    h5_file = "path/to/single_data.h5"
    output_file = "output_single.csv"

    azimuth_start = 45
    azimuth_end = 135
    npt = 4000
    unit = '2th_deg'

    print(f"📁 H5 file: {h5_file}")
    print(f"📐 Azimuthal range: {azimuth_start}° to {azimuth_end}°\n")

    # ===== 初始化积分器 =====
    integrator = AzimuthalIntegrator(
        poni_path=poni_file,
        mask_path=mask_file
    )

    # ===== 读取 H5 文件 =====
    with h5py.File(h5_file, 'r') as f:
        data = f['entry/data/data'][()]

    # ===== 进行积分 =====
    x, intensity = integrator.integrate_azimuthal_range(
        data=data,
        azimuth_start=azimuth_start,
        azimuth_end=azimuth_end,
        npt=npt,
        unit=unit
    )

    # ===== 保存结果 =====
    df = pd.DataFrame({unit: x, 'Intensity': intensity})
    df.to_csv(output_file, index=False)

    print(f"✅ 完成！")
    print(f"📄 输出文件: {output_file}\n")


def main():
    """
    主函数 - 选择要运行的示例
    """
    print("\n" + "="*70)
    print("🎯 Azimuthal Integration Example Scripts")
    print("="*70)
    print("\n请选择要运行的示例:")
    print("1. 单扇区积分 (0° to 90°)")
    print("2. 四象限积分")
    print("3. 自定义扇区积分")
    print("4. 查看所有预设配置")
    print("5. 单个文件积分")
    print("0. 退出")

    while True:
        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == '1':
            example_single_sector()
        elif choice == '2':
            example_quadrants()
        elif choice == '3':
            example_custom_sectors()
        elif choice == '4':
            example_all_presets()
        elif choice == '5':
            example_single_file()
        elif choice == '0':
            print("\n再见！👋\n")
            break
        else:
            print("❌ 无效选项，请重新输入")


if __name__ == "__main__":
    # ===== 重要提示 =====
    print("\n" + "⚠️ " * 35)
    print("重要提示:")
    print("在运行示例之前，请先修改脚本中的文件路径:")
    print("  - poni_file: PONI 校准文件路径")
    print("  - mask_file: Mask 文件路径（可选）")
    print("  - input_pattern: H5 输入文件路径")
    print("  - output_dir: 输出目录路径")
    print("⚠️ " * 35 + "\n")

    # 运行主函数
    main()

    # 或者直接运行某个示例:
    # example_single_sector()
    # example_quadrants()
    # example_custom_sectors()
    # example_all_presets()
    # example_single_file()
