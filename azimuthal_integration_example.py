#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Azimuthal Integration Usage Examples
示例展示如何使用方位角积分功能
"""

from radial_xrd_azimuthal_integration import azimuthal_integration, get_preset_sectors


# ==============================================================================
# 示例 1: 单个扇区积分
# ==============================================================================
def example_single_sector():
    """单个扇区的方位角积分"""
    print("=" * 60)
    print("示例 1: 单个扇区积分")
    print("=" * 60)

    # 定义输入参数
    poni_file = "path/to/calibration.poni"          # PONI标定文件路径
    input_pattern = "path/to/data/*.h5"             # 输入H5文件通配符
    output_dir = "path/to/output/single_sector"     # 输出目录

    # 定义单个扇区: 0度到90度，标签为 "Q1"
    sectors = [(0, 90, "Q1")]

    # 运行积分
    results = azimuthal_integration(
        poni_file=poni_file,
        input_pattern=input_pattern,
        output_dir=output_dir,
        sectors=sectors,
        npt=4000,                    # 积分点数
        unit='2th_deg',              # 单位：2theta角度
        save_csv=True,               # 保存CSV文件
        verbose=True                 # 显示进度信息
    )

    print(f"\n输出文件: {results}")


# ==============================================================================
# 示例 2: 使用预设模板 - 四象限
# ==============================================================================
def example_quadrants_preset():
    """使用四象限预设模板"""
    print("=" * 60)
    print("示例 2: 四象限积分")
    print("=" * 60)

    # 定义输入参数
    poni_file = "path/to/calibration.poni"
    input_pattern = "path/to/data/*.h5"
    output_dir = "path/to/output/quadrants"

    # 获取四象限预设
    sectors = get_preset_sectors('quadrants')
    # 结果: [(0, 90, "Q1_0-90"), (90, 180, "Q2_90-180"),
    #        (180, 270, "Q3_180-270"), (270, 360, "Q4_270-360")]

    print(f"使用预设扇区: {sectors}")

    # 运行积分
    results = azimuthal_integration(
        poni_file=poni_file,
        input_pattern=input_pattern,
        output_dir=output_dir,
        sectors=sectors,
        unit='q_A^-1',               # 单位：q (Å^-1)
        save_csv=True,
        verbose=True
    )

    print(f"\n输出文件: {results}")


# ==============================================================================
# 示例 3: 自定义多个扇区
# ==============================================================================
def example_custom_sectors():
    """自定义多个扇区"""
    print("=" * 60)
    print("示例 3: 自定义多扇区积分")
    print("=" * 60)

    # 定义输入参数
    poni_file = "path/to/calibration.poni"
    input_pattern = "path/to/data/*.h5"
    output_dir = "path/to/output/custom"

    # 自定义扇区列表
    sectors = [
        (0, 45, "Sector_A"),        # 0-45度
        (45, 135, "Sector_B"),      # 45-135度
        (135, 225, "Sector_C"),     # 135-225度
        (225, 315, "Sector_D"),     # 225-315度
    ]

    # 运行积分（带掩膜文件）
    results = azimuthal_integration(
        poni_file=poni_file,
        input_pattern=input_pattern,
        output_dir=output_dir,
        sectors=sectors,
        mask_file="path/to/mask.edf",   # 可选：掩膜文件
        dataset_path="entry/data/data",  # HDF5数据集路径
        npt=5000,
        unit='2th_deg',
        save_csv=True,
        verbose=True
    )

    print(f"\n输出文件: {results}")


# ==============================================================================
# 示例 4: 八分区积分
# ==============================================================================
def example_octants():
    """使用八分区预设"""
    print("=" * 60)
    print("示例 4: 八分区积分")
    print("=" * 60)

    poni_file = "path/to/calibration.poni"
    input_pattern = "path/to/data/*.h5"
    output_dir = "path/to/output/octants"

    # 获取八分区预设
    sectors = get_preset_sectors('octants')

    results = azimuthal_integration(
        poni_file=poni_file,
        input_pattern=input_pattern,
        output_dir=output_dir,
        sectors=sectors,
        npt=4000,
        unit='2th_deg',
        save_csv=True,
        verbose=True
    )

    print(f"\n输出文件: {results}")


# ==============================================================================
# 示例 5: 半球积分
# ==============================================================================
def example_hemispheres():
    """半球积分（左右半球）"""
    print("=" * 60)
    print("示例 5: 半球积分")
    print("=" * 60)

    poni_file = "path/to/calibration.poni"
    input_pattern = "path/to/data/*.h5"
    output_dir = "path/to/output/hemispheres"

    # 获取半球预设
    sectors = get_preset_sectors('hemispheres')
    # 结果: [(0, 180, "Right_Hemisphere"), (180, 360, "Left_Hemisphere")]

    results = azimuthal_integration(
        poni_file=poni_file,
        input_pattern=input_pattern,
        output_dir=output_dir,
        sectors=sectors,
        npt=4000,
        unit='q_nm^-1',              # 单位：q (nm^-1)
        save_csv=True,
        verbose=True
    )

    print(f"\n输出文件: {results}")


# ==============================================================================
# 示例 6: 实际完整工作流程
# ==============================================================================
def example_complete_workflow():
    """完整的工作流程示例"""
    print("=" * 60)
    print("示例 6: 完整工作流程")
    print("=" * 60)

    # 步骤 1: 定义所有路径
    poni_file = "/data/experiment/calibration.poni"
    input_pattern = "/data/experiment/raw_data/sample_*.h5"
    output_dir = "/data/experiment/results/azimuthal_integration"
    mask_file = "/data/experiment/mask.npy"

    # 步骤 2: 选择积分方案
    # 方案A: 使用四象限
    sectors_quadrants = get_preset_sectors('quadrants')

    # 方案B: 自定义特定角度范围
    sectors_custom = [
        (0, 30, "Rolling_Direction"),
        (60, 120, "Transverse_Direction"),
        (150, 210, "Opposite_Rolling"),
        (240, 300, "Opposite_Transverse")
    ]

    # 步骤 3: 运行积分（这里使用自定义扇区）
    print("\n使用自定义扇区进行积分...")
    results = azimuthal_integration(
        poni_file=poni_file,
        input_pattern=input_pattern,
        output_dir=output_dir,
        sectors=sectors_custom,
        mask_file=mask_file,
        dataset_path="entry/data/data",
        npt=4000,
        unit='2th_deg',
        save_csv=True,
        verbose=True
    )

    # 步骤 4: 处理结果
    print("\n积分完成！生成的文件:")
    for sector_label, file_list in results.items():
        print(f"\n扇区: {sector_label}")
        for filepath in file_list:
            print(f"  - {filepath}")


# ==============================================================================
# 所有可用的预设模板
# ==============================================================================
def show_all_presets():
    """显示所有可用的预设模板"""
    print("=" * 60)
    print("所有可用的预设模板")
    print("=" * 60)

    presets = ['quadrants', 'octants', 'hemispheres', 'horizontal_vertical']

    for preset_name in presets:
        sectors = get_preset_sectors(preset_name)
        print(f"\n{preset_name.upper()}:")
        for start, end, label in sectors:
            print(f"  {label}: {start}° → {end}°")


# ==============================================================================
# 命令行使用说明
# ==============================================================================
def print_cli_usage():
    """打印命令行使用说明"""
    print("=" * 60)
    print("命令行使用方法")
    print("=" * 60)

    examples = [
        {
            "title": "单个扇区",
            "command": """python radial_xrd_azimuthal_integration.py \\
    --poni calibration.poni \\
    --input "data/*.h5" \\
    --output results/ \\
    --sector 0 90 "Q1"
"""
        },
        {
            "title": "四象限预设",
            "command": """python radial_xrd_azimuthal_integration.py \\
    --poni calibration.poni \\
    --input "data/*.h5" \\
    --output results/ \\
    --preset quadrants
"""
        },
        {
            "title": "自定义多扇区（带掩膜）",
            "command": """python radial_xrd_azimuthal_integration.py \\
    --poni calibration.poni \\
    --input "data/*.h5" \\
    --output results/ \\
    --mask mask.edf \\
    --sector 0 90 "Q1" \\
    --sector 90 180 "Q2" \\
    --sector 180 270 "Q3" \\
    --sector 270 360 "Q4"
"""
        },
        {
            "title": "指定单位和点数",
            "command": """python radial_xrd_azimuthal_integration.py \\
    --poni calibration.poni \\
    --input "data/*.h5" \\
    --output results/ \\
    --preset octants \\
    --unit q_A^-1 \\
    --npt 5000
"""
        }
    ]

    for example in examples:
        print(f"\n【{example['title']}】")
        print(example['command'])


# ==============================================================================
# Main函数
# ==============================================================================
if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════╗
║          方位角积分 (Azimuthal Integration) 使用示例         ║
╚══════════════════════════════════════════════════════════════╝

方位角角度参考系:
  0° = 右 (→)  |  90° = 上 (↑)  |  180° = 左 (←)  |  270° = 下 (↓)
  从右水平方向逆时针旋转

请选择要运行的示例（取消注释对应行）:
""")

    # 取消注释以运行相应示例
    # example_single_sector()
    # example_quadrants_preset()
    # example_custom_sectors()
    # example_octants()
    # example_hemispheres()
    # example_complete_workflow()

    # 显示所有预设模板
    show_all_presets()

    # 显示命令行使用方法
    print_cli_usage()

    print("\n" + "=" * 60)
    print("提示: 修改示例代码中的文件路径后运行")
    print("=" * 60)
