# -*- coding: utf-8 -*-
"""
XRayDiffractionAnalyzer 使用示例
Examples of using the XRayDiffractionAnalyzer class
"""

from xray_diffraction_analyzer import XRayDiffractionAnalyzer


# ==================== 示例 1: 基本用法（交互模式）====================
def example_1_interactive_mode():
    """
    基本用法：交互模式
    程序会提示用户选择晶体系统
    """
    print("\n" + "="*80)
    print("示例 1: 交互模式 - 程序会提示您选择晶体系统")
    print("="*80 + "\n")

    # 创建分析器实例
    analyzer = XRayDiffractionAnalyzer(
        wavelength=0.4133,           # X射线波长 (Å)
        peak_tolerance_1=0.3,         # 相变识别容差
        peak_tolerance_2=0.4,         # 新峰确定容差
        peak_tolerance_3=0.01,        # 新峰跟踪容差
        n_pressure_points=4           # 稳定新峰所需压力点数
    )

    # 运行完整分析（交互模式）
    csv_path = 'path/to/your/data.csv'
    results = analyzer.analyze(csv_path)

    # 访问结果
    if results and 'original_results' in results:
        print("\n原始相晶格参数:")
        for pressure, params in results['original_results'].items():
            print(f"  {pressure:.2f} GPa: a = {params['a']:.6f} Å, V = {params['V_cell']:.6f} Å³")

        print("\n新相晶格参数:")
        for pressure, params in results['new_results'].items():
            print(f"  {pressure:.2f} GPa: a = {params['a']:.6f} Å, V = {params['V_cell']:.6f} Å³")


# ==================== 示例 2: 自动模式（预设晶体系统）====================
def example_2_auto_mode():
    """
    自动模式：预先指定晶体系统，无需用户交互
    适合批处理或已知晶体系统的情况
    """
    print("\n" + "="*80)
    print("示例 2: 自动模式 - 预设晶体系统，无需用户交互")
    print("="*80 + "\n")

    # 创建分析器实例
    analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

    # 运行分析，指定晶体系统
    csv_path = 'path/to/your/data.csv'
    results = analyzer.analyze(
        csv_path,
        original_system='cubic_FCC',   # 原始相：面心立方
        new_system='Hexagonal',        # 新相：六方密排
        auto_mode=True                 # 启用自动模式
    )

    return results


# ==================== 示例 3: 分步操作（高级用法）====================
def example_3_step_by_step():
    """
    分步操作：手动控制每个分析步骤
    适合需要自定义流程或中间结果的情况
    """
    print("\n" + "="*80)
    print("示例 3: 分步操作 - 手动控制每个分析步骤")
    print("="*80 + "\n")

    # 创建分析器
    analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

    # 步骤 1: 读取数据
    csv_path = 'path/to/your/data.csv'
    pressure_data = analyzer.read_pressure_peak_data(csv_path)
    print(f"读取到 {len(pressure_data)} 个压力点")

    # 步骤 2: 识别相变点
    transition_p, before_p, after_p = analyzer.find_phase_transition_point()

    if transition_p:
        print(f"相变压力: {transition_p:.2f} GPa")

        # 步骤 3: 获取新峰
        transition_peaks = pressure_data[transition_p]
        prev_pressure = before_p[-1]
        prev_peaks = pressure_data[prev_pressure]

        # 识别新峰
        tolerance_windows = [(p - analyzer.peak_tolerance_1,
                            p + analyzer.peak_tolerance_1) for p in prev_peaks]
        new_peaks_at_transition = [
            peak for peak in transition_peaks
            if not any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
        ]

        # 步骤 4: 跟踪新峰
        stable_count, tracked_new_peaks = analyzer.collect_tracked_new_peaks(
            pressure_data, transition_p, after_p,
            new_peaks_at_transition, analyzer.peak_tolerance_2
        )

        # 步骤 5: 构建原始峰数据集
        original_peak_dataset = analyzer.build_original_peak_dataset(
            pressure_data, tracked_new_peaks, analyzer.peak_tolerance_3
        )

        # 步骤 6: 拟合晶格参数
        print("\n拟合原始相...")
        original_results = analyzer.fit_lattice_parameters(
            original_peak_dataset, 'cubic_FCC'
        )

        print("\n拟合新相...")
        new_results = analyzer.fit_lattice_parameters(
            tracked_new_peaks, 'Hexagonal'
        )

        # 步骤 7: 保存结果
        analyzer.save_lattice_results_to_csv(
            original_results,
            'original_phase_results.csv',
            'cubic_FCC'
        )

        analyzer.save_lattice_results_to_csv(
            new_results,
            'new_phase_results.csv',
            'Hexagonal'
        )

        return original_results, new_results
    else:
        print("未检测到相变")
        return None


# ==================== 示例 4: 单相分析 ====================
def example_4_single_phase():
    """
    单相分析：适用于没有相变的情况
    """
    print("\n" + "="*80)
    print("示例 4: 单相分析")
    print("="*80 + "\n")

    analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

    # 读取数据
    csv_path = 'path/to/your/single_phase_data.csv'
    pressure_data = analyzer.read_pressure_peak_data(csv_path)

    # 直接拟合（假设为FCC结构）
    results = analyzer.fit_lattice_parameters(pressure_data, 'cubic_FCC')

    # 保存结果
    analyzer.save_lattice_results_to_csv(
        results,
        'single_phase_results.csv',
        'cubic_FCC'
    )

    return results


# ==================== 示例 5: 批量处理多个文件 ====================
def example_5_batch_processing():
    """
    批量处理：分析多个CSV文件
    """
    print("\n" + "="*80)
    print("示例 5: 批量处理多个文件")
    print("="*80 + "\n")

    csv_files = [
        'sample1.csv',
        'sample2.csv',
        'sample3.csv'
    ]

    # 为每个样品定义晶体系统
    crystal_systems = {
        'sample1.csv': ('cubic_FCC', 'Hexagonal'),
        'sample2.csv': ('cubic_BCC', 'cubic_FCC'),
        'sample3.csv': ('Hexagonal', 'cubic_FCC')
    }

    all_results = {}

    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file}")

        analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

        original_sys, new_sys = crystal_systems[csv_file]

        results = analyzer.analyze(
            csv_file,
            original_system=original_sys,
            new_system=new_sys,
            auto_mode=True
        )

        all_results[csv_file] = results

    return all_results


# ==================== 示例 6: 自定义参数配置 ====================
def example_6_custom_parameters():
    """
    自定义参数：根据具体实验调整容差参数
    """
    print("\n" + "="*80)
    print("示例 6: 自定义参数配置")
    print("="*80 + "\n")

    # 创建分析器，使用自定义参数
    analyzer = XRayDiffractionAnalyzer(
        wavelength=0.5000,            # 不同的X射线波长
        peak_tolerance_1=0.5,         # 更宽的相变识别容差
        peak_tolerance_2=0.6,         # 更宽的新峰确定容差
        peak_tolerance_3=0.02,        # 更宽的新峰跟踪容差
        n_pressure_points=3           # 更少的压力点要求
    )

    csv_path = 'path/to/your/data.csv'
    results = analyzer.analyze(
        csv_path,
        original_system='cubic_BCC',
        new_system='Tetragonal',
        auto_mode=True
    )

    return results


# ==================== 示例 7: 访问中间结果 ====================
def example_7_access_intermediate_results():
    """
    访问中间结果：获取分析过程中的中间数据
    """
    print("\n" + "="*80)
    print("示例 7: 访问中间结果")
    print("="*80 + "\n")

    analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

    csv_path = 'path/to/your/data.csv'
    results = analyzer.analyze(
        csv_path,
        original_system='cubic_FCC',
        new_system='Hexagonal',
        auto_mode=True
    )

    # 访问存储的中间结果
    print("\n压力-峰位数据:")
    for pressure, peaks in sorted(analyzer.pressure_data.items()):
        print(f"  {pressure:.2f} GPa: {len(peaks)} 个峰")

    print(f"\n相变压力: {analyzer.transition_pressure:.2f} GPa")
    print(f"相变前压力点: {analyzer.before_pressures}")
    print(f"相变后压力点: {analyzer.after_pressures}")

    print("\n原始峰数据集:")
    for pressure, data in sorted(analyzer.original_peak_dataset.items()):
        print(f"  {pressure:.2f} GPa: {data['count']} 个原始峰")

    print("\n跟踪的新峰:")
    for pressure, peaks in sorted(analyzer.tracked_new_peaks.items()):
        print(f"  {pressure:.2f} GPa: {len(peaks)} 个新峰")

    return analyzer


# ==================== 示例 8: 使用静态方法进行单独计算 ====================
def example_8_static_methods():
    """
    使用静态方法：进行独立的晶体学计算
    """
    print("\n" + "="*80)
    print("示例 8: 使用静态方法进行单独计算")
    print("="*80 + "\n")

    # 不需要创建实例，直接使用静态方法

    # 2theta 转 d spacing
    two_theta = 30.0  # degrees
    wavelength = 0.4133  # Å
    d_spacing = XRayDiffractionAnalyzer.two_theta_to_d(two_theta, wavelength)
    print(f"2θ = {two_theta}° → d = {d_spacing:.6f} Å")

    # d spacing 转 2theta
    d = 2.5  # Å
    two_theta_calc = XRayDiffractionAnalyzer.d_to_two_theta(d, wavelength)
    print(f"d = {d} Å → 2θ = {two_theta_calc:.6f}°")

    # 计算立方晶系的d spacing
    hkl = (1, 1, 1)
    a = 4.05  # Å
    d_cubic = XRayDiffractionAnalyzer.calculate_d_cubic(hkl, a)
    print(f"\n立方晶系 (FCC):")
    print(f"  hkl = {hkl}, a = {a} Å → d = {d_cubic:.6f} Å")

    # 计算六方晶系的d spacing
    hkl_hex = (1, 0, 1)
    a_hex = 3.0  # Å
    c_hex = 5.0  # Å
    d_hex = XRayDiffractionAnalyzer.calculate_d_hexagonal(hkl_hex, a_hex, c_hex)
    print(f"\n六方晶系 (HCP):")
    print(f"  hkl = {hkl_hex}, a = {a_hex} Å, c = {c_hex} Å → d = {d_hex:.6f} Å")

    # 计算晶胞体积
    V_cubic = XRayDiffractionAnalyzer.calculate_cell_volume_cubic(a)
    print(f"\n立方晶胞体积: V = {V_cubic:.6f} Å³")

    V_hex = XRayDiffractionAnalyzer.calculate_cell_volume_hexagonal(a_hex, c_hex)
    print(f"六方晶胞体积: V = {V_hex:.6f} Å³")


# ==================== 示例 9: 不同晶体系统的完整示例 ====================
def example_9_different_crystal_systems():
    """
    不同晶体系统：演示各种晶体系统的分析
    """
    print("\n" + "="*80)
    print("示例 9: 不同晶体系统的分析")
    print("="*80 + "\n")

    # 可用的晶体系统
    systems = [
        'cubic_FCC',      # 面心立方
        'cubic_BCC',      # 体心立方
        'cubic_SC',       # 简单立方
        'Hexagonal',      # 六方
        'Tetragonal',     # 四方
        'Orthorhombic',   # 正交
        'Monoclinic',     # 单斜
        'Triclinic'       # 三斜
    ]

    print("可用的晶体系统:")
    for i, system in enumerate(systems, 1):
        system_info = XRayDiffractionAnalyzer.CRYSTAL_SYSTEMS[system]
        print(f"  [{i}] {system}: {system_info['name']}")
        print(f"      最少需要峰数: {system_info['min_peaks']}")
        print(f"      每晶胞原子数: {system_info['atoms_per_cell']}")

    # 示例：分析不同晶体系统
    analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

    csv_path = 'path/to/your/data.csv'

    # FCC → HCP 相变
    print("\n\n分析 FCC → HCP 相变:")
    results_fcc_hcp = analyzer.analyze(
        csv_path,
        original_system='cubic_FCC',
        new_system='Hexagonal',
        auto_mode=True
    )

    # BCC → FCC 相变
    print("\n\n分析 BCC → FCC 相变:")
    analyzer2 = XRayDiffractionAnalyzer(wavelength=0.4133)
    results_bcc_fcc = analyzer2.analyze(
        csv_path,
        original_system='cubic_BCC',
        new_system='cubic_FCC',
        auto_mode=True
    )


# ==================== 快速开始示例 ====================
def quick_start():
    """
    快速开始：最简单的使用方式
    """
    print("\n" + "="*80)
    print("快速开始 - 三行代码完成分析")
    print("="*80 + "\n")

    # 只需三行代码！
    analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
    csv_path = 'path/to/your/data.csv'
    results = analyzer.analyze(csv_path, original_system='cubic_FCC',
                              new_system='Hexagonal', auto_mode=True)

    print("\n分析完成！结果已保存到CSV文件。")
    return results


# ==================== 主函数：展示所有示例 ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" XRayDiffractionAnalyzer 使用示例集合")
    print("="*80)

    print("\n可用示例:")
    print("  1. 交互模式（程序会提示选择晶体系统）")
    print("  2. 自动模式（预设晶体系统）")
    print("  3. 分步操作（高级用法）")
    print("  4. 单相分析")
    print("  5. 批量处理多个文件")
    print("  6. 自定义参数配置")
    print("  7. 访问中间结果")
    print("  8. 使用静态方法进行单独计算")
    print("  9. 不同晶体系统的完整示例")
    print("  0. 快速开始")

    print("\n" + "="*80)
    print("注意：运行前请将 'path/to/your/data.csv' 替换为实际的CSV文件路径")
    print("="*80 + "\n")

    # 取消注释以运行特定示例
    # quick_start()
    # example_2_auto_mode()
    # example_8_static_methods()
