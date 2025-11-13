# -*- coding: utf-8 -*-
"""
BirchMurnaghanFitter 使用示例
Examples of using the BirchMurnaghanFitter class
"""

from birch_murnaghan_fitter import BirchMurnaghanFitter
import numpy as np


# ==================== 示例 1: 最简单的用法（完整分析）====================
def example_1_basic_usage():
    """
    最简单的用法：一行代码完成完整分析
    """
    print("\n" + "="*80)
    print("示例 1: 最简单的用法 - 自动完成所有步骤")
    print("="*80 + "\n")

    # 创建拟合器
    fitter = BirchMurnaghanFitter()

    # 设置文件路径
    original_csv = 'data/original_phase.csv'
    new_csv = 'data/new_phase.csv'
    output_dir = 'output/BM_fitting'

    # 一行代码完成所有分析
    results = fitter.analyze(original_csv, new_csv, output_dir)

    # 访问结果
    if results:
        print("\n原始相 - 2阶BM方程:")
        print(f"  V₀ = {results['original_phase']['2nd_order']['V0']:.4f} Å³/atom")
        print(f"  B₀ = {results['original_phase']['2nd_order']['B0']:.2f} GPa")


# ==================== 示例 2: 自定义参数 ====================
def example_2_custom_parameters():
    """
    自定义拟合参数
    """
    print("\n" + "="*80)
    print("示例 2: 自定义拟合参数")
    print("="*80 + "\n")

    # 创建拟合器，自定义参数边界
    fitter = BirchMurnaghanFitter(
        V0_bounds=(0.7, 1.4),        # V0范围：0.7-1.4倍最大体积
        B0_bounds=(30, 600),          # B0范围：30-600 GPa
        B0_prime_bounds=(2.0, 7.0),   # B0'范围：2.0-7.0
        max_iterations=20000          # 最大迭代次数
    )

    # 执行分析
    results = fitter.analyze(
        'data/original_phase.csv',
        'data/new_phase.csv',
        'output/custom_params'
    )


# ==================== 示例 3: 手动输入数据 ====================
def example_3_manual_data():
    """
    手动输入压力-体积数据
    """
    print("\n" + "="*80)
    print("示例 3: 手动输入数据")
    print("="*80 + "\n")

    # 创建拟合器
    fitter = BirchMurnaghanFitter()

    # 手动设置数据（示例数据）
    V_original = np.array([16.8, 16.5, 16.2, 15.9, 15.6, 15.3])  # Å³/atom
    P_original = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])    # GPa

    V_new = np.array([15.2, 14.9, 14.6, 14.3, 14.0])
    P_new = np.array([15.0, 20.0, 25.0, 30.0, 35.0])

    fitter.set_data_manually(V_original, P_original, V_new, P_new)

    # 执行拟合
    results_orig, results_new = fitter.fit_all_phases()

    # 绘图（不保存）
    fitter.plot_pv_curves()
    fitter.plot_residuals()


# ==================== 示例 4: 分步操作 ====================
def example_4_step_by_step():
    """
    分步操作：手动控制每个步骤
    """
    print("\n" + "="*80)
    print("示例 4: 分步操作")
    print("="*80 + "\n")

    # 第1步：创建拟合器
    fitter = BirchMurnaghanFitter()

    # 第2步：加载数据
    success = fitter.load_data_from_csv(
        'data/original_phase.csv',
        'data/new_phase.csv'
    )

    if not success:
        print("数据加载失败")
        return

    # 第3步：执行拟合
    print("\n拟合原始相...")
    results_orig = fitter.fit_single_phase(
        fitter.V_original,
        fitter.P_original,
        "Original Phase"
    )

    print("\n拟合新相...")
    results_new = fitter.fit_single_phase(
        fitter.V_new,
        fitter.P_new,
        "New Phase"
    )

    # 存储结果
    fitter.results_original = results_orig
    fitter.results_new = results_new

    # 第4步：绘图
    print("\n绘制P-V曲线...")
    fitter.plot_pv_curves(save_path='output/step_by_step/pv_curves.png')

    print("\n绘制残差图...")
    fitter.plot_residuals(save_path='output/step_by_step/residuals.png')

    # 第5步：保存结果
    print("\n保存结果...")
    fitter.save_results_to_csv('output/step_by_step/results.csv')


# ==================== 示例 5: 只拟合单相 ====================
def example_5_single_phase():
    """
    只拟合单个相
    """
    print("\n" + "="*80)
    print("示例 5: 只拟合单个相")
    print("="*80 + "\n")

    # 创建拟合器
    fitter = BirchMurnaghanFitter()

    # 准备单相数据
    V_data = np.array([16.8, 16.5, 16.2, 15.9, 15.6, 15.3, 15.0])
    P_data = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

    # 拟合单相
    results = fitter.fit_single_phase(V_data, P_data, "Test Phase")

    # 打印结果
    if results['2nd_order']:
        print(f"\n2阶BM方程结果:")
        print(f"  V₀ = {results['2nd_order']['V0']:.4f} Å³/atom")
        print(f"  B₀ = {results['2nd_order']['B0']:.2f} GPa")
        print(f"  R² = {results['2nd_order']['R_squared']:.6f}")

    if results['3rd_order']:
        print(f"\n3阶BM方程结果:")
        print(f"  V₀ = {results['3rd_order']['V0']:.4f} Å³/atom")
        print(f"  B₀ = {results['3rd_order']['B0']:.2f} GPa")
        print(f"  B₀' = {results['3rd_order']['B0_prime']:.3f}")
        print(f"  R² = {results['3rd_order']['R_squared']:.6f}")


# ==================== 示例 6: 使用静态方法计算压力 ====================
def example_6_static_methods():
    """
    使用静态方法：直接计算给定参数下的压力
    """
    print("\n" + "="*80)
    print("示例 6: 使用静态方法计算压力")
    print("="*80 + "\n")

    # 不需要创建实例，直接使用静态方法

    # 设置已知参数
    V0 = 16.8  # Å³/atom
    B0 = 150   # GPa
    B0_prime = 4.0

    # 计算不同体积下的压力
    volumes = np.array([16.0, 15.5, 15.0, 14.5, 14.0])

    print("使用2阶BM方程计算压力:")
    for V in volumes:
        P = BirchMurnaghanFitter.birch_murnaghan_2nd(V, V0, B0)
        print(f"  V = {V:.2f} Å³/atom → P = {P:.2f} GPa")

    print("\n使用3阶BM方程计算压力:")
    for V in volumes:
        P = BirchMurnaghanFitter.birch_murnaghan_3rd(V, V0, B0, B0_prime)
        print(f"  V = {V:.2f} Å³/atom → P = {P:.2f} GPa")


# ==================== 示例 7: 批量处理多个样品 ====================
def example_7_batch_processing():
    """
    批量处理多个样品
    """
    print("\n" + "="*80)
    print("示例 7: 批量处理多个样品")
    print("="*80 + "\n")

    # 样品列表
    samples = [
        {
            'name': 'Sample_A',
            'original': 'data/sampleA_original.csv',
            'new': 'data/sampleA_new.csv',
            'output': 'output/sampleA'
        },
        {
            'name': 'Sample_B',
            'original': 'data/sampleB_original.csv',
            'new': 'data/sampleB_new.csv',
            'output': 'output/sampleB'
        },
        {
            'name': 'Sample_C',
            'original': 'data/sampleC_original.csv',
            'new': 'data/sampleC_new.csv',
            'output': 'output/sampleC'
        }
    ]

    # 批量处理
    all_results = {}

    for sample in samples:
        print(f"\n处理样品: {sample['name']}")
        print("-" * 60)

        fitter = BirchMurnaghanFitter()
        results = fitter.analyze(
            sample['original'],
            sample['new'],
            sample['output']
        )

        all_results[sample['name']] = results

    # 汇总结果
    print("\n" + "="*80)
    print("批量处理汇总")
    print("="*80)

    for sample_name, results in all_results.items():
        if results and results['original_phase']['2nd_order']:
            print(f"\n{sample_name}:")
            print(f"  原始相 B₀ = {results['original_phase']['2nd_order']['B0']:.2f} GPa")
            print(f"  新相 B₀ = {results['new_phase']['2nd_order']['B0']:.2f} GPa")


# ==================== 示例 8: 访问和分析结果 ====================
def example_8_result_analysis():
    """
    详细访问和分析拟合结果
    """
    print("\n" + "="*80)
    print("示例 8: 访问和分析拟合结果")
    print("="*80 + "\n")

    # 创建拟合器并执行分析
    fitter = BirchMurnaghanFitter()
    results = fitter.analyze(
        'data/original_phase.csv',
        'data/new_phase.csv',
        'output/result_analysis'
    )

    if not results:
        return

    # 详细访问原始相结果
    print("\n原始相详细结果:")
    print("-" * 60)

    if results['original_phase']['2nd_order']:
        res_2nd = results['original_phase']['2nd_order']
        print("\n2阶BM方程:")
        print(f"  V₀ = {res_2nd['V0']:.6f} ± {res_2nd['V0_err']:.6f} Å³/atom")
        print(f"  B₀ = {res_2nd['B0']:.4f} ± {res_2nd['B0_err']:.4f} GPa")
        print(f"  B₀' = {res_2nd['B0_prime']:.4f} (fixed)")
        print(f"  R² = {res_2nd['R_squared']:.8f}")
        print(f"  RMSE = {res_2nd['RMSE']:.6f} GPa")

    if results['original_phase']['3rd_order']:
        res_3rd = results['original_phase']['3rd_order']
        print("\n3阶BM方程:")
        print(f"  V₀ = {res_3rd['V0']:.6f} ± {res_3rd['V0_err']:.6f} Å³/atom")
        print(f"  B₀ = {res_3rd['B0']:.4f} ± {res_3rd['B0_err']:.4f} GPa")
        print(f"  B₀' = {res_3rd['B0_prime']:.6f} ± {res_3rd['B0_prime_err']:.6f}")
        print(f"  R² = {res_3rd['R_squared']:.8f}")
        print(f"  RMSE = {res_3rd['RMSE']:.6f} GPa")

    # 比较2阶和3阶拟合
    print("\n" + "="*80)
    print("2阶 vs 3阶拟合比较:")
    print("="*80)

    for phase_name in ['original_phase', 'new_phase']:
        print(f"\n{phase_name.replace('_', ' ').title()}:")

        res_2nd = results[phase_name]['2nd_order']
        res_3rd = results[phase_name]['3rd_order']

        if res_2nd and res_3rd:
            print(f"  B₀差异: {abs(res_2nd['B0'] - res_3rd['B0']):.2f} GPa")
            print(f"  R²提升: {(res_3rd['R_squared'] - res_2nd['R_squared'])*100:.4f}%")
            print(f"  RMSE改善: {(res_2nd['RMSE'] - res_3rd['RMSE']):.6f} GPa")

            # 判断哪个拟合更好
            if res_3rd['R_squared'] > res_2nd['R_squared']:
                print("  → 3阶BM方程拟合效果更好")
            else:
                print("  → 2阶BM方程已足够")


# ==================== 示例 9: 不保存文件，只显示结果 ====================
def example_9_no_save():
    """
    不保存文件，只在屏幕上显示结果和图表
    """
    print("\n" + "="*80)
    print("示例 9: 不保存文件，只显示结果")
    print("="*80 + "\n")

    # 创建拟合器
    fitter = BirchMurnaghanFitter()

    # 执行分析，不指定输出目录
    results = fitter.analyze(
        'data/original_phase.csv',
        'data/new_phase.csv',
        output_dir=None  # 不保存文件
    )

    print("\n分析完成，结果已在图表中显示")


# ==================== 快速开始 ====================
def quick_start():
    """
    快速开始：最简单的使用方式
    """
    print("\n" + "="*80)
    print("快速开始 - 三行代码完成分析")
    print("="*80 + "\n")

    # 只需三行代码！
    fitter = BirchMurnaghanFitter()
    results = fitter.analyze('data/original_phase.csv',
                            'data/new_phase.csv',
                            'output/quick_start')

    print("\n分析完成！")


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" BirchMurnaghanFitter 使用示例集合")
    print("="*80)

    print("\n可用示例:")
    print("  1. 最简单的用法（完整分析）")
    print("  2. 自定义参数")
    print("  3. 手动输入数据")
    print("  4. 分步操作")
    print("  5. 只拟合单相")
    print("  6. 使用静态方法计算压力")
    print("  7. 批量处理多个样品")
    print("  8. 访问和分析结果")
    print("  9. 不保存文件，只显示结果")
    print("  0. 快速开始")

    print("\n" + "="*80)
    print("注意：运行前请将数据路径替换为实际的CSV文件路径")
    print("="*80 + "\n")

    # 取消注释以运行特定示例
    # quick_start()
    # example_1_basic_usage()
    # example_3_manual_data()
    # example_6_static_methods()
