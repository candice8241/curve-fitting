# -*- coding: utf-8 -*-
"""
Birch-Murnaghan方程拟合PV曲线
用于拟合压力-体积数据并计算体模量相关参数
@author: candicewang928@gmail.com
Created: 2025-11-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def birch_murnaghan_2nd(V, V0, B0):
    """
    二阶Birch-Murnaghan状态方程

    参数:
    V: 体积 (Å³/atom)
    V0: 零压体积 (Å³/atom)
    B0: 零压体模量 (GPa)

    返回:
    P: 压力 (GPa)
    """
    eta = (V0 / V) ** (1/3)
    P = 3 * B0 / 2 * (eta**7 - eta**5)
    return P


def birch_murnaghan_3rd(V, V0, B0, B0_prime):
    """
    三阶Birch-Murnaghan状态方程

    参数:
    V: 体积 (Å³/atom)
    V0: 零压体积 (Å³/atom)
    B0: 零压体模量 (GPa)
    B0_prime: 体模量一阶导数 (无量纲)

    返回:
    P: 压力 (GPa)
    """
    eta = (V0 / V) ** (1/3)
    P = 3 * B0 / 2 * (eta**7 - eta**5) * (1 + 0.75 * (B0_prime - 4) * (eta**2 - 1))
    return P


def fit_bm_equations(V_data, P_data, phase_name=""):
    """
    对给定的PV数据进行2阶和3阶BM方程拟合

    参数:
    V_data: 体积数据数组
    P_data: 压力数据数组
    phase_name: 相的名称（用于输出）

    返回:
    results: 包含拟合参数和统计信息的字典
    """
    results = {}

    # 初始猜测值
    V0_guess = np.max(V_data) * 1.02  # 零压体积略大于最大体积
    B0_guess = 150  # 体模量初始猜测 (GPa)
    B0_prime_guess = 4.0  # 体模量一阶导初始猜测

    # 2阶BM方程拟合
    # 设置合理的参数边界以避免过拟合
    # V0: 最大体积的0.8-1.3倍
    # B0: 50-500 GPa（涵盖大多数材料）
    bounds_2nd = ([np.max(V_data) * 0.8, 50],
                  [np.max(V_data) * 1.3, 500])

    try:
        popt_2nd, pcov_2nd = curve_fit(
            birch_murnaghan_2nd,
            V_data,
            P_data,
            p0=[V0_guess, B0_guess],
            bounds=bounds_2nd,
            maxfev=10000
        )

        V0_2nd, B0_2nd = popt_2nd
        perr_2nd = np.sqrt(np.diag(pcov_2nd))

        # 计算拟合残差和R²
        P_fit_2nd = birch_murnaghan_2nd(V_data, *popt_2nd)
        residuals_2nd = P_data - P_fit_2nd
        ss_res_2nd = np.sum(residuals_2nd**2)
        ss_tot_2nd = np.sum((P_data - np.mean(P_data))**2)
        r_squared_2nd = 1 - (ss_res_2nd / ss_tot_2nd)
        rmse_2nd = np.sqrt(np.mean(residuals_2nd**2))

        results['2nd_order'] = {
            'V0': V0_2nd,
            'V0_err': perr_2nd[0],
            'B0': B0_2nd,
            'B0_err': perr_2nd[1],
            'B0_prime': 4.0,  # 2阶方程固定为4
            'B0_prime_err': 0,
            'R_squared': r_squared_2nd,
            'RMSE': rmse_2nd,
            'fitted_P': P_fit_2nd
        }

        print(f"\n{'='*60}")
        print(f"{phase_name} - 二阶Birch-Murnaghan拟合结果:")
        print(f"{'='*60}")
        print(f"V₀ = {V0_2nd:.4f} ± {perr_2nd[0]:.4f} Å³/atom")
        print(f"B₀ = {B0_2nd:.2f} ± {perr_2nd[1]:.2f} GPa")
        print(f"B₀' = 4.0 (固定)")
        print(f"R² = {r_squared_2nd:.6f}")
        print(f"RMSE = {rmse_2nd:.4f} GPa")

    except Exception as e:
        print(f"⚠️ {phase_name} - 二阶BM拟合失败: {e}")
        results['2nd_order'] = None

    # 3阶BM方程拟合
    # 设置合理的参数边界
    # V0: 最大体积的0.8-1.3倍
    # B0: 50-500 GPa
    # B0': 2.5-6.5（基于文献值，大多数材料在3-6之间）
    bounds_3rd = ([np.max(V_data) * 0.8, 50, 2.5],
                  [np.max(V_data) * 1.3, 500, 6.5])

    try:
        popt_3rd, pcov_3rd = curve_fit(
            birch_murnaghan_3rd,
            V_data,
            P_data,
            p0=[V0_guess, B0_guess, B0_prime_guess],
            bounds=bounds_3rd,
            maxfev=10000
        )

        V0_3rd, B0_3rd, B0_prime_3rd = popt_3rd
        perr_3rd = np.sqrt(np.diag(pcov_3rd))

        # 计算拟合残差和R²
        P_fit_3rd = birch_murnaghan_3rd(V_data, *popt_3rd)
        residuals_3rd = P_data - P_fit_3rd
        ss_res_3rd = np.sum(residuals_3rd**2)
        ss_tot_3rd = np.sum((P_data - np.mean(P_data))**2)
        r_squared_3rd = 1 - (ss_res_3rd / ss_tot_3rd)
        rmse_3rd = np.sqrt(np.mean(residuals_3rd**2))

        results['3rd_order'] = {
            'V0': V0_3rd,
            'V0_err': perr_3rd[0],
            'B0': B0_3rd,
            'B0_err': perr_3rd[1],
            'B0_prime': B0_prime_3rd,
            'B0_prime_err': perr_3rd[2],
            'R_squared': r_squared_3rd,
            'RMSE': rmse_3rd,
            'fitted_P': P_fit_3rd
        }

        print(f"\n{'='*60}")
        print(f"{phase_name} - 三阶Birch-Murnaghan拟合结果:")
        print(f"{'='*60}")
        print(f"V₀ = {V0_3rd:.4f} ± {perr_3rd[0]:.4f} Å³/atom")
        print(f"B₀ = {B0_3rd:.2f} ± {perr_3rd[1]:.2f} GPa")
        print(f"B₀' = {B0_prime_3rd:.3f} ± {perr_3rd[2]:.3f}")
        print(f"R² = {r_squared_3rd:.6f}")
        print(f"RMSE = {rmse_3rd:.4f} GPa")

    except Exception as e:
        print(f"⚠️ {phase_name} - 三阶BM拟合失败: {e}")
        results['3rd_order'] = None

    return results


def plot_pv_curves(V_orig, P_orig, V_new, P_new,
                   results_orig, results_new, save_dir):
    """
    绘制PV曲线和拟合结果

    参数:
    V_orig, P_orig: 原相的体积和压力数据
    V_new, P_new: 新相的体积和压力数据
    results_orig, results_new: 拟合结果
    save_dir: 保存图片的目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Birch-Murnaghan方程拟合PV曲线', fontsize=16, fontweight='bold')

    # 原相 - 2阶BM
    ax = axes[0, 0]
    ax.scatter(V_orig, P_orig, s=80, c='blue', marker='o',
               label='实验数据 (原相)', alpha=0.7, edgecolors='black')
    if results_orig['2nd_order'] is not None:
        V_fit = np.linspace(V_orig.min()*0.95, V_orig.max()*1.05, 200)
        P_fit = birch_murnaghan_2nd(V_fit,
                                     results_orig['2nd_order']['V0'],
                                     results_orig['2nd_order']['B0'])
        ax.plot(V_fit, P_fit, 'r-', linewidth=2.5, label='2阶BM拟合', alpha=0.8)

        # 添加拟合参数文本
        textstr = f"V₀ = {results_orig['2nd_order']['V0']:.4f} Ų/atom\n"
        textstr += f"B₀ = {results_orig['2nd_order']['B0']:.2f} GPa\n"
        textstr += f"B₀' = 4.0 (固定)\n"
        textstr += f"R² = {results_orig['2nd_order']['R_squared']:.6f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    ax.set_xlabel('体积 V (Ų/atom)', fontsize=12)
    ax.set_ylabel('压力 P (GPa)', fontsize=12)
    ax.set_title('原相 - 二阶BM方程', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 原相 - 3阶BM
    ax = axes[0, 1]
    ax.scatter(V_orig, P_orig, s=80, c='blue', marker='o',
               label='实验数据 (原相)', alpha=0.7, edgecolors='black')
    if results_orig['3rd_order'] is not None:
        V_fit = np.linspace(V_orig.min()*0.95, V_orig.max()*1.05, 200)
        P_fit = birch_murnaghan_3rd(V_fit,
                                     results_orig['3rd_order']['V0'],
                                     results_orig['3rd_order']['B0'],
                                     results_orig['3rd_order']['B0_prime'])
        ax.plot(V_fit, P_fit, 'g-', linewidth=2.5, label='3阶BM拟合', alpha=0.8)

        textstr = f"V₀ = {results_orig['3rd_order']['V0']:.4f} Ų/atom\n"
        textstr += f"B₀ = {results_orig['3rd_order']['B0']:.2f} GPa\n"
        textstr += f"B₀' = {results_orig['3rd_order']['B0_prime']:.3f}\n"
        textstr += f"R² = {results_orig['3rd_order']['R_squared']:.6f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

    ax.set_xlabel('体积 V (Ų/atom)', fontsize=12)
    ax.set_ylabel('压力 P (GPa)', fontsize=12)
    ax.set_title('原相 - 三阶BM方程', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 新相 - 2阶BM
    ax = axes[1, 0]
    ax.scatter(V_new, P_new, s=80, c='red', marker='s',
               label='实验数据 (新相)', alpha=0.7, edgecolors='black')
    if results_new['2nd_order'] is not None:
        V_fit = np.linspace(V_new.min()*0.95, V_new.max()*1.05, 200)
        P_fit = birch_murnaghan_2nd(V_fit,
                                     results_new['2nd_order']['V0'],
                                     results_new['2nd_order']['B0'])
        ax.plot(V_fit, P_fit, 'r-', linewidth=2.5, label='2阶BM拟合', alpha=0.8)

        textstr = f"V₀ = {results_new['2nd_order']['V0']:.4f} Ų/atom\n"
        textstr += f"B₀ = {results_new['2nd_order']['B0']:.2f} GPa\n"
        textstr += f"B₀' = 4.0 (固定)\n"
        textstr += f"R² = {results_new['2nd_order']['R_squared']:.6f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    ax.set_xlabel('体积 V (Ų/atom)', fontsize=12)
    ax.set_ylabel('压力 P (GPa)', fontsize=12)
    ax.set_title('新相 - 二阶BM方程', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 新相 - 3阶BM
    ax = axes[1, 1]
    ax.scatter(V_new, P_new, s=80, c='red', marker='s',
               label='实验数据 (新相)', alpha=0.7, edgecolors='black')
    if results_new['3rd_order'] is not None:
        V_fit = np.linspace(V_new.min()*0.95, V_new.max()*1.05, 200)
        P_fit = birch_murnaghan_3rd(V_fit,
                                     results_new['3rd_order']['V0'],
                                     results_new['3rd_order']['B0'],
                                     results_new['3rd_order']['B0_prime'])
        ax.plot(V_fit, P_fit, 'g-', linewidth=2.5, label='3阶BM拟合', alpha=0.8)

        textstr = f"V₀ = {results_new['3rd_order']['V0']:.4f} Ų/atom\n"
        textstr += f"B₀ = {results_new['3rd_order']['B0']:.2f} GPa\n"
        textstr += f"B₀' = {results_new['3rd_order']['B0_prime']:.3f}\n"
        textstr += f"R² = {results_new['3rd_order']['R_squared']:.6f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

    ax.set_xlabel('体积 V (Ų/atom)', fontsize=12)
    ax.set_ylabel('压力 P (GPa)', fontsize=12)
    ax.set_title('新相 - 三阶BM方程', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(save_dir, 'BM_fitting_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ PV曲线图已保存至: {output_path}")

    plt.show()


def plot_residuals(V_orig, P_orig, V_new, P_new,
                   results_orig, results_new, save_dir):
    """
    绘制拟合残差图

    参数:
    V_orig, P_orig: 原相的体积和压力数据
    V_new, P_new: 新相的体积和压力数据
    results_orig, results_new: 拟合结果
    save_dir: 保存图片的目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('拟合残差分析', fontsize=16, fontweight='bold')

    # 原相 - 2阶BM残差
    ax = axes[0, 0]
    if results_orig['2nd_order'] is not None:
        residuals = P_orig - results_orig['2nd_order']['fitted_P']
        ax.scatter(V_orig, residuals, s=60, c='blue', marker='o', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('体积 V (Ų/atom)', fontsize=11)
        ax.set_ylabel('残差 (GPa)', fontsize=11)
        ax.set_title('原相 - 二阶BM残差', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        textstr = f"RMSE = {results_orig['2nd_order']['RMSE']:.4f} GPa"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    # 原相 - 3阶BM残差
    ax = axes[0, 1]
    if results_orig['3rd_order'] is not None:
        residuals = P_orig - results_orig['3rd_order']['fitted_P']
        ax.scatter(V_orig, residuals, s=60, c='blue', marker='o', alpha=0.7)
        ax.axhline(y=0, color='g', linestyle='--', linewidth=2)
        ax.set_xlabel('体积 V (Ų/atom)', fontsize=11)
        ax.set_ylabel('残差 (GPa)', fontsize=11)
        ax.set_title('原相 - 三阶BM残差', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        textstr = f"RMSE = {results_orig['3rd_order']['RMSE']:.4f} GPa"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

    # 新相 - 2阶BM残差
    ax = axes[1, 0]
    if results_new['2nd_order'] is not None:
        residuals = P_new - results_new['2nd_order']['fitted_P']
        ax.scatter(V_new, residuals, s=60, c='red', marker='s', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('体积 V (Ų/atom)', fontsize=11)
        ax.set_ylabel('残差 (GPa)', fontsize=11)
        ax.set_title('新相 - 二阶BM残差', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        textstr = f"RMSE = {results_new['2nd_order']['RMSE']:.4f} GPa"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    # 新相 - 3阶BM残差
    ax = axes[1, 1]
    if results_new['3rd_order'] is not None:
        residuals = P_new - results_new['3rd_order']['fitted_P']
        ax.scatter(V_new, residuals, s=60, c='red', marker='s', alpha=0.7)
        ax.axhline(y=0, color='g', linestyle='--', linewidth=2)
        ax.set_xlabel('体积 V (Ų/atom)', fontsize=11)
        ax.set_ylabel('残差 (GPa)', fontsize=11)
        ax.set_title('新相 - 三阶BM残差', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        textstr = f"RMSE = {results_new['3rd_order']['RMSE']:.4f} GPa"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    output_path = os.path.join(save_dir, 'BM_fitting_residuals.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 残差图已保存至: {output_path}")

    plt.show()


def save_results_to_csv(results_orig, results_new, save_dir):
    """
    将拟合结果保存为CSV文件

    参数:
    results_orig: 原相拟合结果
    results_new: 新相拟合结果
    save_dir: 保存目录
    """
    # 创建结果汇总表
    summary_data = []

    for phase_name, results in [('原相', results_orig), ('新相', results_new)]:
        for order in ['2nd_order', '3rd_order']:
            if results[order] is not None:
                row = {
                    '相': phase_name,
                    '拟合阶数': '2阶' if order == '2nd_order' else '3阶',
                    'V₀ (Ų/atom)': f"{results[order]['V0']:.6f}",
                    'V₀误差': f"{results[order]['V0_err']:.6f}",
                    'B₀ (GPa)': f"{results[order]['B0']:.4f}",
                    'B₀误差': f"{results[order]['B0_err']:.4f}",
                    "B₀'": f"{results[order]['B0_prime']:.6f}",
                    "B₀'误差": f"{results[order]['B0_prime_err']:.6f}",
                    'R²': f"{results[order]['R_squared']:.8f}",
                    'RMSE (GPa)': f"{results[order]['RMSE']:.6f}"
                }
                summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)

    output_path = os.path.join(save_dir, 'BM_fitting_parameters.csv')
    df_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 拟合参数已保存至: {output_path}")

    return df_summary


def main():
    """
    主函数：读取数据、拟合、绘图、保存结果
    """
    print("\n" + "="*80)
    print("Birch-Murnaghan方程拟合PV曲线程序")
    print("="*80)

    # 设置数据路径（请根据实际情况修改）
    data_dir = r"D:\HEPS\ID31\dioptas_data\Al0"  # 修改为你的数据目录
    orig_file = os.path.join(data_dir, "all_results_original_peaks_lattice.csv")
    new_file = os.path.join(data_dir, "all_results_new_peaks_lattice.csv")

    # 创建输出目录
    save_dir = os.path.join(data_dir, "BM_fitting_output")
    os.makedirs(save_dir, exist_ok=True)

    # 读取数据
    print(f"\n📂 正在读取数据文件...")
    print(f"   原相数据: {orig_file}")
    print(f"   新相数据: {new_file}")

    try:
        df_orig = pd.read_csv(orig_file)
        df_new = pd.read_csv(new_file)
        print("✅ 数据读取成功!")
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到数据文件")
        print(f"   请确保以下文件存在:")
        print(f"   - {orig_file}")
        print(f"   - {new_file}")
        print(f"\n💡 提示: 请修改 main() 函数中的 data_dir 变量为你的实际数据目录")
        return

    # 检查必要的列是否存在
    required_columns = ['V_atomic', 'Pressure (GPa)']
    for col in required_columns:
        if col not in df_orig.columns or col not in df_new.columns:
            print(f"❌ 错误: 数据文件中缺少必要的列 '{col}'")
            print(f"   原相列名: {df_orig.columns.tolist()}")
            print(f"   新相列名: {df_new.columns.tolist()}")
            return

    # 提取数据并移除空值
    V_orig = df_orig['V_atomic'].dropna().values
    P_orig = df_orig['Pressure (GPa)'].dropna().values
    V_new = df_new['V_atomic'].dropna().values
    P_new = df_new['Pressure (GPa)'].dropna().values

    # 确保数据配对
    min_len_orig = min(len(V_orig), len(P_orig))
    V_orig = V_orig[:min_len_orig]
    P_orig = P_orig[:min_len_orig]

    min_len_new = min(len(V_new), len(P_new))
    V_new = V_new[:min_len_new]
    P_new = P_new[:min_len_new]

    print(f"\n📊 数据概览:")
    print(f"   原相数据点数: {len(V_orig)}")
    print(f"   新相数据点数: {len(V_new)}")
    print(f"   原相体积范围: {V_orig.min():.4f} - {V_orig.max():.4f} Ų/atom")
    print(f"   原相压力范围: {P_orig.min():.2f} - {P_orig.max():.2f} GPa")
    print(f"   新相体积范围: {V_new.min():.4f} - {V_new.max():.4f} Ų/atom")
    print(f"   新相压力范围: {P_new.min():.2f} - {P_new.max():.2f} GPa")

    # 进行拟合
    print(f"\n🔧 开始进行Birch-Murnaghan方程拟合...")
    results_orig = fit_bm_equations(V_orig, P_orig, "原相")
    results_new = fit_bm_equations(V_new, P_new, "新相")

    # 绘制PV曲线
    print(f"\n📈 正在绘制PV曲线...")
    plot_pv_curves(V_orig, P_orig, V_new, P_new,
                   results_orig, results_new, save_dir)

    # 绘制残差图
    print(f"\n📉 正在绘制残差图...")
    plot_residuals(V_orig, P_orig, V_new, P_new,
                   results_orig, results_new, save_dir)

    # 保存结果
    print(f"\n💾 正在保存拟合参数...")
    df_summary = save_results_to_csv(results_orig, results_new, save_dir)

    print(f"\n{'='*80}")
    print("✨ 所有任务完成!")
    print(f"{'='*80}")
    print(f"📁 输出目录: {save_dir}")
    print(f"   - BM_fitting_results.png : PV曲线拟合图")
    print(f"   - BM_fitting_residuals.png : 残差分析图")
    print(f"   - BM_fitting_parameters.csv : 拟合参数汇总表")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
