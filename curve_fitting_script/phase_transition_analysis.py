# -*- coding: utf-8 -*-
"""
晶系判断和相变识别程序
Phase Transition Analysis and Crystal System Determination

Created on Nov 12, 2025
@author: candicewang928@gmail.com

功能：
1. 从CSV文件读取压力-峰位数据
2. 识别相变点（新峰出现）
3. 统计新峰数目
4. 根据新峰数目判断晶系
5. 计算晶胞参数
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
WAVELENGTH = 0.6199  # X射线波长 (Å)，可根据实际情况修改
PEAK_TOLERANCE_1 = 0.3  # 识别相变点的峰位容差 (度)
PEAK_TOLERANCE_2 = 0.2  # 确定新峰数目的容差 (度)
PEAK_TOLERANCE_3 = 0.15  # 后续压力点围绕新峰的容差 (度)
N_PRESSURE_POINTS = 4  # 用于确定新峰数目稳定的压力点数量

# ==================== 各晶系HKL顺序定义 ====================
# 定义各晶系按2theta从小到大的hkl顺序（前20个）

CRYSTAL_SYSTEMS = {
    'cubic_fcc': {
        'name': 'Face-Centered Cubic (FCC)',
        'min_peaks': 1,
        'hkl_list': [
            (1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2),
            (4,0,0), (3,3,1), (4,2,0), (4,2,2), (3,3,3),
            (5,1,1), (4,4,0), (5,3,1), (6,0,0), (6,2,0),
            (5,3,3), (6,2,2), (4,4,4), (5,5,1), (6,4,0)
        ]
    },
    'cubic_bcc': {
        'name': 'Body-Centered Cubic (BCC)',
        'min_peaks': 1,
        'hkl_list': [
            (1,1,0), (2,0,0), (2,1,1), (2,2,0), (3,1,0),
            (2,2,2), (3,2,1), (4,0,0), (3,3,0), (4,1,1),
            (3,3,2), (4,2,0), (4,2,2), (3,3,3), (5,1,0),
            (4,3,1), (5,2,1), (4,4,0), (5,3,0), (6,0,0)
        ]
    },
    'cubic_sc': {
        'name': 'Simple Cubic (SC)',
        'min_peaks': 1,
        'hkl_list': [
            (1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0),
            (2,1,1), (2,2,0), (2,2,1), (3,0,0), (3,1,0),
            (3,1,1), (2,2,2), (3,2,0), (3,2,1), (4,0,0),
            (4,1,0), (3,3,0), (4,1,1), (3,3,1), (4,2,0)
        ]
    },
    'hexagonal': {
        'name': 'Hexagonal (HCP)',
        'min_peaks': 2,
        'hkl_list': [
            (1,0,0), (0,0,2), (1,0,1), (1,0,2), (1,1,0),
            (1,0,3), (2,0,0), (1,1,2), (2,0,1), (0,0,4),
            (2,0,2), (1,0,4), (2,0,3), (2,1,0), (2,1,1),
            (2,0,4), (2,1,2), (3,0,0), (2,1,3), (2,2,0)
        ]
    },
    'tetragonal': {
        'name': 'Tetragonal',
        'min_peaks': 2,
        'hkl_list': [
            (1,0,0), (0,0,1), (1,1,0), (1,0,1), (1,1,1),
            (2,0,0), (2,1,0), (0,0,2), (2,1,1), (2,0,1),
            (2,2,0), (2,1,2), (3,0,0), (2,2,1), (3,1,0),
            (2,0,2), (3,1,1), (2,2,2), (3,2,0), (3,0,1)
        ]
    },
    'orthorhombic': {
        'name': 'Orthorhombic',
        'min_peaks': 3,
        'hkl_list': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (1,1,1), (2,0,0), (2,1,0), (2,0,1),
            (1,2,0), (0,2,0), (1,2,1), (0,2,1), (2,1,1),
            (2,2,0), (2,0,2), (0,0,2), (2,2,1), (3,0,0)
        ]
    },
    'monoclinic': {
        'name': 'Monoclinic',
        'min_peaks': 4,
        'hkl_list': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (1,-1,0), (1,0,-1), (1,1,1), (2,0,0),
            (1,-1,1), (2,1,0), (0,2,0), (2,0,1), (1,2,0),
            (0,0,2), (2,1,1), (1,1,-1), (2,-1,0), (2,0,-1)
        ]
    },
    'triclinic': {
        'name': 'Triclinic',
        'min_peaks': 6,
        'hkl_list': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (1,-1,0), (1,0,-1), (0,1,-1), (1,1,1),
            (1,-1,1), (1,1,-1), (2,0,0), (0,2,0), (0,0,2),
            (2,1,0), (2,0,1), (1,2,0), (0,2,1), (1,0,2)
        ]
    }
}

# ==================== 辅助函数 ====================

def two_theta_to_d(two_theta, wavelength=WAVELENGTH):
    """
    将2theta角度转换为d间距

    参数:
        two_theta: 2theta角度 (度)
        wavelength: X射线波长 (Å)

    返回:
        d间距 (Å)
    """
    theta_rad = np.deg2rad(two_theta / 2.0)
    return wavelength / (2.0 * np.sin(theta_rad))

def d_to_two_theta(d, wavelength=WAVELENGTH):
    """
    将d间距转换为2theta角度

    参数:
        d: d间距 (Å)
        wavelength: X射线波长 (Å)

    返回:
        2theta角度 (度)
    """
    sin_theta = wavelength / (2.0 * d)
    if sin_theta > 1.0 or sin_theta < -1.0:
        return None
    theta_rad = np.arcsin(sin_theta)
    return np.rad2deg(2.0 * theta_rad)

def calculate_d_cubic(hkl, a):
    """计算立方晶系的d间距"""
    h, k, l = hkl
    return a / np.sqrt(h**2 + k**2 + l**2)

def calculate_d_hexagonal(hkl, a, c):
    """计算六方晶系的d间距"""
    h, k, l = hkl
    return 1.0 / np.sqrt(4.0/3.0 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)

def calculate_d_tetragonal(hkl, a, c):
    """计算四方晶系的d间距"""
    h, k, l = hkl
    return 1.0 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)

def calculate_d_orthorhombic(hkl, a, b, c):
    """计算正交晶系的d间距"""
    h, k, l = hkl
    return 1.0 / np.sqrt(h**2 / a**2 + k**2 / b**2 + l**2 / c**2)

def calculate_d_monoclinic(hkl, a, b, c, beta):
    """计算单斜晶系的d间距"""
    h, k, l = hkl
    beta_rad = np.deg2rad(beta)
    sin_beta = np.sin(beta_rad)
    cos_beta = np.cos(beta_rad)

    term = (h**2 / a**2 + k**2 * sin_beta**2 / b**2 + l**2 / c**2
            - 2*h*l*cos_beta / (a*c)) / sin_beta**2
    return 1.0 / np.sqrt(term)

# ==================== CSV读取和数据预处理 ====================

def read_pressure_peak_data(csv_path):
    """
    读取CSV文件，提取压力点和峰位数据

    参数:
        csv_path: CSV文件路径

    返回:
        pressure_data: 字典，键为压力值(GPa)，值为峰位列表(2theta)
    """
    df = pd.read_csv(csv_path)

    # 检查必要的列
    if 'File' not in df.columns or 'Center' not in df.columns:
        raise ValueError("CSV文件必须包含'File'和'Center'列")

    pressure_data = {}
    current_pressure = None

    for idx, row in df.iterrows():
        # 检查是否是空行（分隔符）
        if pd.isna(row['File']) or row['File'] == '':
            current_pressure = None
            continue

        # 提取压力值
        try:
            # 假设File列包含压力信息，格式可能为"filename_XXGPa"或直接为数字
            file_str = str(row['File'])
            # 尝试提取数字
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', file_str)
            if numbers:
                pressure = float(numbers[0])
            else:
                pressure = float(file_str)
        except:
            print(f"警告：无法解析压力值：{row['File']}")
            continue

        # 提取峰位
        try:
            peak_position = float(row['Center'])
        except:
            print(f"警告：无法解析峰位：{row['Center']}")
            continue

        # 添加到字典
        if pressure not in pressure_data:
            pressure_data[pressure] = []
        pressure_data[pressure].append(peak_position)

    # 对每个压力点的峰位排序
    for pressure in pressure_data:
        pressure_data[pressure] = sorted(pressure_data[pressure])

    return pressure_data

# ==================== 相变点识别 ====================

def find_phase_transition_point(pressure_data, tolerance=PEAK_TOLERANCE_1):
    """
    识别相变点：找到第一个出现新峰的压力点

    参数:
        pressure_data: 压力-峰位数据字典
        tolerance: 峰位容差 (度)

    返回:
        transition_pressure: 相变压力点 (GPa)
        before_pressures: 相变前的压力点列表
        after_pressures: 相变后的压力点列表
    """
    # 按压力从小到大排序
    sorted_pressures = sorted(pressure_data.keys())

    if len(sorted_pressures) < 2:
        print("警告：压力点数量少于2个，无法识别相变")
        return None, sorted_pressures, []

    # 逐个比较相邻压力点
    for i in range(1, len(sorted_pressures)):
        prev_pressure = sorted_pressures[i-1]
        curr_pressure = sorted_pressures[i]

        prev_peaks = pressure_data[prev_pressure]
        curr_peaks = pressure_data[curr_pressure]

        # 检查是否有新峰出现
        has_new_peak = False
        for peak in curr_peaks:
            # 检查这个峰是否在前一个压力点的峰位附近
            min_distance = min([abs(peak - prev_peak) for prev_peak in prev_peaks])
            if min_distance > tolerance:
                has_new_peak = True
                break

        if has_new_peak:
            print(f"\n>>> 发现相变点：{curr_pressure:.2f} GPa")
            return curr_pressure, sorted_pressures[:i], sorted_pressures[i:]

    print("\n>>> 未发现明显相变点")
    return None, sorted_pressures, []

# ==================== 新峰统计 ====================

def count_new_peaks(reference_peaks, current_peaks, tolerance):
    """
    统计新峰数量

    参数:
        reference_peaks: 参考峰位列表
        current_peaks: 当前峰位列表
        tolerance: 峰位容差 (度)

    返回:
        new_peaks: 新峰列表
        original_peaks: 原有峰列表
    """
    new_peaks = []
    original_peaks = []

    for peak in current_peaks:
        # 检查是否为新峰
        min_distance = min([abs(peak - ref_peak) for ref_peak in reference_peaks]) if reference_peaks else float('inf')

        if min_distance > tolerance:
            new_peaks.append(peak)
        else:
            original_peaks.append(peak)

    return new_peaks, original_peaks

def determine_stable_new_peak_count(pressure_data, transition_pressure,
                                   after_pressures, n_points=N_PRESSURE_POINTS,
                                   tolerance=PEAK_TOLERANCE_2):
    """
    确定稳定的新峰数量

    参数:
        pressure_data: 压力-峰位数据字典
        transition_pressure: 相变压力点
        after_pressures: 相变后的压力点列表
        n_points: 用于判断稳定的压力点数量
        tolerance: 峰位容差

    返回:
        stable_new_peak_count: 稳定的新峰数量
        new_peak_positions: 新峰位置列表
    """
    if len(after_pressures) < n_points:
        print(f"警告：相变后压力点数量不足{n_points}个，使用所有可用点")
        check_pressures = after_pressures
    else:
        check_pressures = after_pressures[:n_points]

    # 获取相变点的峰位作为参考
    reference_peaks = pressure_data[transition_pressure]

    # 统计每个压力点的新峰数量
    new_peak_counts = []
    all_new_peaks = []

    for pressure in check_pressures:
        current_peaks = pressure_data[pressure]
        new_peaks, _ = count_new_peaks(reference_peaks, current_peaks, tolerance)
        new_peak_counts.append(len(new_peaks))
        all_new_peaks.extend(new_peaks)
        print(f"  压力 {pressure:.2f} GPa: {len(new_peaks)} 个新峰")

    # 检查新峰数量是否稳定（最后几个点相同）
    if len(new_peak_counts) >= 3:
        last_counts = new_peak_counts[-3:]
        if len(set(last_counts)) == 1:  # 最后3个点的新峰数量相同
            stable_count = last_counts[0]
            print(f"\n>>> 新峰数量已稳定：{stable_count} 个")
        else:
            stable_count = max(set(new_peak_counts), key=new_peak_counts.count)
            print(f"\n>>> 新峰数量未完全稳定，使用出现最多的值：{stable_count} 个")
    else:
        stable_count = new_peak_counts[-1] if new_peak_counts else 0
        print(f"\n>>> 压力点较少，使用最后一个点的新峰数量：{stable_count} 个")

    # 获取新峰的平均位置
    new_peak_positions = []
    if all_new_peaks:
        # 聚类相似的峰位
        all_new_peaks_sorted = sorted(all_new_peaks)
        clusters = []
        current_cluster = [all_new_peaks_sorted[0]]

        for peak in all_new_peaks_sorted[1:]:
            if peak - current_cluster[-1] < tolerance:
                current_cluster.append(peak)
            else:
                clusters.append(current_cluster)
                current_cluster = [peak]
        clusters.append(current_cluster)

        # 计算每个聚类的平均值
        new_peak_positions = [np.mean(cluster) for cluster in clusters]
        new_peak_positions = sorted(new_peak_positions)[:stable_count]

    return stable_count, new_peak_positions

# ==================== 晶系判断 ====================

def fit_cubic_lattice(peaks_2theta, hkl_list):
    """
    拟合立方晶系晶胞参数

    参数:
        peaks_2theta: 峰位列表 (2theta, 度)
        hkl_list: hkl指标列表

    返回:
        a: 晶胞参数 (Å)
        residual: 拟合残差
    """
    d_values = [two_theta_to_d(tt) for tt in peaks_2theta]

    def objective(params):
        a = params[0]
        residuals = []
        for i, (h, k, l) in enumerate(hkl_list[:len(d_values)]):
            d_calc = calculate_d_cubic((h, k, l), a)
            residuals.append((d_values[i] - d_calc)**2)
        return np.sum(residuals)

    # 初始猜测
    a_init = d_values[0] * np.sqrt(hkl_list[0][0]**2 + hkl_list[0][1]**2 + hkl_list[0][2]**2)

    result = minimize(objective, [a_init], bounds=[(1.0, 20.0)])

    if result.success:
        return result.x[0], result.fun
    else:
        return None, float('inf')

def fit_hexagonal_lattice(peaks_2theta, hkl_list):
    """
    拟合六方晶系晶胞参数

    参数:
        peaks_2theta: 峰位列表 (2theta, 度)
        hkl_list: hkl指标列表

    返回:
        (a, c): 晶胞参数 (Å)
        residual: 拟合残差
    """
    d_values = [two_theta_to_d(tt) for tt in peaks_2theta]

    def objective(params):
        a, c = params
        residuals = []
        for i, (h, k, l) in enumerate(hkl_list[:len(d_values)]):
            d_calc = calculate_d_hexagonal((h, k, l), a, c)
            residuals.append((d_values[i] - d_calc)**2)
        return np.sum(residuals)

    # 初始猜测
    a_init = d_values[0] * 2.0
    c_init = d_values[0] * 3.0

    result = minimize(objective, [a_init, c_init],
                     bounds=[(1.0, 20.0), (1.0, 20.0)])

    if result.success:
        return tuple(result.x), result.fun
    else:
        return None, float('inf')

def fit_tetragonal_lattice(peaks_2theta, hkl_list):
    """
    拟合四方晶系晶胞参数

    参数:
        peaks_2theta: 峰位列表 (2theta, 度)
        hkl_list: hkl指标列表

    返回:
        (a, c): 晶胞参数 (Å)
        residual: 拟合残差
    """
    d_values = [two_theta_to_d(tt) for tt in peaks_2theta]

    def objective(params):
        a, c = params
        residuals = []
        for i, (h, k, l) in enumerate(hkl_list[:len(d_values)]):
            d_calc = calculate_d_tetragonal((h, k, l), a, c)
            residuals.append((d_values[i] - d_calc)**2)
        return np.sum(residuals)

    # 初始猜测
    a_init = d_values[0] * 2.0
    c_init = d_values[0] * 2.0

    result = minimize(objective, [a_init, c_init],
                     bounds=[(1.0, 20.0), (1.0, 20.0)])

    if result.success:
        return tuple(result.x), result.fun
    else:
        return None, float('inf')

def fit_orthorhombic_lattice(peaks_2theta, hkl_list):
    """
    拟合正交晶系晶胞参数

    参数:
        peaks_2theta: 峰位列表 (2theta, 度)
        hkl_list: hkl指标列表

    返回:
        (a, b, c): 晶胞参数 (Å)
        residual: 拟合残差
    """
    d_values = [two_theta_to_d(tt) for tt in peaks_2theta]

    def objective(params):
        a, b, c = params
        residuals = []
        for i, (h, k, l) in enumerate(hkl_list[:len(d_values)]):
            d_calc = calculate_d_orthorhombic((h, k, l), a, b, c)
            residuals.append((d_values[i] - d_calc)**2)
        return np.sum(residuals)

    # 初始猜测
    a_init = d_values[0] * 2.0
    b_init = d_values[0] * 2.0
    c_init = d_values[0] * 2.0

    result = minimize(objective, [a_init, b_init, c_init],
                     bounds=[(1.0, 20.0), (1.0, 20.0), (1.0, 20.0)])

    if result.success:
        return tuple(result.x), result.fun
    else:
        return None, float('inf')

def determine_crystal_system(peaks_2theta, available_peak_count):
    """
    判断晶系

    参数:
        peaks_2theta: 峰位列表 (2theta, 度)
        available_peak_count: 可用于判断的新峰数量

    返回:
        best_system: 最佳匹配的晶系名称
        lattice_params: 晶胞参数
        fit_quality: 拟合质量（残差）
    """
    results = []

    print(f"\n开始晶系判断，可用峰数：{len(peaks_2theta)}")

    # 根据可用新峰数量筛选可能的晶系
    for system_key, system_info in CRYSTAL_SYSTEMS.items():
        min_peaks = system_info['min_peaks']

        # 只考虑新峰数量足够的晶系
        if available_peak_count >= min_peaks:
            hkl_list = system_info['hkl_list']

            # 尝试拟合
            try:
                if 'cubic' in system_key:
                    params, residual = fit_cubic_lattice(peaks_2theta, hkl_list)
                    if params is not None:
                        results.append({
                            'system': system_info['name'],
                            'params': {'a': params},
                            'residual': residual,
                            'key': system_key
                        })
                        print(f"  {system_info['name']}: a={params:.4f} Å, 残差={residual:.6f}")

                elif system_key == 'hexagonal':
                    params, residual = fit_hexagonal_lattice(peaks_2theta, hkl_list)
                    if params is not None:
                        results.append({
                            'system': system_info['name'],
                            'params': {'a': params[0], 'c': params[1], 'c/a': params[1]/params[0]},
                            'residual': residual,
                            'key': system_key
                        })
                        print(f"  {system_info['name']}: a={params[0]:.4f} Å, c={params[1]:.4f} Å, c/a={params[1]/params[0]:.4f}, 残差={residual:.6f}")

                elif system_key == 'tetragonal':
                    params, residual = fit_tetragonal_lattice(peaks_2theta, hkl_list)
                    if params is not None:
                        results.append({
                            'system': system_info['name'],
                            'params': {'a': params[0], 'c': params[1], 'c/a': params[1]/params[0]},
                            'residual': residual,
                            'key': system_key
                        })
                        print(f"  {system_info['name']}: a={params[0]:.4f} Å, c={params[1]:.4f} Å, c/a={params[1]/params[0]:.4f}, 残差={residual:.6f}")

                elif system_key == 'orthorhombic':
                    params, residual = fit_orthorhombic_lattice(peaks_2theta, hkl_list)
                    if params is not None:
                        results.append({
                            'system': system_info['name'],
                            'params': {'a': params[0], 'b': params[1], 'c': params[2]},
                            'residual': residual,
                            'key': system_key
                        })
                        print(f"  {system_info['name']}: a={params[0]:.4f} Å, b={params[1]:.4f} Å, c={params[2]:.4f} Å, 残差={residual:.6f}")

            except Exception as e:
                print(f"  {system_info['name']}: 拟合失败 ({str(e)})")
                continue

    if not results:
        print("\n>>> 未找到合适的晶系")
        return None, None, None

    # 选择残差最小的结果
    best_result = min(results, key=lambda x: x['residual'])

    print(f"\n>>> 最佳匹配晶系：{best_result['system']}")
    print(f"    晶胞参数：{best_result['params']}")
    print(f"    拟合残差：{best_result['residual']:.6f}")

    return best_result['system'], best_result['params'], best_result['residual']

# ==================== 主分析流程 ====================

def analyze_phase_transition(csv_path):
    """
    主分析函数：执行完整的相变分析流程

    参数:
        csv_path: CSV文件路径

    返回:
        analysis_results: 分析结果字典
    """
    print("="*70)
    print("晶系判断和相变识别分析")
    print("="*70)

    # 1. 读取数据
    print("\n[步骤 1] 读取CSV数据...")
    pressure_data = read_pressure_peak_data(csv_path)
    print(f"  共读取 {len(pressure_data)} 个压力点")
    for pressure in sorted(pressure_data.keys()):
        print(f"    {pressure:.2f} GPa: {len(pressure_data[pressure])} 个峰")

    # 2. 识别相变点
    print("\n[步骤 2] 识别相变点...")
    transition_pressure, before_pressures, after_pressures = \
        find_phase_transition_point(pressure_data, PEAK_TOLERANCE_1)

    # 3. 相变前的晶系判断
    print("\n[步骤 3] 分析相变前的晶系...")
    if before_pressures:
        # 使用相变前所有压力点的所有峰
        before_all_peaks = []
        for p in before_pressures:
            before_all_peaks.extend(pressure_data[p])
        before_all_peaks = sorted(list(set(before_all_peaks)))

        print(f"  相变前使用 {len(before_all_peaks)} 个峰进行晶系判断")
        before_system, before_params, before_quality = \
            determine_crystal_system(before_all_peaks, len(before_all_peaks))
    else:
        before_system = None
        before_params = None
        before_quality = None

    # 4. 相变后的新峰分析
    results_after = {
        'transition_pressure': transition_pressure,
        'before_system': before_system,
        'before_params': before_params,
        'after_analysis': []
    }

    if transition_pressure is not None and len(after_pressures) > 0:
        print("\n[步骤 4] 分析相变后的新峰...")

        # 确定稳定的新峰数量
        stable_count, new_peak_positions = determine_stable_new_peak_count(
            pressure_data, transition_pressure, after_pressures,
            N_PRESSURE_POINTS, PEAK_TOLERANCE_2
        )

        print(f"\n  稳定新峰位置：{new_peak_positions}")

        # 5. 对相变后的每个压力点进行分析
        print("\n[步骤 5] 分析相变后各压力点的晶系...")

        # 获取相变前最后一个压力点的峰位作为参考
        reference_pressure = before_pressures[-1] if before_pressures else transition_pressure
        reference_peaks = pressure_data[reference_pressure]

        for pressure in after_pressures:
            print(f"\n  分析压力点：{pressure:.2f} GPa")
            current_peaks = pressure_data[pressure]

            # 分离新峰和原峰
            new_peaks, original_peaks = count_new_peaks(
                reference_peaks, current_peaks, PEAK_TOLERANCE_3
            )

            print(f"    新峰 ({len(new_peaks)} 个)：{new_peaks}")
            print(f"    原峰 ({len(original_peaks)} 个)：{original_peaks}")

            # 使用新峰判断新相晶系
            if len(new_peaks) >= 1:
                new_phase_system, new_phase_params, new_phase_quality = \
                    determine_crystal_system(new_peaks, stable_count)

                results_after['after_analysis'].append({
                    'pressure': pressure,
                    'new_peaks': new_peaks,
                    'original_peaks': original_peaks,
                    'new_phase_system': new_phase_system,
                    'new_phase_params': new_phase_params,
                    'fit_quality': new_phase_quality
                })

    # 6. 输出总结
    print("\n" + "="*70)
    print("分析总结")
    print("="*70)

    if transition_pressure:
        print(f"\n相变压力：{transition_pressure:.2f} GPa")
    else:
        print("\n未检测到相变")

    if before_system:
        print(f"\n相变前晶系：{before_system}")
        print(f"晶胞参数：{before_params}")

    if results_after['after_analysis']:
        print(f"\n相变后分析：")
        for result in results_after['after_analysis']:
            if result['new_phase_system']:
                print(f"\n  压力 {result['pressure']:.2f} GPa:")
                print(f"    新相晶系：{result['new_phase_system']}")
                print(f"    晶胞参数：{result['new_phase_params']}")

    print("\n" + "="*70)

    return results_after

# ==================== 命令行接口 ====================

def main():
    """主函数"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python phase_transition_analysis.py <csv_file_path>")
        print("\n示例:")
        print("  python phase_transition_analysis.py data.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        results = analyze_phase_transition(csv_path)

        # 可选：保存结果到JSON文件
        import json
        output_path = csv_path.replace('.csv', '_phase_analysis.json')

        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        results_serializable = convert_to_serializable(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        print(f"\n分析结果已保存到：{output_path}")

    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
