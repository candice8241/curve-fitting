# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 09:45:31 2025

@author: 16961
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 10:49:55 2025
@author: candicewang928@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
import pandas as pd
from scipy.special import wofz

# ---------- Voigt ----------
def voigt(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def fit_voigt(xdata, ydata, p0=None):
    if p0 is None:
        amplitude_guess = np.max(ydata)
        center_guess = xdata[np.argmax(ydata)]
        sigma_guess = np.std(xdata) / 5
        gamma_guess = sigma_guess
        p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess]
    bounds = ([0, xdata.min(), 0, 0], [np.inf, xdata.max(), np.inf, np.inf])
    popt, pcov = curve_fit(voigt, xdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
    return popt, pcov

# ---------- Pseudo-Voigt ----------
def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian

def fit_pseudo_voigt(xdata, ydata, p0=None):
    if p0 is None:
        amplitude_guess = np.max(ydata)
        center_guess = xdata[np.argmax(ydata)]
        sigma_guess = np.std(xdata) / 5
        gamma_guess = sigma_guess
        eta_guess = 0.5
        p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess, eta_guess]
    bounds = ([0, xdata.min(), 0, 0, 0], [np.inf, xdata.max(), np.inf, np.inf, 1.0])
    popt, pcov = curve_fit(pseudo_voigt, xdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
    return popt, pcov

# ---------- FITTING METHOD ----------
fit_method = "pseudo"  # Choose "voigt" or "pseudo"
#fit_method = "voigt"

# ---------- 多峰同时拟合 (解决P6/P7重叠问题) ----------
def fit_multi_peaks(x, y, peak_indices, fit_method="pseudo"):
    """
    同时拟合多个重叠峰
    改进点：
    1. 更好的初始参数估计
    2. 小峰的约束更严格
    3. 背景斜率限制
    """
    n_peaks = len(peak_indices)
    dx = np.mean(np.diff(x))

    # 建立拟合窗口
    window = 55
    left_idx = max(0, min(peak_indices) - window)
    right_idx = min(len(x), max(peak_indices) + window)

    x_fit = x[left_idx:right_idx]
    y_fit = y[left_idx:right_idx]

    # 估计背景 - 使用边缘点，减小斜率影响
    bg_left = np.mean(y_fit[:5])
    bg_right = np.mean(y_fit[-5:])
    bg_slope = (bg_right - bg_left) / (x_fit[-1] - x_fit[0])

    # 关键改进：对于重叠峰，减小背景斜率
    bg_slope *= 0.3  # 减少70%的斜率

    background = bg_left + bg_slope * (x_fit - x_fit[0])
    y_fit_input = y_fit - background

    # 构建初始参数和边界
    p0 = []
    bounds_lower = []
    bounds_upper = []

    # 找出最大峰和最小峰的高度比
    heights = [y[idx] - background[idx - left_idx] for idx in peak_indices]
    max_height = max(heights)

    for i, idx in enumerate(peak_indices):
        local_idx = idx - left_idx
        center = x[idx]
        height = max(heights[i], max_height * 0.05)  # 至少5%

        # 估计FWHM
        sigma_guess = 0.02  # 默认值
        gamma_guess = 0.02

        # 振幅估计
        amp_guess = height * sigma_guess * np.sqrt(2 * np.pi)

        # 判断是否为小峰
        is_small = height < max_height * 0.5

        # 设置边界
        if is_small:
            # 小峰：更严格的约束防止被大峰吞掉
            amp_min = amp_guess * 0.3
            amp_max = amp_guess * 5
            center_tol = 0.05  # 更小的中心偏移容差
            sig_max = 0.05
            gam_max = 0.05
        else:
            # 大峰：正常约束
            amp_min = 0
            amp_max = amp_guess * 20
            center_tol = 0.1
            sig_max = 0.1
            gam_max = 0.1

        if fit_method == "voigt":
            p0.extend([amp_guess, center, sigma_guess, gamma_guess])
            bounds_lower.extend([amp_min, center - center_tol, dx, dx])
            bounds_upper.extend([amp_max, center + center_tol, sig_max, gam_max])
        else:
            p0.extend([amp_guess, center, sigma_guess, gamma_guess, 0.5])
            bounds_lower.extend([amp_min, center - center_tol, dx, dx, 0])
            bounds_upper.extend([amp_max, center + center_tol, sig_max, gam_max, 1.0])

    # 定义多峰函数
    if fit_method == "voigt":
        def multi_func(x, *params):
            y = np.zeros_like(x)
            for i in range(n_peaks):
                y += voigt(x, *params[i*4:(i+1)*4])
            return y
    else:
        def multi_func(x, *params):
            y = np.zeros_like(x)
            for i in range(n_peaks):
                y += pseudo_voigt(x, *params[i*5:(i+1)*5])
            return y

    # 拟合
    popt, pcov = curve_fit(multi_func, x_fit, y_fit_input,
                          p0=p0, bounds=(bounds_lower, bounds_upper),
                          maxfev=50000, ftol=1e-9, xtol=1e-9)

    return popt, x_fit, y_fit, background

# ---------- Process a single file ----------
def process_file(file_path, save_dir):
    try:
        with open(file_path, encoding='latin1') as f:
            data = np.genfromtxt(f, comments="#")
        x = data[:, 0]
        y = data[:, 1]
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return

    filename = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n📄 Processing file: {filename}")

    all_peaks, _ = find_peaks(y, distance=30)

    filtered_peaks = []
    for i in all_peaks:
        if 1 <= i < len(y) - 100:
            left = y[i - 90]
            right = y[i + 90]
            min_neighbor = min(left, right)
            if y[i] >= min_neighbor * 1.15:
                filtered_peaks.append(i)

    peaks = filtered_peaks
    print(f"   ➤ Detected {len(peaks)} peaks (with +15% neighbor filter)")
    if len(peaks) == 0:
        print("   ⚠️ No valid peaks found, skipping this file.")
        return

    # ---------- 新增：将重叠峰分组 ----------
    peak_groups = []
    current_group = [peaks[0]]

    for i in range(1, len(peaks)):
        distance = x[peaks[i]] - x[peaks[i-1]]
        # 如果两峰间距小于0.15度，认为是重叠峰
        if distance < 0.15:
            current_group.append(peaks[i])
        else:
            peak_groups.append(current_group)
            current_group = [peaks[i]]
    peak_groups.append(current_group)

    # 报告重叠峰组
    for g_idx, group in enumerate(peak_groups):
        if len(group) > 1:
            centers = [f"{x[p]:.3f}" for p in group]
            print(f"   ➤ 发现重叠峰组 {g_idx+1}: {len(group)}个峰 @ {', '.join(centers)}")

    results = []
    subplot_cols = 3
    subplot_rows = int(np.ceil(len(peak_groups) / subplot_cols))
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(5*subplot_cols, 4*subplot_rows))
    if len(peak_groups) == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    peak_counter = 0

    for g_idx, group in enumerate(peak_groups):

        if len(group) == 1:
            # ---------- 单峰：原始方法 ----------
            peak = group[0]
            window = 55
            left = max(0, peak - window)
            right = min(len(x), peak + window)
            x_local = x[left:right]
            y_local = y[left:right]

            # Background subtraction
            bg_left = np.mean(y_local[:1])
            bg_right = np.mean(y_local[-1:])
            slope = (bg_right - bg_left) / (x_local[-1] - x_local[0])
            background = bg_left + slope * (x_local - x_local[0])
            y_fit_input = y_local - background

            x_smooth = np.linspace(x_local.min(), x_local.max(), 5000)

            try:
                if fit_method == "voigt":
                    popt, _ = fit_voigt(x_local, y_fit_input)
                    y_fit_corrected = voigt(x_smooth, *popt) + (bg_left + slope * (x_smooth - x_local[0]))
                    fit_label = "Voigt Fit"
                    peak_counter += 1
                    results.append({
                        "Peak #": peak_counter,
                        "Center": popt[1],
                        "Amplitude": popt[0],
                        "Sigma": popt[2],
                        "Gamma": popt[3],
                        "Eta": "N/A"
                    })
                else:
                    popt, _ = fit_pseudo_voigt(x_local, y_fit_input)
                    y_fit_corrected = pseudo_voigt(x_smooth, *popt) + (bg_left + slope * (x_smooth - x_local[0]))
                    bg_line = bg_left + slope * (x_smooth - x_local[0])
                    fit_label = "Pseudo-Voigt Fit"
                    peak_counter += 1
                    results.append({
                        "Peak #": peak_counter,
                        "Center": popt[1],
                        "Amplitude": popt[0],
                        "Sigma": popt[2],
                        "Gamma": popt[3],
                        "Eta": popt[4]
                    })

                ax = axs[g_idx]
                ax.plot(x_local, y_local, color='black', label="Raw Data")
                ax.plot(x_smooth, y_fit_corrected, color='#BA55D3', linestyle='--', linewidth=2, label=fit_label)
                ax.plot(x_smooth, bg_line, color='#FF69B4', linestyle='-', linewidth=1.5, label="Background")
                ax.set_title(f"Peak {peak_counter} @ {popt[1]:.3f}")
                ax.set_xlabel("2θ (degree)")
                ax.set_ylabel("Intensity")
                ax.legend()
                ax.set_facecolor("white")
                ax.grid(False)

            except RuntimeError:
                print(f"⚠️ Peak {peak_counter+1} fit failed 💔")
                axs[g_idx].text(0.3, 0.5, "Fit failed", transform=axs[g_idx].transAxes, fontsize=12, color='red')

        else:
            # ---------- 多峰：同时拟合 ----------
            try:
                popt, x_fit, y_fit, background = fit_multi_peaks(x, y, group, fit_method)

                x_smooth = np.linspace(x_fit.min(), x_fit.max(), 5000)
                bg_line = background[0] + (background[-1] - background[0]) / (x_fit[-1] - x_fit[0]) * (x_smooth - x_fit[0])

                # 计算总拟合和各分量
                n_params = 4 if fit_method == "voigt" else 5
                y_total = np.zeros_like(x_smooth)

                ax = axs[g_idx]
                ax.plot(x_fit, y_fit, color='black', label="Raw Data")

                colors = ['#BA55D3', '#4169E1', '#32CD32', '#FF6347']

                for i in range(len(group)):
                    params = popt[i*n_params:(i+1)*n_params]

                    if fit_method == "voigt":
                        y_peak = voigt(x_smooth, *params)
                        peak_counter += 1
                        results.append({
                            "Peak #": peak_counter,
                            "Center": params[1],
                            "Amplitude": params[0],
                            "Sigma": params[2],
                            "Gamma": params[3],
                            "Eta": "N/A"
                        })
                    else:
                        y_peak = pseudo_voigt(x_smooth, *params)
                        peak_counter += 1
                        results.append({
                            "Peak #": peak_counter,
                            "Center": params[1],
                            "Amplitude": params[0],
                            "Sigma": params[2],
                            "Gamma": params[3],
                            "Eta": params[4]
                        })

                    y_total += y_peak

                    # 画各分量
                    color = colors[i % len(colors)]
                    ax.plot(x_smooth, y_peak + bg_line, '--', color=color,
                           linewidth=1.5, label=f"P{peak_counter}", alpha=0.8)

                # 画总拟合
                ax.plot(x_smooth, y_total + bg_line, 'r-', linewidth=2, label="Total Fit")
                ax.plot(x_smooth, bg_line, color='#FF69B4', linestyle='-', linewidth=1.5, label="Background")

                centers_str = ", ".join([f"{x[p]:.3f}" for p in group])
                ax.set_title(f"Peaks @ {centers_str}")
                ax.set_xlabel("2θ (degree)")
                ax.set_ylabel("Intensity")
                ax.legend(fontsize=8)
                ax.set_facecolor("white")
                ax.grid(False)

            except Exception as e:
                print(f"⚠️ Multi-peak fit failed: {e}")
                axs[g_idx].text(0.3, 0.5, "Fit failed", transform=axs[g_idx].transAxes, fontsize=12, color='red')

    for j in range(len(peak_groups), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"{filename} - {fit_method.capitalize()} Fit", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig_path = os.path.join(save_dir, f"{filename}_fit.png")
    plt.savefig(fig_path)
    plt.close()

    df = pd.DataFrame(results)
    df["File"] = filename
    csv_path = os.path.join(save_dir, f"{filename}_results.csv")
    df.to_csv(csv_path, index=False)

    return df

# ---------- Main ----------
def main():
    folder = r"D:\HEPS\ID31\dioptas_data\Al0"
    save_dir = os.path.join(folder, "fit_output")
    os.makedirs(save_dir, exist_ok=True)

    files = [f for f in os.listdir(folder) if f.endswith(".xy")]
    files.sort()

    all_dfs = []
    for fname in files:
        fpath = os.path.join(folder, fname)
        df = process_file(fpath, save_dir)
        if df is not None:
            all_dfs.append(df)
            all_dfs.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))  # add blank line

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv_path = os.path.join(save_dir, "all_results.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n📦 Total results saved to: {combined_csv_path}")

if __name__ == "__main__":
    main()
