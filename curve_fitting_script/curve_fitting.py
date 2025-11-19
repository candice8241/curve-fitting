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
from scipy.signal import find_peaks, savgol_filter
import os
import pandas as pd
from scipy.special import wofz

# ---------- Helper Functions ----------
def estimate_fwhm_robust(x, y, peak_idx, smooth=True):
    """
    Robust FWHM estimation using interpolation
    """
    if smooth and len(y) > 11:
        try:
            y_smooth = savgol_filter(y, min(11, len(y)//2*2+1), 3)
        except:
            y_smooth = y
    else:
        y_smooth = y

    peak_height = y_smooth[peak_idx]

    # Estimate local baseline from edges
    n_edge = max(3, len(y) // 10)
    baseline = (np.mean(y_smooth[:n_edge]) + np.mean(y_smooth[-n_edge:])) / 2

    half_max = (peak_height + baseline) / 2

    # Find left half-max point with interpolation
    left_x = x[0]
    for j in range(peak_idx, 0, -1):
        if y_smooth[j] <= half_max:
            if y_smooth[j+1] != y_smooth[j]:
                frac = (half_max - y_smooth[j]) / (y_smooth[j+1] - y_smooth[j])
                left_x = x[j] + frac * (x[j+1] - x[j])
            else:
                left_x = x[j]
            break

    # Find right half-max point with interpolation
    right_x = x[-1]
    for j in range(peak_idx, len(y_smooth)-1):
        if y_smooth[j] <= half_max:
            if y_smooth[j-1] != y_smooth[j]:
                frac = (half_max - y_smooth[j]) / (y_smooth[j-1] - y_smooth[j])
                right_x = x[j] - frac * (x[j] - x[j-1])
            else:
                right_x = x[j]
            break

    fwhm = abs(right_x - left_x)

    # Sanity check
    dx = np.mean(np.diff(x))
    if fwhm < dx * 2:
        fwhm = dx * 8

    return fwhm, baseline

def find_valley_between_peaks(x, y, peak_idx1, peak_idx2):
    """
    Find the valley (minimum point) between two peaks
    Returns the index and value of the minimum point
    """
    if peak_idx1 > peak_idx2:
        peak_idx1, peak_idx2 = peak_idx2, peak_idx1

    # Search for minimum between the two peaks
    search_region = y[peak_idx1:peak_idx2+1]
    if len(search_region) == 0:
        return peak_idx1, y[peak_idx1]

    local_min_idx = np.argmin(search_region)
    global_min_idx = peak_idx1 + local_min_idx

    return global_min_idx, y[global_min_idx]

def group_peaks_by_distance(x, y, peak_indices, fwhm_multiplier=2.5):
    """
    Group peaks based on their distance relative to FWHM
    Peaks closer than fwhm_multiplier * average_fwhm are grouped together

    Returns: list of groups, where each group is a list of peak indices
    """
    if len(peak_indices) <= 1:
        return [[idx] for idx in peak_indices]

    # Sort peaks by position
    sorted_indices = sorted(peak_indices, key=lambda idx: x[idx])

    # Estimate FWHM for each peak
    fwhm_estimates = []
    for idx in sorted_indices:
        window_size = 50
        left = max(0, idx - window_size)
        right = min(len(x), idx + window_size)
        x_local = x[left:right]
        y_local = y[left:right]
        local_peak_idx = idx - left
        fwhm, _ = estimate_fwhm_robust(x_local, y_local, local_peak_idx)
        fwhm_estimates.append(fwhm)

    # Group overlapping peaks
    peak_groups = []
    current_group = [sorted_indices[0]]
    current_fwhms = [fwhm_estimates[0]]

    for i in range(1, len(sorted_indices)):
        prev_idx = sorted_indices[i-1]
        curr_idx = sorted_indices[i]
        distance = abs(x[curr_idx] - x[prev_idx])
        avg_fwhm = (fwhm_estimates[i-1] + fwhm_estimates[i]) / 2

        if distance < avg_fwhm * fwhm_multiplier:
            current_group.append(curr_idx)
            current_fwhms.append(fwhm_estimates[i])
        else:
            peak_groups.append((current_group, current_fwhms))
            current_group = [curr_idx]
            current_fwhms = [fwhm_estimates[i]]

    peak_groups.append((current_group, current_fwhms))

    return peak_groups

def create_valley_background(x, y, peak_indices):
    """
    Create background curve by connecting valleys between adjacent peaks
    For grouped peaks, the background is the piecewise linear interpolation
    through the minimum points between adjacent peaks

    Returns: background array for the data range
    """
    if len(peak_indices) <= 1:
        # Single peak: use edge-based linear background
        return None

    # Sort peaks by position
    sorted_peaks = sorted(peak_indices, key=lambda idx: x[idx])

    # Find valleys between adjacent peaks
    valley_points = []

    # Add left edge point
    left_edge_idx = max(0, sorted_peaks[0] - 30)
    valley_points.append((x[left_edge_idx], y[left_edge_idx]))

    # Find valleys between each pair of adjacent peaks
    for i in range(len(sorted_peaks) - 1):
        valley_idx, valley_val = find_valley_between_peaks(x, y, sorted_peaks[i], sorted_peaks[i+1])
        valley_points.append((x[valley_idx], valley_val))

    # Add right edge point
    right_edge_idx = min(len(x) - 1, sorted_peaks[-1] + 30)
    valley_points.append((x[right_edge_idx], y[right_edge_idx]))

    # Create piecewise linear interpolation
    valley_x = np.array([p[0] for p in valley_points])
    valley_y = np.array([p[1] for p in valley_points])

    return valley_x, valley_y

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

# ---------- Multi-peak fitting functions ----------
def fit_multi_pseudo_voigt(x, y, peak_indices, fwhm_estimates, use_valley_bg=True):
    """
    Fit multiple peaks simultaneously with valley-based background for grouped peaks

    Args:
        x, y: data arrays
        peak_indices: list of peak indices in the group
        fwhm_estimates: list of FWHM estimates for each peak
        use_valley_bg: whether to use valley-based background (for grouped peaks)

    Returns:
        popt: optimized parameters
        background_info: information about the background used
    """
    n_peaks = len(peak_indices)
    dx = np.mean(np.diff(x))

    # Create fitting window
    sorted_peaks = sorted(peak_indices, key=lambda idx: x[idx])
    left_fwhm = fwhm_estimates[0]
    right_fwhm = fwhm_estimates[-1]

    # Window size depends on whether peaks are grouped
    if n_peaks > 1:
        window_multiplier = 3  # Wider window for grouped peaks
    else:
        window_multiplier = 2.5

    window_left = x[sorted_peaks[0]] - left_fwhm * window_multiplier
    window_right = x[sorted_peaks[-1]] + right_fwhm * window_multiplier

    left_idx = max(0, np.searchsorted(x, window_left))
    right_idx = min(len(x), np.searchsorted(x, window_right))

    x_fit = x[left_idx:right_idx]
    y_fit = y[left_idx:right_idx]

    if len(x_fit) < 5:
        return None, None, (left_idx, right_idx)

    # Create background based on method
    if use_valley_bg and n_peaks > 1:
        # Valley-based background for grouped peaks
        valley_result = create_valley_background(x, y, sorted_peaks)
        if valley_result is not None:
            valley_x, valley_y = valley_result
            background = np.interp(x_fit, valley_x, valley_y)
            bg_info = {'type': 'valley', 'valley_x': valley_x, 'valley_y': valley_y}
        else:
            # Fallback to linear
            bg_left = np.mean(y_fit[:3])
            bg_right = np.mean(y_fit[-3:])
            slope = (bg_right - bg_left) / (x_fit[-1] - x_fit[0])
            background = bg_left + slope * (x_fit - x_fit[0])
            bg_info = {'type': 'linear', 'bg_left': bg_left, 'slope': slope, 'x_ref': x_fit[0]}
    else:
        # Linear background for single peaks
        bg_left = np.mean(y_fit[:3])
        bg_right = np.mean(y_fit[-3:])
        slope = (bg_right - bg_left) / (x_fit[-1] - x_fit[0])
        background = bg_left + slope * (x_fit - x_fit[0])
        bg_info = {'type': 'linear', 'bg_left': bg_left, 'slope': slope, 'x_ref': x_fit[0]}

    # Subtract background for fitting
    y_fit_corrected = y_fit - background

    # Build initial parameters
    p0 = []
    bounds_lower = []
    bounds_upper = []

    for i, peak_idx in enumerate(sorted_peaks):
        cen_guess = x[peak_idx]
        fwhm_est = fwhm_estimates[i]

        sig_guess = fwhm_est / 2.355
        gam_guess = fwhm_est / 2

        # Amplitude estimation
        peak_intensity = y[peak_idx] - np.interp(x[peak_idx], x_fit, background)
        if peak_intensity <= 0:
            peak_intensity = np.max(y_fit_corrected) * 0.3

        # Different parameters for grouped vs isolated peaks
        if n_peaks > 1:
            # Grouped peaks: more relaxed constraints
            amp_guess = peak_intensity * sig_guess * np.sqrt(2 * np.pi) * 1.5
            center_tolerance = fwhm_est * 0.8
            amp_multiplier = 10
        else:
            # Single peak: tighter constraints
            amp_guess = peak_intensity * sig_guess * np.sqrt(2 * np.pi)
            center_tolerance = fwhm_est * 0.5
            amp_multiplier = 5

        y_range = np.max(y_fit_corrected) - np.min(y_fit_corrected)
        amp_upper = y_range * sig_guess * np.sqrt(2 * np.pi) * amp_multiplier

        p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, 0.5])
        bounds_lower.extend([0, cen_guess - center_tolerance, dx * 0.5, dx * 0.5, 0])
        bounds_upper.extend([amp_upper, cen_guess + center_tolerance, fwhm_est * 3, fwhm_est * 3, 1.0])

    # Define multi-peak function
    def multi_peak_func(x, *params):
        y = np.zeros_like(x)
        for i in range(n_peaks):
            offset = i * 5
            amp, cen, sig, gam, eta = params[offset:offset+5]
            y += pseudo_voigt(x, amp, cen, sig, gam, eta)
        return y

    # Perform fitting
    try:
        if n_peaks > 1:
            # More iterations for grouped peaks
            popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_corrected,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               method='trf', maxfev=30000, ftol=1e-9, xtol=1e-9)
        else:
            popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_corrected,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               method='trf', maxfev=10000)
    except Exception as e:
        try:
            # Fallback to dogbox method
            popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_corrected,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               method='dogbox', maxfev=50000)
        except Exception:
            return None, bg_info, (left_idx, right_idx)

    return popt, bg_info, (left_idx, right_idx)

# ---------- Process a single file ----------
def process_file(file_path, save_dir):
    try:
        with open(file_path, encoding='latin1') as f:
            data = np.genfromtxt(f, comments="#")
        x = data[:, 0]
        y = data[:, 1]
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    filename = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nProcessing file: {filename}")

    all_peaks, _ = find_peaks(y, distance=30)

    filtered_peaks = []
    for i in all_peaks:
        if 90 <= i < len(y) - 100:
            left = y[i - 90]
            right = y[i + 90]
            min_neighbor = min(left, right)
            if y[i] >= min_neighbor * 1.15:
                filtered_peaks.append(i)

    peaks = filtered_peaks
    print(f"   Detected {len(peaks)} peaks (with +15% neighbor filter)")
    if len(peaks) == 0:
        print("   No valid peaks found, skipping this file.")
        return

    # Group peaks by distance
    peak_groups = group_peaks_by_distance(x, y, peaks, fwhm_multiplier=2.5)

    print(f"   Grouped into {len(peak_groups)} group(s):")
    for g_idx, (group, fwhms) in enumerate(peak_groups):
        if len(group) > 1:
            print(f"     Group {g_idx+1}: {len(group)} peaks (close together, using valley background)")
        else:
            print(f"     Group {g_idx+1}: 1 peak (isolated, using linear background)")

    results = []

    # Calculate total number of plots needed
    total_plots = sum(1 if len(group) == 1 else 1 for group, _ in peak_groups)
    subplot_cols = 3
    subplot_rows = int(np.ceil(total_plots / subplot_cols))
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(5*subplot_cols, 4*subplot_rows))
    if total_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    plot_idx = 0
    peak_counter = 0

    for g_idx, (group, fwhms) in enumerate(peak_groups):
        n_peaks_in_group = len(group)
        use_valley_bg = n_peaks_in_group > 1

        # Fit the group
        popt, bg_info, (left_idx, right_idx) = fit_multi_pseudo_voigt(
            x, y, group, fwhms, use_valley_bg=use_valley_bg
        )

        if popt is None:
            print(f"   Group {g_idx+1} fit failed")
            if plot_idx < len(axs):
                axs[plot_idx].text(0.3, 0.5, "Fit failed", transform=axs[plot_idx].transAxes,
                                  fontsize=12, color='red')
                plot_idx += 1
            peak_counter += n_peaks_in_group
            continue

        # Extract data for plotting
        x_local = x[left_idx:right_idx]
        y_local = y[left_idx:right_idx]
        x_smooth = np.linspace(x_local.min(), x_local.max(), 5000)

        # Calculate background for plotting
        if bg_info['type'] == 'valley':
            bg_smooth = np.interp(x_smooth, bg_info['valley_x'], bg_info['valley_y'])
            bg_local = np.interp(x_local, bg_info['valley_x'], bg_info['valley_y'])
        else:
            bg_smooth = bg_info['bg_left'] + bg_info['slope'] * (x_smooth - bg_info['x_ref'])
            bg_local = bg_info['bg_left'] + bg_info['slope'] * (x_local - bg_info['x_ref'])

        # Calculate total fit
        y_fit_total = np.zeros_like(x_smooth)
        for i in range(n_peaks_in_group):
            offset = i * 5
            amp, cen, sig, gam, eta = popt[offset:offset+5]
            y_fit_total += pseudo_voigt(x_smooth, amp, cen, sig, gam, eta)

        y_fit_with_bg = y_fit_total + bg_smooth

        # Store results
        for i in range(n_peaks_in_group):
            offset = i * 5
            amp, cen, sig, gam, eta = popt[offset:offset+5]
            results.append({
                "Peak #": peak_counter + i + 1,
                "Center": cen,
                "Amplitude": amp,
                "Sigma": sig,
                "Gamma": gam,
                "Eta": eta,
                "Group": g_idx + 1,
                "Group_Size": n_peaks_in_group
            })

        # Plot
        if plot_idx < len(axs):
            ax = axs[plot_idx]
            ax.plot(x_local, y_local, color='black', linewidth=1, label="Raw Data")
            ax.plot(x_smooth, y_fit_with_bg, color='#BA55D3', linestyle='--',
                   linewidth=2, label="Total Fit")

            # Plot background
            if bg_info['type'] == 'valley':
                # Plot valley points and connecting lines
                ax.plot(bg_info['valley_x'], bg_info['valley_y'], 'o-',
                       color='#FF69B4', markersize=4, linewidth=1.5,
                       label="Valley BG", alpha=0.8)
            else:
                ax.plot(x_smooth, bg_smooth, color='#FF69B4', linestyle='-',
                       linewidth=1.5, label="Linear BG")

            # Plot individual peak components for grouped peaks
            if n_peaks_in_group > 1:
                colors = plt.cm.tab10(np.linspace(0, 1, n_peaks_in_group))
                for i in range(n_peaks_in_group):
                    offset = i * 5
                    amp, cen, sig, gam, eta = popt[offset:offset+5]
                    y_component = pseudo_voigt(x_smooth, amp, cen, sig, gam, eta) + bg_smooth
                    ax.plot(x_smooth, y_component, '--', color=colors[i],
                           linewidth=1, alpha=0.7, label=f"P{peak_counter+i+1}")

            if n_peaks_in_group == 1:
                ax.set_title(f"Peak {peak_counter+1} @ {popt[1]:.3f}")
            else:
                peak_nums = [peak_counter + i + 1 for i in range(n_peaks_in_group)]
                ax.set_title(f"Group {g_idx+1}: Peaks {peak_nums}")

            ax.set_xlabel("2theta (degree)")
            ax.set_ylabel("Intensity")
            ax.legend(fontsize=8, loc='best')
            ax.set_facecolor("white")
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        peak_counter += n_peaks_in_group

    # Remove unused subplots
    for j in range(plot_idx, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"{filename} - {fit_method.capitalize()} Fit (Distance-based Grouping)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig_path = os.path.join(save_dir, f"{filename}_fit.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    df = pd.DataFrame(results)
    df["File"] = filename
    csv_path = os.path.join(save_dir, f"{filename}_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"   Results saved: {csv_path}")

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
