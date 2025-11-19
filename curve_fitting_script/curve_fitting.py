# -*- coding: utf-8 -*-
"""
Improved Peak Fitting with Better Handling of Overlapping Peaks
@author: candicewang928@gmail.com
Enhanced to handle:
- Close peaks with large amplitude differences (P6/P7 problem)
- Better initial parameter estimation
- Improved baseline handling
- Constraints to prevent small peaks from being absorbed
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
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

# ---------- Helper Functions ----------
def estimate_fwhm_robust(x, y, peak_idx):
    """
    Robust FWHM estimation using interpolation
    """
    if len(y) > 11:
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

def calculate_fwhm(sigma, gamma, eta):
    """Calculate FWHM from Pseudo-Voigt parameters"""
    fwhm_g = 2.355 * sigma
    fwhm_l = 2 * gamma
    return eta * fwhm_l + (1 - eta) * fwhm_g

def calculate_area(amplitude, sigma, gamma, eta):
    """Calculate integrated area"""
    area_g = amplitude * sigma * np.sqrt(2 * np.pi)
    area_l = amplitude * np.pi * gamma
    return eta * area_l + (1 - eta) * area_g

# ---------- FITTING METHOD ----------
fit_method = "pseudo"  # Choose "voigt" or "pseudo"
#fit_method = "voigt"

# ---------- Improved Multi-Peak Fitting ----------
def fit_overlapping_peaks(x, y, peak_indices, fit_method="pseudo"):
    """
    Fit multiple overlapping peaks simultaneously with improved handling
    of peaks with large amplitude differences.

    Key improvements:
    1. Better initial parameter estimation for each peak
    2. Amplitude constraints based on actual peak heights
    3. Tighter center constraints for small peaks
    4. Flatter baseline for overlapping peak groups
    """
    n_peaks = len(peak_indices)
    dx = np.mean(np.diff(x))

    # Step 1: Estimate parameters for each peak individually
    peak_params = []
    for idx in peak_indices:
        # Use local window for FWHM estimation
        window = 30
        left = max(0, idx - window)
        right = min(len(x), idx + window)
        x_local = x[left:right]
        y_local = y[left:right]
        local_idx = idx - left

        fwhm, baseline = estimate_fwhm_robust(x_local, y_local, local_idx)
        peak_height = y[idx] - baseline

        peak_params.append({
            'idx': idx,
            'center': x[idx],
            'height': max(peak_height, y[idx] * 0.1),  # Ensure positive
            'fwhm': fwhm,
            'baseline': baseline
        })

    # Step 2: Determine fitting window
    # Use wider window for better baseline estimation
    centers = [p['center'] for p in peak_params]
    fwhms = [p['fwhm'] for p in peak_params]

    window_left = min(centers) - max(fwhms) * 4
    window_right = max(centers) + max(fwhms) * 4

    left_idx = max(0, np.searchsorted(x, window_left))
    right_idx = min(len(x), np.searchsorted(x, window_right))

    x_fit = x[left_idx:right_idx]
    y_fit = y[left_idx:right_idx]

    if len(x_fit) < 10:
        return None, None

    # Step 3: Estimate baseline with reduced slope for overlapping peaks
    # Use polynomial fit on edges only to avoid being influenced by peaks
    edge_n = max(5, len(x_fit) // 10)

    # Get edge points
    x_edges = np.concatenate([x_fit[:edge_n], x_fit[-edge_n:]])
    y_edges = np.concatenate([y_fit[:edge_n], y_fit[-edge_n:]])

    # Fit linear baseline to edges
    bg_coeffs = np.polyfit(x_edges, y_edges, 1)
    bg_slope = bg_coeffs[0]
    bg_offset = bg_coeffs[1]

    # For overlapping peaks, reduce baseline slope to prevent tilting
    # This is key to fixing the P6/P7 problem
    if n_peaks > 1:
        # Calculate peak separation
        peak_separation = max(centers) - min(centers)
        avg_fwhm = np.mean(fwhms)

        # If peaks are very close (< 2 FWHM), reduce baseline slope significantly
        if peak_separation < avg_fwhm * 2:
            slope_reduction = 0.3  # Reduce slope by 70%
            bg_slope *= slope_reduction

    x_ref = x_fit[0]

    # Step 4: Build initial parameters and bounds
    p0 = []
    bounds_lower = []
    bounds_upper = []

    # Sort peaks by height to identify small vs large peaks
    sorted_by_height = sorted(range(n_peaks), key=lambda i: peak_params[i]['height'], reverse=True)
    height_ratio = peak_params[sorted_by_height[-1]]['height'] / peak_params[sorted_by_height[0]]['height'] if n_peaks > 1 else 1.0

    use_voigt = (fit_method == "voigt")

    for i in range(n_peaks):
        params = peak_params[i]

        cen_guess = params['center']
        fwhm_est = params['fwhm']
        height = params['height']

        sig_guess = fwhm_est / 2.355
        gam_guess = fwhm_est / 2

        # Improved amplitude estimation
        # Use actual peak height, not just maximum
        amp_guess = height * sig_guess * np.sqrt(2 * np.pi)

        # Determine if this is a "small" peak relative to others
        is_small_peak = (i in sorted_by_height[n_peaks//2:]) and (height_ratio < 0.5) if n_peaks > 1 else False

        # Set bounds based on peak characteristics
        y_range = np.max(y_fit) - np.min(y_fit)

        # Amplitude bounds
        # Key fix: ensure minimum amplitude based on actual observed height
        amp_min = height * sig_guess * np.sqrt(2 * np.pi) * 0.3  # At least 30% of estimated
        amp_max = y_range * sig_guess * np.sqrt(2 * np.pi) * 5

        # For small peaks, tighten the bounds to prevent them from disappearing
        if is_small_peak:
            amp_min = height * sig_guess * np.sqrt(2 * np.pi) * 0.5  # At least 50% for small peaks
            amp_max = height * sig_guess * np.sqrt(2 * np.pi) * 3   # Don't let it grow too much

        # Center bounds
        # Tighter constraints for small peaks to prevent drift toward large peaks
        if is_small_peak:
            center_tolerance = fwhm_est * 0.3  # Tighter for small peaks
        else:
            center_tolerance = fwhm_est * 0.5

        # Sigma/Gamma bounds
        # Prevent small peaks from getting too wide (which would effectively merge them)
        sig_min = dx * 0.5
        gam_min = dx * 0.5

        if is_small_peak:
            sig_max = fwhm_est * 1.5  # Tighter upper bound for small peaks
            gam_max = fwhm_est * 1.5
        else:
            sig_max = fwhm_est * 3
            gam_max = fwhm_est * 3

        if use_voigt:
            p0.extend([amp_guess, cen_guess, sig_guess, gam_guess])
            bounds_lower.extend([amp_min, cen_guess - center_tolerance, sig_min, gam_min])
            bounds_upper.extend([amp_max, cen_guess + center_tolerance, sig_max, gam_max])
        else:
            p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, 0.5])
            bounds_lower.extend([amp_min, cen_guess - center_tolerance, sig_min, gam_min, 0])
            bounds_upper.extend([amp_max, cen_guess + center_tolerance, sig_max, gam_max, 1.0])

    # Add baseline parameters
    p0.extend([bg_offset, bg_slope])
    bounds_lower.extend([-np.max(np.abs(y_fit)) * 2, -abs(bg_slope) * 3])
    bounds_upper.extend([np.max(y_fit) * 2, abs(bg_slope) * 3])

    # Step 5: Define fitting function
    n_params = 4 if use_voigt else 5

    if use_voigt:
        def multi_peak_func(x, *params):
            y = np.zeros_like(x)
            for i in range(n_peaks):
                offset = i * 4
                amp, cen, sig, gam = params[offset:offset+4]
                y += voigt(x, amp, cen, sig, gam)
            bg_off, bg_slp = params[-2], params[-1]
            y += bg_off + bg_slp * (x - x_ref)
            return y
    else:
        def multi_peak_func(x, *params):
            y = np.zeros_like(x)
            for i in range(n_peaks):
                offset = i * 5
                amp, cen, sig, gam, eta = params[offset:offset+5]
                y += pseudo_voigt(x, amp, cen, sig, gam, eta)
            bg_off, bg_slp = params[-2], params[-1]
            y += bg_off + bg_slp * (x - x_ref)
            return y

    # Step 6: Perform fitting with multiple attempts
    try:
        # First attempt with trf method
        popt, pcov = curve_fit(
            multi_peak_func, x_fit, y_fit,
            p0=p0, bounds=(bounds_lower, bounds_upper),
            method='trf', maxfev=30000,
            ftol=1e-9, xtol=1e-9
        )
    except Exception:
        try:
            # Fallback to dogbox
            popt, pcov = curve_fit(
                multi_peak_func, x_fit, y_fit,
                p0=p0, bounds=(bounds_lower, bounds_upper),
                method='dogbox', maxfev=50000
            )
        except Exception as e:
            print(f"      Fitting failed: {e}")
            return None, None

    return popt, {
        'x_fit': x_fit,
        'y_fit': y_fit,
        'n_peaks': n_peaks,
        'n_params': n_params,
        'x_ref': x_ref,
        'use_voigt': use_voigt
    }

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

    # Find all peaks
    all_peaks, properties = find_peaks(y, distance=30, prominence=np.std(y)*0.5)

    # Filter peaks
    filtered_peaks = []
    for i in all_peaks:
        if 90 <= i < len(y) - 90:
            left = y[i - 90]
            right = y[i + 90]
            min_neighbor = min(left, right)
            if y[i] >= min_neighbor * 1.15:
                filtered_peaks.append(i)
        elif 1 <= i < len(y) - 1:
            # For edge peaks, use smaller window
            window = min(i, len(y) - i - 1, 30)
            if window > 5:
                left = y[i - window]
                right = y[i + window]
                min_neighbor = min(left, right)
                if y[i] >= min_neighbor * 1.15:
                    filtered_peaks.append(i)

    peaks = filtered_peaks
    print(f"   ➤ Detected {len(peaks)} peaks (with +15% neighbor filter)")
    if len(peaks) == 0:
        print("   ⚠️ No valid peaks found, skipping this file.")
        return

    # Group overlapping peaks
    dx = np.mean(np.diff(x))
    peak_groups = []
    current_group = [peaks[0]]

    for i in range(1, len(peaks)):
        prev_peak = peaks[i-1]
        curr_peak = peaks[i]
        distance = x[curr_peak] - x[prev_peak]

        # Estimate FWHM for grouping decision
        window = 30
        left = max(0, curr_peak - window)
        right = min(len(x), curr_peak + window)
        x_local = x[left:right]
        y_local = y[left:right]
        fwhm, _ = estimate_fwhm_robust(x_local, y_local, curr_peak - left)

        # Group if distance < 2.5 * FWHM
        if distance < fwhm * 2.5:
            current_group.append(curr_peak)
        else:
            peak_groups.append(current_group)
            current_group = [curr_peak]

    peak_groups.append(current_group)

    # Report groups
    for g_idx, group in enumerate(peak_groups):
        if len(group) > 1:
            centers = [f"{x[p]:.2f}" for p in group]
            print(f"   ➤ Group {g_idx+1}: {len(group)} overlapping peaks at 2θ = {', '.join(centers)}")

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
            # Single peak - use original method
            peak = group[0]
            window = 55
            left = max(0, peak - window)
            right = min(len(x), peak + window)
            x_local = x[left:right]
            y_local = y[left:right]

            # Background subtraction
            bg_left = np.mean(y_local[:3])
            bg_right = np.mean(y_local[-3:])
            slope = (bg_right - bg_left) / (x_local[-1] - x_local[0])
            background = bg_left + slope * (x_local - x_local[0])
            y_fit_input = y_local - background

            x_smooth = np.linspace(x_local.min(), x_local.max(), 1000)

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
                        "Eta": "N/A",
                        "FWHM": 2.355 * popt[2],
                        "Area": popt[0]
                    })
                else:
                    popt, _ = fit_pseudo_voigt(x_local, y_fit_input)
                    y_fit_corrected = pseudo_voigt(x_smooth, *popt) + (bg_left + slope * (x_smooth - x_local[0]))
                    fit_label = "Pseudo-Voigt Fit"
                    peak_counter += 1
                    fwhm = calculate_fwhm(popt[2], popt[3], popt[4])
                    area = calculate_area(popt[0], popt[2], popt[3], popt[4])
                    results.append({
                        "Peak #": peak_counter,
                        "Center": popt[1],
                        "Amplitude": popt[0],
                        "Sigma": popt[2],
                        "Gamma": popt[3],
                        "Eta": popt[4],
                        "FWHM": fwhm,
                        "Area": area
                    })

                bg_line = bg_left + slope * (x_smooth - x_local[0])

                ax = axs[g_idx]
                ax.plot(x_local, y_local, 'k-', linewidth=1, label="Raw Data")
                ax.plot(x_smooth, y_fit_corrected, color='#BA55D3', linestyle='--', linewidth=2, label=fit_label)
                ax.plot(x_smooth, bg_line, color='#FF69B4', linestyle='-', linewidth=1, label="Background", alpha=0.7)
                ax.set_title(f"Peak {peak_counter} @ {popt[1]:.3f}")
                ax.set_xlabel("2θ (degree)")
                ax.set_ylabel("Intensity")
                ax.legend(fontsize=8)
                ax.set_facecolor("white")
                ax.grid(True, alpha=0.3)

            except Exception as e:
                print(f"   ⚠️ Peak at {x[peak]:.2f} fit failed: {e}")
                axs[g_idx].text(0.3, 0.5, "Fit failed", transform=axs[g_idx].transAxes, fontsize=12, color='red')

        else:
            # Multiple overlapping peaks - use improved multi-peak fitting
            print(f"   ➤ Fitting {len(group)} overlapping peaks together...")

            # Convert global indices to fit
            popt, fit_info = fit_overlapping_peaks(x, y, group, fit_method)

            if popt is None:
                axs[g_idx].text(0.3, 0.5, "Multi-peak fit failed", transform=axs[g_idx].transAxes, fontsize=12, color='red')
                continue

            x_fit = fit_info['x_fit']
            y_fit = fit_info['y_fit']
            n_params = fit_info['n_params']
            x_ref = fit_info['x_ref']
            use_voigt = fit_info['use_voigt']

            x_smooth = np.linspace(x_fit.min(), x_fit.max(), 1000)

            # Calculate total fit
            y_total = np.zeros_like(x_smooth)
            bg_off, bg_slp = popt[-2], popt[-1]
            y_total += bg_off + bg_slp * (x_smooth - x_ref)

            colors = plt.cm.Set1(np.linspace(0, 1, len(group)))

            ax = axs[g_idx]
            ax.plot(x_fit, y_fit, 'k-', linewidth=1, label="Raw Data")

            for i in range(len(group)):
                offset = i * n_params

                if use_voigt:
                    amp, cen, sig, gam = popt[offset:offset+4]
                    y_peak = voigt(x_smooth, amp, cen, sig, gam)
                    fwhm = 2.355 * sig
                    area = amp
                    eta = "N/A"
                else:
                    amp, cen, sig, gam, eta = popt[offset:offset+5]
                    y_peak = pseudo_voigt(x_smooth, amp, cen, sig, gam, eta)
                    fwhm = calculate_fwhm(sig, gam, eta)
                    area = calculate_area(amp, sig, gam, eta)

                y_total += y_peak

                # Plot individual peak with baseline
                y_peak_with_bg = y_peak + bg_off + bg_slp * (x_smooth - x_ref)
                ax.plot(x_smooth, y_peak_with_bg, '--', color=colors[i], linewidth=1.5,
                       label=f"P{peak_counter+i+1}", alpha=0.8)

                results.append({
                    "Peak #": peak_counter + i + 1,
                    "Center": cen,
                    "Amplitude": amp,
                    "Sigma": sig,
                    "Gamma": gam,
                    "Eta": eta if isinstance(eta, str) else f"{eta:.4f}",
                    "FWHM": fwhm,
                    "Area": area
                })

            peak_counter += len(group)

            # Plot total fit
            ax.plot(x_smooth, y_total, 'r-', linewidth=2, label="Total Fit")

            # Plot baseline
            bg_line = bg_off + bg_slp * (x_smooth - x_ref)
            ax.plot(x_smooth, bg_line, color='#FF69B4', linestyle='-', linewidth=1,
                   label="Background", alpha=0.7)

            # Title with all peak centers
            centers_str = ", ".join([f"{x[p]:.2f}" for p in group])
            ax.set_title(f"Peaks @ {centers_str}")
            ax.set_xlabel("2θ (degree)")
            ax.set_ylabel("Intensity")
            ax.legend(fontsize=7, loc='best')
            ax.set_facecolor("white")
            ax.grid(True, alpha=0.3)

    # Remove unused subplots
    for j in range(len(peak_groups), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"{filename} - {fit_method.capitalize()} Fit (Improved)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig_path = os.path.join(save_dir, f"{filename}_fit.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    df = pd.DataFrame(results)
    df["File"] = filename
    csv_path = os.path.join(save_dir, f"{filename}_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"   ✅ Saved {len(results)} peaks to {csv_path}")

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
            all_dfs.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv_path = os.path.join(save_dir, "all_results.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n📦 Total results saved to: {combined_csv_path}")

if __name__ == "__main__":
    main()
