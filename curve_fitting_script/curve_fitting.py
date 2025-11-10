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

    results = []
    subplot_cols = 3
    subplot_rows = int(np.ceil(len(peaks) / subplot_cols))
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(5*subplot_cols, 4*subplot_rows))
    axs = axs.flatten()

    for idx, peak in enumerate(peaks):
        window = 55
        left = max(0, peak - window)
        right = min(len(x), peak + window)
        x_local = x[left:right]
        y_local = y[left:right]
        
        # ----------- Background subtraction: linear interpolation ----------------
        # Use the average values of the edges to construct the background line
        # Estimate and subtract background (used only for fitting)
        bg_left = np.mean(y_local[:1])
        bg_right = np.mean(y_local[-1:])
        slope = (bg_right - bg_left) / (x_local[-1] - x_local[0])
        background = bg_left + slope * (x_local - x_local[0])
        y_fit_input = y_local - background  # for fitting only

        x_smooth = np.linspace(x_local.min(), x_local.max(), 5000)

        try:
            if fit_method == "voigt":
                popt, _ = fit_voigt(x_local, y_fit_input)
                y_fit_corrected = pseudo_voigt(x_smooth, *popt) + (bg_left + slope * (x_smooth - x_local[0]))
                fit_label = "Voigt Fit"
                results.append({
                    "Peak #": idx+1,
                    "Center": popt[1],
                    "Amplitude": popt[0],
                    "Sigma": popt[2],
                    "Gamma": popt[3],
                    "Eta": "N/A"
                })
            else:
                popt, _ = fit_pseudo_voigt(x_local, y_fit_input)
                y_fit_corrected = pseudo_voigt(x_smooth, *popt) + (bg_left + slope * (x_smooth - x_local[0]))
                bg_line = bg_left + slope * (x_smooth - x_local[0])  # background line
                fit_label = "Pseudo-Voigt Fit"
                results.append({
                    "Peak #": idx+1,
                    "Center": popt[1],
                    "Amplitude": popt[0],
                    "Sigma": popt[2],
                    "Gamma": popt[3],
                    "Eta": popt[4]
                })

            ax = axs[idx]
            ax.plot(x_local, y_local, color='black', label="Raw Data")
            ax.plot(x_smooth, y_fit_corrected, color='#BA55D3', linestyle='--', linewidth=2, label=fit_label)
            ax.plot(x_smooth, bg_line, color='#FF69B4', linestyle='-', linewidth=1.5, label="Background")
            ax.set_title(f"Peak {idx+1} @ {popt[1]:.3f}")
            ax.set_xlabel("2θ (degree)")
            ax.set_ylabel("Intensity")
            ax.legend()
            ax.set_facecolor("white")
            ax.grid(False)

        except RuntimeError:
            print(f"⚠️ Peak {idx+1} fit failed 💔")
            axs[idx].text(0.3, 0.5, "Fit failed", transform=axs[idx].transAxes, fontsize=12, color='red')

    for j in range(len(peaks), len(axs)):
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

def run_peak_fitting(xy_folder, save_dir=None, method="pseudo", show_plot=False):
    """
    Batch peak fitting for multiple .xy files.

    Args:
        xy_folder (str): Directory containing .xy files
        save_dir (str): Output directory for plots and results (default: xy_folder/fit_output)
        method (str): 'pseudo' or 'voigt'
        show_plot (bool): Whether to show matplotlib plot window (default: False)
    """
    method = method.lower()
    assert method in ["pseudo", "voigt"], "method must be 'pseudo' or 'voigt'"

    if save_dir is None:
        save_dir = os.path.join(xy_folder, "fit_output")
    os.makedirs(save_dir, exist_ok=True)

    files = [f for f in os.listdir(xy_folder) if f.endswith(".xy")]
    files.sort()

    all_dfs = []
    for fname in files:
        fpath = os.path.join(xy_folder, fname)
        df = process_file(fpath, save_dir, method=method)
        if df is not None:
            all_dfs.append(df)
            all_dfs.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))  # blank line

        if show_plot:
            plt.show()

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv_path = os.path.join(save_dir, "all_results.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n📦 Total results saved to: {combined_csv_path}")
    else:
        print("⚠️ No peak results were generated.")


if __name__ == "__main__":
    main()
