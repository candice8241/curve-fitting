# -*- coding: utf-8 -*-
"""
Batch Peak Fitting Module for XRD Data
Fits multiple peaks in XRD patterns using Voigt or Pseudo-Voigt profiles
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz


class BatchFitter:
    """Batch peak fitter for XRD data"""

    def __init__(self, folder, fit_method='pseudo'):
        """
        Initialize the batch fitter

        Parameters:
        -----------
        folder : str
            Folder containing XY data files
        fit_method : str
            Fitting method: 'voigt' or 'pseudo' (pseudo-Voigt)
        """
        self.folder = folder
        self.fit_method = fit_method
        self.output_dir = os.path.join(folder, "fit_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def voigt(self, x, amplitude, center, sigma, gamma):
        """Voigt profile"""
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    def pseudo_voigt(self, x, amplitude, center, sigma, gamma, eta):
        """Pseudo-Voigt profile"""
        gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
        return eta * lorentzian + (1 - eta) * gaussian

    def fit_voigt(self, xdata, ydata, p0=None):
        """Fit Voigt profile"""
        if p0 is None:
            amplitude_guess = np.max(ydata)
            center_guess = xdata[np.argmax(ydata)]
            sigma_guess = np.std(xdata) / 5
            gamma_guess = sigma_guess
            p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess]

        bounds = ([0, xdata.min(), 0, 0], [np.inf, xdata.max(), np.inf, np.inf])
        popt, pcov = curve_fit(self.voigt, xdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
        return popt, pcov

    def fit_pseudo_voigt(self, xdata, ydata, p0=None):
        """Fit Pseudo-Voigt profile"""
        if p0 is None:
            amplitude_guess = np.max(ydata)
            center_guess = xdata[np.argmax(ydata)]
            sigma_guess = np.std(xdata) / 5
            gamma_guess = sigma_guess
            eta_guess = 0.5
            p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess, eta_guess]

        bounds = ([0, xdata.min(), 0, 0, 0], [np.inf, xdata.max(), np.inf, np.inf, 1.0])
        popt, pcov = curve_fit(self.pseudo_voigt, xdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
        return popt, pcov

    def process_file(self, file_path):
        """Process a single XY file"""
        try:
            # Load data
            data = np.loadtxt(file_path, comments='#')
            x = data[:, 0]
            y = data[:, 1]
        except Exception as e:
            print(f"❌ Failed to read {file_path}: {e}")
            return None

        filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n📄 Processing: {filename}")

        # Find peaks
        all_peaks, _ = find_peaks(y, distance=30)

        # Filter peaks
        filtered_peaks = []
        for i in all_peaks:
            if 1 <= i < len(y) - 100:
                left = y[max(0, i - 90)]
                right = y[min(len(y) - 1, i + 90)]
                min_neighbor = min(left, right)
                if y[i] >= min_neighbor * 1.15:
                    filtered_peaks.append(i)

        peaks = filtered_peaks
        print(f"   ➤ Detected {len(peaks)} peaks")

        if len(peaks) == 0:
            print("   ⚠️ No valid peaks found")
            return None

        # Fit peaks
        results = []
        subplot_cols = 3
        subplot_rows = int(np.ceil(len(peaks) / subplot_cols))
        fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(5*subplot_cols, 4*subplot_rows))

        if len(peaks) == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        for idx, peak in enumerate(peaks):
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

            x_smooth = np.linspace(x_local.min(), x_local.max(), 500)

            try:
                if self.fit_method == 'voigt':
                    popt, _ = self.fit_voigt(x_local, y_fit_input)
                    y_fit_corrected = self.voigt(x_smooth, *popt) + (bg_left + slope * (x_smooth - x_local[0]))
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
                    popt, _ = self.fit_pseudo_voigt(x_local, y_fit_input)
                    y_fit_corrected = self.pseudo_voigt(x_smooth, *popt) + (bg_left + slope * (x_smooth - x_local[0]))
                    bg_line = bg_left + slope * (x_smooth - x_local[0])
                    fit_label = "Pseudo-Voigt Fit"
                    results.append({
                        "Peak #": idx+1,
                        "Center": popt[1],
                        "Amplitude": popt[0],
                        "Sigma": popt[2],
                        "Gamma": popt[3],
                        "Eta": popt[4]
                    })

                # Plot
                ax = axs[idx]
                ax.plot(x_local, y_local, 'k-', label="Data")
                ax.plot(x_smooth, y_fit_corrected, 'r--', linewidth=2, label=fit_label)
                ax.plot(x_smooth, bg_line if self.fit_method == 'pseudo' else background[:len(x_smooth)],
                       'b-', linewidth=1, label="Background")
                ax.set_title(f"Peak {idx+1} @ {popt[1]:.3f}")
                ax.set_xlabel("2θ (degree)")
                ax.set_ylabel("Intensity")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            except RuntimeError:
                print(f"⚠️ Peak {idx+1} fit failed")
                axs[idx].text(0.3, 0.5, "Fit failed", transform=axs[idx].transAxes,
                            fontsize=12, color='red')

        # Remove empty subplots
        for j in range(len(peaks), len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"{filename} - {self.fit_method.capitalize()} Fit", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Save figure
        fig_path = os.path.join(self.output_dir, f"{filename}_fit.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # Save results
        df = pd.DataFrame(results)
        df["File"] = filename
        csv_path = os.path.join(self.output_dir, f"{filename}_results.csv")
        df.to_csv(csv_path, index=False)

        return df

    def run_batch_fitting(self):
        """Run batch fitting on all XY files in folder"""
        files = glob.glob(os.path.join(self.folder, "*.xy"))
        files.sort()

        if not files:
            raise FileNotFoundError(f"No .xy files found in {self.folder}")

        print(f"Found {len(files)} files to process")

        all_dfs = []
        for file_path in files:
            df = self.process_file(file_path)
            if df is not None:
                all_dfs.append(df)
                # Add blank row separator
                all_dfs.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))

        # Save combined results
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_csv = os.path.join(self.output_dir, "all_results.csv")
            combined_df.to_csv(combined_csv, index=False)
            print(f"\n✅ Batch fitting complete! Results saved to {self.output_dir}")
            print(f"📦 Combined results: {combined_csv}")
