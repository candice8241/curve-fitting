# -*- coding: utf-8 -*-
"""
Batch Peak Fitting Module
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz


class BatchFitter:
    """Batch peak fitting for XRD data"""

    def __init__(self, folder, fit_method='pseudo'):
        """
        Initialize batch fitter

        Parameters:
        -----------
        folder : str
            Folder containing .xy files
        fit_method : str
            Fitting method ('pseudo' or 'voigt')
        """
        self.folder = folder
        self.fit_method = fit_method

    @staticmethod
    def voigt(x, amplitude, center, sigma, gamma):
        """Voigt profile function"""
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
        """Pseudo-Voigt profile function"""
        gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
        return eta * lorentzian + (1 - eta) * gaussian

    def run_batch_fitting(self):
        """Run batch fitting on all .xy files in folder"""
        files = glob.glob(os.path.join(self.folder, "*.xy"))

        if not files:
            raise ValueError(f"No .xy files found in {self.folder}")

        print(f"Found {len(files)} files to fit")
        all_results = []

        for i, file_path in enumerate(files, 1):
            filename = os.path.basename(file_path)
            print(f"Fitting ({i}/{len(files)}): {filename}")

            try:
                # Read data
                data = np.loadtxt(file_path, comments='#')
                x = data[:, 0]
                y = data[:, 1]

                # Find peaks
                peaks, _ = find_peaks(y, distance=30)

                if len(peaks) == 0:
                    print(f"  No peaks found in {filename}")
                    continue

                # Fit each peak
                for peak_idx, peak in enumerate(peaks):
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
                    y_fit = y_local - background

                    try:
                        # Initial guess
                        amplitude = np.max(y_fit)
                        center = x_local[np.argmax(y_fit)]
                        sigma = np.std(x_local) / 5

                        if self.fit_method == 'voigt':
                            p0 = [amplitude, center, sigma, sigma]
                            bounds = ([0, x_local.min(), 0, 0],
                                      [np.inf, x_local.max(), np.inf, np.inf])
                            popt, _ = curve_fit(self.voigt, x_local, y_fit,
                                                p0=p0, bounds=bounds, maxfev=100000)
                            result = {
                                'File': filename.replace('.xy', ''),
                                'Peak': peak_idx + 1,
                                'Center': popt[1],
                                'Amplitude': popt[0],
                                'Sigma': popt[2],
                                'Gamma': popt[3]
                            }
                        else:  # pseudo-voigt
                            p0 = [amplitude, center, sigma, sigma, 0.5]
                            bounds = ([0, x_local.min(), 0, 0, 0],
                                      [np.inf, x_local.max(), np.inf, np.inf, 1.0])
                            popt, _ = curve_fit(self.pseudo_voigt, x_local, y_fit,
                                                p0=p0, bounds=bounds, maxfev=100000)
                            result = {
                                'File': filename.replace('.xy', ''),
                                'Peak': peak_idx + 1,
                                'Center': popt[1],
                                'Amplitude': popt[0],
                                'Sigma': popt[2],
                                'Gamma': popt[3],
                                'Eta': popt[4]
                            }

                        all_results.append(result)

                    except RuntimeError:
                        print(f"  Peak {peak_idx + 1} fit failed")

            except Exception as e:
                print(f"  Error processing {filename}: {e}")

        # Save all results
        if all_results:
            df = pd.DataFrame(all_results)
            output_csv = os.path.join(self.folder, 'all_peaks_fitted.csv')
            df.to_csv(output_csv, index=False)
            print(f"✓ All results saved to: {output_csv}")
        else:
            print("⚠️ No successful fits")
