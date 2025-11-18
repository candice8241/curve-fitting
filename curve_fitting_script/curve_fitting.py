# -*- coding: utf-8 -*-
"""
Manual Peak Fitting for XRD Data
Click on peaks to fit them and get peak position, FWHM, and area
@author: candicewang928@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import os
import pandas as pd
from scipy.special import wofz

# ---------- Peak Functions ----------
def voigt(x, amplitude, center, sigma, gamma):
    """Voigt profile"""
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    """Pseudo-Voigt profile: linear combination of Gaussian and Lorentzian"""
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian

def pseudo_voigt_with_bg(x, amplitude, center, sigma, gamma, eta, bg0, bg1):
    """Pseudo-Voigt with linear background"""
    bg = bg0 + bg1 * x
    return pseudo_voigt(x, amplitude, center, sigma, gamma, eta) + bg

# ---------- FWHM Calculation ----------
def calculate_fwhm(sigma, gamma, eta):
    """
    Calculate FWHM for Pseudo-Voigt profile
    Using the approximation formula
    """
    fwhm_g = 2.355 * sigma  # Gaussian FWHM
    fwhm_l = 2 * gamma      # Lorentzian FWHM
    # Thompson et al. approximation for Pseudo-Voigt FWHM
    fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
    return fwhm

# ---------- Fit single peak ----------
def fit_single_peak(x_local, y_local):
    """
    Fit a single peak with linear background
    Returns: fitted parameters, success flag
    """
    # Initial guesses
    bg0 = np.mean([y_local[0], y_local[-1]])
    bg1 = (y_local[-1] - y_local[0]) / (x_local[-1] - x_local[0])

    y_no_bg = y_local - (bg0 + bg1 * x_local)
    amplitude_guess = np.max(y_no_bg)
    center_guess = x_local[np.argmax(y_no_bg)]
    sigma_guess = 0.05
    gamma_guess = 0.05
    eta_guess = 0.5

    p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess, eta_guess, bg0, bg1]

    bounds_lower = [0, x_local.min(), 0.001, 0.001, 0, 0, -np.inf]
    bounds_upper = [np.inf, x_local.max(), 1.0, 1.0, 1.0, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(pseudo_voigt_with_bg, x_local, y_local,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               maxfev=100000)
        return popt, True
    except Exception as e:
        print(f"   Fit failed: {e}")
        return None, False

# ---------- Interactive Peak Selection ----------
class PeakSelector:
    def __init__(self, x, y, filename, save_dir):
        self.x = x
        self.y = y
        self.filename = filename
        self.save_dir = save_dir
        self.selected_positions = []
        self.results = []
        self.peak_count = 0

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.ax.plot(x, y, 'b-', linewidth=0.8, label='Data')
        self.ax.set_xlabel('2θ (degree)', fontsize=12)
        self.ax.set_ylabel('Intensity', fontsize=12)
        self.ax.set_title(f'{filename}\nLeft-click to select peaks, Right-click to finish', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        # Connect events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        print("\n" + "="*60)
        print("Manual Peak Selection Mode")
        print("="*60)
        print("Left-click: Select a peak position")
        print("Right-click or press 'Enter': Finish selection and fit all peaks")
        print("Press 'q': Quit without saving")
        print("="*60 + "\n")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click - select peak
            x_click = event.xdata
            self.selected_positions.append(x_click)

            # Mark the selected position
            self.ax.axvline(x=x_click, color='r', linestyle='--', alpha=0.5)
            self.ax.plot(x_click, event.ydata, 'ro', markersize=8)
            self.fig.canvas.draw()

            print(f"   Peak {len(self.selected_positions)} selected at 2θ = {x_click:.4f}")

        elif event.button == 3:  # Right click - finish
            self.finish_selection()

    def on_key(self, event):
        if event.key == 'enter':
            self.finish_selection()
        elif event.key == 'q':
            print("\nQuitting without saving...")
            plt.close(self.fig)

    def finish_selection(self):
        if len(self.selected_positions) == 0:
            print("\nNo peaks selected!")
            plt.close(self.fig)
            return

        print(f"\n   Fitting {len(self.selected_positions)} selected peaks...")

        # Fit each selected peak
        for i, pos in enumerate(self.selected_positions):
            self.fit_peak_at_position(pos, i+1)

        # Save results
        self.save_results()

        # Close the selection window
        plt.close(self.fig)

        # Show fitted results
        self.show_fit_results()

    def fit_peak_at_position(self, x_pos, peak_num):
        """Fit a peak at the clicked position"""
        # Find the index closest to clicked position
        idx = np.argmin(np.abs(self.x - x_pos))

        # Extract local region (window around the peak)
        window = 50  # Points on each side
        left = max(0, idx - window)
        right = min(len(self.x), idx + window)

        x_local = self.x[left:right]
        y_local = self.y[left:right]

        # Fit the peak
        popt, success = fit_single_peak(x_local, y_local)

        if success:
            amplitude, center, sigma, gamma, eta, bg0, bg1 = popt

            # Calculate FWHM
            fwhm = calculate_fwhm(sigma, gamma, eta)

            # Calculate area (integrate peak without background)
            x_fine = np.linspace(x_local.min(), x_local.max(), 1000)
            y_peak = pseudo_voigt(x_fine, amplitude, center, sigma, gamma, eta)
            area = trapezoid(y_peak, x_fine)

            # Store results
            self.results.append({
                'Peak #': peak_num,
                'Center (2θ)': center,
                'FWHM': fwhm,
                'Area': area,
                'Amplitude': amplitude,
                'Sigma': sigma,
                'Gamma': gamma,
                'Eta': eta,
                'x_local': x_local,
                'y_local': y_local,
                'popt': popt
            })

            print(f"   Peak {peak_num}: Center={center:.4f}, FWHM={fwhm:.4f}, Area={area:.2f}")
        else:
            print(f"   Peak {peak_num}: Fitting failed!")

    def save_results(self):
        """Save fitting results to CSV"""
        if not self.results:
            return

        # Create DataFrame with main results
        df_data = []
        for r in self.results:
            df_data.append({
                'Peak #': r['Peak #'],
                'Center (2θ)': r['Center (2θ)'],
                'FWHM': r['FWHM'],
                'Area': r['Area'],
                'Amplitude': r['Amplitude'],
                'Sigma': r['Sigma'],
                'Gamma': r['Gamma'],
                'Eta': r['Eta']
            })

        df = pd.DataFrame(df_data)
        csv_path = os.path.join(self.save_dir, f"{self.filename}_manual_fit.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n   Results saved to: {csv_path}")

    def show_fit_results(self):
        """Display fitted peaks"""
        n_peaks = len(self.results)
        if n_peaks == 0:
            return

        # Create subplot layout
        cols = min(3, n_peaks)
        rows = int(np.ceil(n_peaks / cols))

        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_peaks == 1:
            axs = [axs]
        else:
            axs = axs.flatten() if n_peaks > 1 else [axs]

        for i, result in enumerate(self.results):
            ax = axs[i]
            x_local = result['x_local']
            y_local = result['y_local']
            popt = result['popt']

            # Plot data
            ax.plot(x_local, y_local, 'ko', markersize=3, label='Data')

            # Plot fit
            x_smooth = np.linspace(x_local.min(), x_local.max(), 500)
            y_fit = pseudo_voigt_with_bg(x_smooth, *popt)
            ax.plot(x_smooth, y_fit, 'r-', linewidth=2, label='Fit')

            # Plot background
            bg_line = popt[5] + popt[6] * x_smooth
            ax.plot(x_smooth, bg_line, 'g--', linewidth=1, label='Background')

            # Labels
            ax.set_xlabel('2θ (degree)')
            ax.set_ylabel('Intensity')
            ax.set_title(f"Peak {result['Peak #']}\n2θ={result['Center (2θ)']:.4f}, FWHM={result['FWHM']:.4f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for j in range(n_peaks, len(axs)):
            fig.delaxes(axs[j])

        plt.suptitle(f"{self.filename} - Manual Peak Fitting Results", fontsize=14)
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.save_dir, f"{self.filename}_manual_fit.png")
        plt.savefig(fig_path, dpi=150)
        print(f"   Figure saved to: {fig_path}")

        plt.show()

# ---------- Main function for single file ----------
def run_peak_fitting(file_path, save_dir=None):
    """
    Main function to run manual peak fitting on a single file

    Parameters:
    -----------
    file_path : str
        Path to the data file (.xy or .dat)
    save_dir : str, optional
        Directory to save results. If None, creates 'fit_output' in the same folder
    """
    # Read data
    try:
        with open(file_path, encoding='latin1') as f:
            data = np.genfromtxt(f, comments="#")
        x = data[:, 0]
        y = data[:, 1]
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Set save directory
    if save_dir is None:
        folder = os.path.dirname(file_path)
        save_dir = os.path.join(folder, "fit_output")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLoading: {filename}")
    print(f"Data range: 2θ = {x.min():.2f} to {x.max():.2f}")

    # Start interactive selection
    selector = PeakSelector(x, y, filename, save_dir)
    plt.show()

# ---------- Main ----------
def main():
    """
    Example usage - modify the file path as needed
    """
    # Option 1: Single file
    file_path = r"D:\HEPS\ID31\dioptas_data\test\sample.xy"

    # Option 2: Or use command line argument
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("\nUsage: python curve_fitting.py <data_file>")
        print("Example: python curve_fitting.py sample.xy")
        return

    run_peak_fitting(file_path)

if __name__ == "__main__":
    main()
