# -*- coding: utf-8 -*-
"""
Manual Peak Fitting for XRD Data
Click on peaks to fit them and get peak position, FWHM, and area
@author: candicewang928@gmail.com
"""

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
        self.results = []
        self.peak_count = 0
        self.fit_lines = []  # Store fit plot objects
        self.picking_enabled = True

    def run(self):
        """Run interactive peak selection with zoom/pan support"""
        # Create figure with space for buttons
        self.fig = plt.figure(figsize=(14, 8))

        # Main plot area
        self.ax = self.fig.add_axes([0.1, 0.15, 0.85, 0.75])
        self.ax.plot(self.x, self.y, 'b-', linewidth=0.8, label='Data')
        self.ax.set_xlabel('2θ (degree)', fontsize=12)
        self.ax.set_ylabel('Intensity', fontsize=12)
        self.ax.set_title(f'{self.filename}\nClick on peaks to fit (use toolbar to zoom/pan)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')

        # Add buttons
        ax_finish = self.fig.add_axes([0.7, 0.02, 0.1, 0.04])
        ax_clear = self.fig.add_axes([0.55, 0.02, 0.1, 0.04])
        ax_undo = self.fig.add_axes([0.4, 0.02, 0.1, 0.04])

        self.btn_finish = Button(ax_finish, 'Save & Exit')
        self.btn_finish.on_clicked(self.on_finish)

        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self.on_clear)

        self.btn_undo = Button(ax_undo, 'Undo Last')
        self.btn_undo.on_clicked(self.on_undo)

        # Connect mouse click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        print("\n" + "="*60)
        print("Interactive Peak Fitting Mode")
        print("="*60)
        print("- Use toolbar to ZOOM and PAN")
        print("- LEFT-CLICK on a peak to fit it immediately")
        print("- Click 'Undo Last' to remove the last fit")
        print("- Click 'Clear All' to remove all fits")
        print("- Click 'Save & Exit' when done")
        print("="*60 + "\n")

        plt.show(block=True)

    def on_click(self, event):
        """Handle mouse click - fit peak immediately"""
        # Ignore clicks outside the main axes
        if event.inaxes != self.ax:
            return

        # Only respond to left click (button 1)
        if event.button != 1:
            return

        # Check if toolbar is in zoom/pan mode
        toolbar = self.fig.canvas.manager.toolbar
        if toolbar.mode != '':
            return  # Don't pick peaks while zooming/panning

        x_click = event.xdata

        # Fit the peak immediately
        self.fit_and_plot_peak(x_click)

    def fit_and_plot_peak(self, x_pos):
        """Fit a peak at the clicked position and plot result immediately"""
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
            self.peak_count += 1
            amplitude, center, sigma, gamma, eta, bg0, bg1 = popt

            # Calculate FWHM
            fwhm = calculate_fwhm(sigma, gamma, eta)

            # Calculate area (integrate peak without background)
            x_fine = np.linspace(x_local.min(), x_local.max(), 1000)
            y_peak = pseudo_voigt(x_fine, amplitude, center, sigma, gamma, eta)
            area = trapezoid(y_peak, x_fine)

            # Plot fit on main figure
            x_smooth = np.linspace(x_local.min(), x_local.max(), 500)
            y_fit = pseudo_voigt_with_bg(x_smooth, *popt)
            bg_line = bg0 + bg1 * x_smooth

            # Plot and store references for undo
            fit_line, = self.ax.plot(x_smooth, y_fit, 'r-', linewidth=2, alpha=0.8)
            bg_plot, = self.ax.plot(x_smooth, bg_line, 'g--', linewidth=1, alpha=0.6)
            peak_marker, = self.ax.plot(center, pseudo_voigt_with_bg(center, *popt), 'r*', markersize=15)

            # Add text annotation
            text_y = pseudo_voigt_with_bg(center, *popt)
            annotation = self.ax.annotate(
                f'#{self.peak_count}\n2θ={center:.3f}\nFWHM={fwhm:.4f}\nArea={area:.1f}',
                xy=(center, text_y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )

            # Store plot objects for undo
            self.fit_lines.append({
                'lines': [fit_line, bg_plot, peak_marker],
                'annotation': annotation,
                'result': {
                    'Peak #': self.peak_count,
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
                }
            })

            # Update canvas
            self.fig.canvas.draw()

            print(f"   Peak {self.peak_count}: Center={center:.4f}, FWHM={fwhm:.4f}, Area={area:.2f}")
        else:
            print(f"   Fitting failed at 2θ = {x_pos:.4f}")

    def on_undo(self, event):
        """Remove the last fitted peak"""
        if len(self.fit_lines) == 0:
            print("   Nothing to undo")
            return

        # Get last fit
        last_fit = self.fit_lines.pop()

        # Remove plot objects
        for line in last_fit['lines']:
            line.remove()
        last_fit['annotation'].remove()

        self.peak_count -= 1
        self.fig.canvas.draw()
        print(f"   Removed last peak. {len(self.fit_lines)} peaks remaining.")

    def on_clear(self, event):
        """Clear all fitted peaks"""
        for fit_data in self.fit_lines:
            for line in fit_data['lines']:
                line.remove()
            fit_data['annotation'].remove()

        self.fit_lines = []
        self.peak_count = 0
        self.fig.canvas.draw()
        print("   All fits cleared.")

    def on_finish(self, event):
        """Save results and close"""
        if len(self.fit_lines) == 0:
            print("\nNo peaks fitted!")
            plt.close(self.fig)
            return

        # Collect results
        self.results = [fit_data['result'] for fit_data in self.fit_lines]

        # Save results
        self.save_results()

        # Save the current figure
        fig_path = os.path.join(self.save_dir, f"{self.filename}_manual_fit.png")
        self.fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   Figure saved to: {fig_path}")

        plt.close(self.fig)

        # Show individual peak fits
        self.show_individual_fits()

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

    def show_individual_fits(self):
        """Display individual fitted peaks in subplots"""
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

            # Plot peak without background
            y_peak_only = pseudo_voigt(x_smooth, popt[0], popt[1], popt[2], popt[3], popt[4])
            ax.fill_between(x_smooth, bg_line, y_fit, alpha=0.3, color='red', label='Peak Area')

            # Labels
            ax.set_xlabel('2θ (degree)')
            ax.set_ylabel('Intensity')
            ax.set_title(f"Peak {result['Peak #']}\n2θ={result['Center (2θ)']:.4f}, FWHM={result['FWHM']:.4f}, Area={result['Area']:.1f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for j in range(n_peaks, len(axs)):
            fig.delaxes(axs[j])

        plt.suptitle(f"{self.filename} - Individual Peak Fits", fontsize=14)
        plt.tight_layout()
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
    selector.run()

# ---------- File Selection Dialog ----------
def select_file_dialog(initial_dir=None):
    """
    Open a file selection dialog
    Returns the selected file path or None if cancelled
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select XRD Data File",
        filetypes=[
            ("XRD Data Files", "*.xy *.dat *.txt"),
            ("XY Files", "*.xy"),
            ("DAT Files", "*.dat"),
            ("Text Files", "*.txt"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()
    return file_path if file_path else None

# ---------- Main ----------
def main():
    """
    Example usage - modify the file path as needed
    """
    import sys

    # Get file path from command line or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # No argument provided - open file dialog
        print("No file specified. Opening file selection dialog...")
        file_path = select_file_dialog()
        if not file_path:
            print("No file selected. Exiting.")
            return
        run_peak_fitting(file_path)
        return

    # Check if path exists
    if not os.path.exists(file_path):
        print(f"Path not found: {file_path}")
        print("\nUsage: python curve_fitting.py <data_file>")
        print("Example: python curve_fitting.py sample.xy")
        return

    # If it's a directory, open file dialog in that directory
    if os.path.isdir(file_path):
        print(f"Opening file selection dialog in: {file_path}")
        selected_file = select_file_dialog(initial_dir=file_path)
        if not selected_file:
            print("No file selected. Exiting.")
            return
        file_path = selected_file

    run_peak_fitting(file_path)

if __name__ == "__main__":
    main()
