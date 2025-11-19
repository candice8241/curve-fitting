# -*- coding: utf-8 -*-
"""
Interactive Peak Fitting with GUI - File Selection & Mouse Zoom
@author: candicewang928@gmail.com
Enhanced with GUI interface and zoom functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Button
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd

# ---------- Peak profile functions ----------
def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    """Pseudo-Voigt: eta*Lorentzian + (1-eta)*Gaussian"""
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma**2 / ((x - center)**2 + gamma**2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian

def multi_pseudo_voigt(x, *params):
    """
    Multi-peak with linear background
    params = [bg0, bg1, amp1, cen1, sig1, gam1, eta1, amp2, cen2, ...]
    """
    bg = params[0] + params[1] * x
    n_peaks = (len(params) - 2) // 5
    y = bg.copy()
    for i in range(n_peaks):
        offset = 2 + i * 5
        amp, cen, sig, gam, eta = params[offset:offset+5]
        y += pseudo_voigt(x, amp, cen, sig, gam, eta)
    return y

def calculate_fwhm(sigma, gamma, eta):
    """Calculate FWHM from Pseudo-Voigt parameters"""
    fwhm_g = 2.355 * sigma  # Gaussian FWHM
    fwhm_l = 2 * gamma      # Lorentzian FWHM
    return eta * fwhm_l + (1 - eta) * fwhm_g

def calculate_area(amplitude, sigma, gamma, eta):
    """Calculate integrated area"""
    area_g = amplitude * sigma * np.sqrt(2 * np.pi)
    area_l = amplitude * np.pi * gamma
    return eta * area_l + (1 - eta) * area_g

# ---------- Main GUI Application ----------
class PeakFittingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive XRD Peak Fitting Tool")
        self.master.geometry("1400x850")
        self.master.configure(bg='#F0E6FA')

        # Data storage
        self.x = None
        self.y = None
        self.filename = None
        self.filepath = None
        self.selected_peaks = []
        self.peak_markers = []
        self.peak_texts = []
        self.fitted = False
        self.fit_results = None
        self.fit_lines = []
        self.result_annotations = []  # Store result text annotations

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        """Create all GUI components"""
        # Top control panel
        control_frame = tk.Frame(self.master, bg='#BA55D3', height=60)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        control_frame.pack_propagate(False)

        # Buttons with beautiful styling
        btn_style = {
            'font': ('Arial', 11, 'bold'),
            'width': 15,
            'height': 2,
            'relief': tk.RAISED,
            'bd': 3
        }

        self.btn_load = tk.Button(control_frame, text="Load File",
                                   bg='#9370DB', fg='white',
                                   command=self.load_file, **btn_style)
        self.btn_load.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_fit = tk.Button(control_frame, text="Fit Peaks",
                                 bg='#BA55D3', fg='white',
                                 command=self.fit_peaks, state=tk.DISABLED, **btn_style)
        self.btn_fit.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_reset = tk.Button(control_frame, text="Reset",
                                    bg='#FF69B4', fg='white',
                                    command=self.reset_peaks, state=tk.DISABLED, **btn_style)
        self.btn_reset.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_save = tk.Button(control_frame, text="Save Results",
                                  bg='#32CD32', fg='white',
                                  command=self.save_results, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_clear_fit = tk.Button(control_frame, text="Clear Fit",
                                       bg='#FF8C00', fg='white',
                                       command=self.clear_fit, state=tk.DISABLED, **btn_style)
        self.btn_clear_fit.pack(side=tk.LEFT, padx=10, pady=8)

        # Status label
        self.status_label = tk.Label(control_frame, text="Please load a file to start",
                                     bg='#BA55D3', fg='white',
                                     font=('Arial', 11, 'bold'))
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # Main plot area
        plot_frame = tk.Frame(self.master, bg='white')
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure with adjusted size
        self.fig, self.ax = plt.subplots(figsize=(13, 6.5), facecolor='white')
        self.fig.tight_layout(pad=3.0)  # Add padding to prevent text cutoff

        self.ax.set_facecolor('#FAF0FF')
        self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
        self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
        self.ax.set_title('Click on peaks to select | Use toolbar or scroll wheel to zoom',
                         fontsize=13, fontweight='bold', color='#9370DB', pad=15)

        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add matplotlib navigation toolbar
        toolbar_frame = tk.Frame(plot_frame, bg='#E6D5F5')
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.config(bg='#E6D5F5')

        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)  # Add scroll event

        # Info panel at bottom
        info_frame = tk.Frame(self.master, bg='#F0E6FA', height=80)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        info_frame.pack_propagate(False)

        self.info_text = tk.Text(info_frame, height=4, bg='#FAF0FF',
                                 fg='#4B0082', font=('Courier', 10),
                                 relief=tk.SUNKEN, bd=2)
        self.info_text.pack(fill=tk.BOTH, padx=10, pady=5)
        self.info_text.insert('1.0', 'Welcome! Load your XRD data file to begin peak fitting.\n')
        self.info_text.insert('2.0', 'Use toolbar buttons or scroll wheel to zoom the plot.\n')
        self.info_text.insert('3.0', 'Click on peaks in the plot to select them for fitting.\n')
        self.info_text.config(state=tk.DISABLED)

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes != self.ax:
            return

        # Get current axis limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        # Zoom factor
        if event.button == 'up':
            scale_factor = 0.9  # Zoom in
        elif event.button == 'down':
            scale_factor = 1.1  # Zoom out
        else:
            return

        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        self.canvas.draw()

    def load_file(self):
        """Load XRD data file"""
        filepath = filedialog.askopenfilename(
            title="Select XRD Data File",
            filetypes=[("XY files", "*.xy"), ("DAT files", "*.dat"),
                      ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            # Try reading the file
            with open(filepath, encoding='latin1') as f:
                data = np.genfromtxt(f, comments="#")

            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Data must have at least 2 columns (2theta, Intensity)")

            self.x = data[:, 0]
            self.y = data[:, 1]
            self.filepath = filepath
            self.filename = os.path.splitext(os.path.basename(filepath))[0]

            # Reset state
            self.reset_peaks()
            self.fitted = False

            # Plot data with smaller markers
            self.ax.clear()
            self.ax.plot(self.x, self.y, 'o', color='#4B0082', markersize=2.5,
                        label='Data', markeredgewidth=0.3, markeredgecolor='#BA55D3', alpha=0.8)
            self.ax.set_facecolor('#FAF0FF')
            self.ax.grid(True, alpha=0.3, linestyle='--', color='#BA55D3')
            self.ax.set_xlabel('2theta (degree)', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_ylabel('Intensity', fontsize=13, fontweight='bold', color='#BA55D3')
            self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar or scroll wheel to zoom',
                            fontsize=13, fontweight='bold', color='#9370DB', pad=15)
            self.ax.legend(fontsize=11, loc='best', framealpha=0.9)

            # Adjust layout to prevent cutoff
            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()

            # Enable buttons
            self.btn_fit.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL)

            # Update status
            self.status_label.config(text=f"Loaded: {self.filename}")
            self.update_info(f"File loaded: {self.filename}\n"
                           f"Data points: {len(self.x)}\n"
                           f"Click on peaks to select them for fitting\n")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.update_info(f"Error loading file: {str(e)}\n")

    def on_click(self, event):
        """Handle mouse clicks to select peaks"""
        if event.inaxes != self.ax or self.x is None or self.fitted:
            return

        # Check if we're in zoom/pan mode
        if self.toolbar.mode != '':
            return  # Don't select peaks when zooming/panning

        # Find nearest data point
        x_click = event.xdata
        idx = np.argmin(np.abs(self.x - x_click))
        peak_x = self.x[idx]
        peak_y = self.y[idx]

        # Mark the peak
        marker, = self.ax.plot(peak_x, peak_y, '*', color='#FF1493',
                              markersize=18, markeredgecolor='#FFD700',
                              markeredgewidth=2, zorder=10)
        text = self.ax.text(peak_x, peak_y * 1.05, f'P{len(self.selected_peaks)+1}',
                           ha='center', fontsize=10, color='#FF1493',
                           fontweight='bold', zorder=11,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE4E1',
                                   edgecolor='#FF69B4', linewidth=2))

        self.selected_peaks.append(idx)
        self.peak_markers.append(marker)
        self.peak_texts.append(text)
        self.canvas.draw()

        self.update_info(f"Peak {len(self.selected_peaks)} selected at 2theta = {peak_x:.4f} deg\n")
        self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    def estimate_background(self):
        """Estimate background using edge regions away from peaks"""
        # Get data range
        x_range = self.x.max() - self.x.min()

        # Define edge regions (first and last 5% of data)
        edge_width = int(len(self.x) * 0.05)
        if edge_width < 3:
            edge_width = 3

        # Get edge data points
        left_x = self.x[:edge_width]
        left_y = self.y[:edge_width]
        right_x = self.x[-edge_width:]
        right_y = self.y[-edge_width:]

        # Calculate background at edges
        bg_left = np.median(left_y)
        bg_right = np.median(right_y)

        # Linear background parameters
        bg1 = (bg_right - bg_left) / (right_x.mean() - left_x.mean())
        bg0 = bg_left - bg1 * left_x.mean()

        return bg0, bg1

    def fit_peaks(self):
        """Fit all selected peaks"""
        if len(self.selected_peaks) == 0:
            messagebox.showwarning("No Peaks", "Please select at least one peak first!")
            return

        self.update_info(f"Fitting {len(self.selected_peaks)} peaks...\n")

        try:
            # Optimized background estimation
            bg0, bg1 = self.estimate_background()

            # Initial parameters
            p0 = [bg0, bg1]
            bounds_lower = [-np.inf, -np.inf]
            bounds_upper = [np.inf, np.inf]

            for idx in self.selected_peaks:
                # Estimate local background at peak position
                local_bg = bg0 + bg1 * self.x[idx]
                amp_guess = self.y[idx] - local_bg
                if amp_guess < 0:
                    amp_guess = self.y[idx] * 0.5

                cen_guess = self.x[idx]
                sig_guess = 0.05
                gam_guess = 0.05
                eta_guess = 0.5

                p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, eta_guess])
                bounds_lower.extend([0, self.x.min(), 0.001, 0.001, 0])
                bounds_upper.extend([np.inf, self.x.max(), 1.0, 1.0, 1.0])

            # Perform fitting
            popt, pcov = curve_fit(multi_pseudo_voigt, self.x, self.y,
                                  p0=p0, bounds=(bounds_lower, bounds_upper),
                                  maxfev=100000)

            # Plot fit with thinner lines
            x_smooth = np.linspace(self.x.min(), self.x.max(), 2000)
            y_fit = multi_pseudo_voigt(x_smooth, *popt)
            bg_line = popt[0] + popt[1] * x_smooth

            # Thinner lines: linewidth reduced
            line1, = self.ax.plot(x_smooth, y_fit, color='#BA55D3', linewidth=1.5,
                                label='Total Fit', zorder=5)
            line2, = self.ax.plot(x_smooth, bg_line, '--', color='#FF69B4',
                                linewidth=1, label='Background', zorder=4)

            self.fit_lines.extend([line1, line2])

            # Extract and plot individual peaks
            n_peaks = len(self.selected_peaks)
            results = []

            info_msg = "Fitting Results:\n" + "="*50 + "\n"

            # Colors for individual peaks
            peak_colors = ['#1E90FF', '#32CD32', '#FF6347', '#FFD700', '#8A2BE2',
                          '#00CED1', '#FF69B4', '#20B2AA', '#FFA500', '#9370DB']

            for i in range(n_peaks):
                offset = 2 + i * 5
                amp, cen, sig, gam, eta = popt[offset:offset+5]

                # Plot individual peak (thinner line)
                y_single = pseudo_voigt(x_smooth, amp, cen, sig, gam, eta)
                color = peak_colors[i % len(peak_colors)]
                line, = self.ax.plot(x_smooth, y_single + popt[0] + popt[1] * x_smooth,
                                    '-', color=color, linewidth=1, alpha=0.8,
                                    label=f'Peak {i+1}', zorder=3)
                self.fit_lines.append(line)

                # Calculate metrics
                fwhm = calculate_fwhm(sig, gam, eta)
                area = calculate_area(amp, sig, gam, eta)

                results.append({
                    'Peak': i + 1,
                    'Center_2theta': cen,
                    'FWHM': fwhm,
                    'Area': area,
                    'Amplitude': amp,
                    'Sigma': sig,
                    'Gamma': gam,
                    'Eta': eta
                })

                # Add annotation directly on plot
                peak_height = amp / (sig * np.sqrt(2 * np.pi)) + popt[0] + popt[1] * cen
                annotation_text = f'P{i+1}: {cen:.3f}°\nFWHM: {fwhm:.4f}°\nArea: {area:.0f}'

                # Position annotation above peak
                ann = self.ax.annotate(annotation_text,
                                      xy=(cen, peak_height),
                                      xytext=(cen, peak_height * 1.08),
                                      fontsize=8, color=color,
                                      ha='center', va='bottom',
                                      fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3',
                                               facecolor='white',
                                               edgecolor=color,
                                               linewidth=1,
                                               alpha=0.9),
                                      zorder=15)
                self.result_annotations.append(ann)

                info_msg += f"Peak {i+1}: 2theta={cen:.4f}°, FWHM={fwhm:.4f}°, Area={area:.1f}\n"

            self.fit_results = pd.DataFrame(results)
            self.fitted = True

            self.ax.legend(fontsize=9, loc='best', framealpha=0.9)
            self.ax.set_title(f'{self.filename} - Fit Complete',
                            fontsize=13, fontweight='bold', color='#32CD32', pad=15)

            # Adjust layout
            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()

            self.update_info(info_msg)
            self.status_label.config(text="Fitting successful!")

            # Enable save button
            self.btn_save.config(state=tk.NORMAL)
            self.btn_clear_fit.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Fitting Error", f"Failed to fit peaks:\n{str(e)}")
            self.update_info(f"Fitting failed: {str(e)}\n")

    def clear_fit(self):
        """Clear fitting results but keep peak selections"""
        # Remove fit lines
        for line in self.fit_lines:
            line.remove()
        self.fit_lines = []

        # Remove result annotations
        for ann in self.result_annotations:
            ann.remove()
        self.result_annotations = []

        self.fitted = False
        self.fit_results = None

        self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar or scroll wheel to zoom',
                         fontsize=13, fontweight='bold', color='#9370DB', pad=15)
        self.ax.legend(fontsize=11, loc='best', framealpha=0.9)

        # Adjust layout
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

        self.btn_save.config(state=tk.DISABLED)
        self.btn_clear_fit.config(state=tk.DISABLED)
        self.update_info("Fit cleared. Peak selections preserved.\n")
        self.status_label.config(text=f"{len(self.selected_peaks)} peak(s) selected")

    def reset_peaks(self):
        """Clear all peak selections and fits"""
        # Remove markers and texts
        for marker in self.peak_markers:
            marker.remove()
        for text in self.peak_texts:
            text.remove()

        # Remove fit lines
        for line in self.fit_lines:
            line.remove()

        # Remove result annotations
        for ann in self.result_annotations:
            ann.remove()

        self.selected_peaks = []
        self.peak_markers = []
        self.peak_texts = []
        self.fit_lines = []
        self.result_annotations = []
        self.fitted = False
        self.fit_results = None

        if self.x is not None:
            self.ax.set_title(f'{self.filename}\nClick on peaks to select | Use toolbar or scroll wheel to zoom',
                            fontsize=13, fontweight='bold', color='#9370DB', pad=15)
            self.ax.legend(fontsize=11, loc='best', framealpha=0.9)

            # Adjust layout
            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()

            self.update_info("All peaks and fits cleared\n")
            self.status_label.config(text="Ready to select peaks")

        self.btn_save.config(state=tk.DISABLED)
        self.btn_clear_fit.config(state=tk.DISABLED)

    def save_results(self):
        """Save fitting results"""
        if self.fit_results is None:
            messagebox.showwarning("No Results", "Please fit peaks before saving!")
            return

        # Ask for save directory
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return

        try:
            # Save CSV
            self.fit_results['File'] = self.filename
            csv_path = os.path.join(save_dir, f"{self.filename}_fit_results.csv")
            self.fit_results.to_csv(csv_path, index=False)

            # Save figure
            fig_path = os.path.join(save_dir, f"{self.filename}_fit_plot.png")
            self.fig.savefig(fig_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')

            messagebox.showinfo("Success",
                              f"Results saved successfully!\n\n"
                              f"CSV: {csv_path}\n"
                              f"Plot: {fig_path}")

            self.update_info(f"Results saved to:\n{save_dir}\n")
            self.status_label.config(text="Results saved successfully!")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")
            self.update_info(f"Save failed: {str(e)}\n")

    def update_info(self, message):
        """Update info text area"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, message)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

# ---------- Main ----------
def main():
    root = tk.Tk()
    app = PeakFittingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
