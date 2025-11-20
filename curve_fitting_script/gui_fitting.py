# -*- coding: utf-8 -*-
"""
Peak Fitting Methods for GUI

Separated for better organization - these are the core fitting algorithms.

@author: candicewang928@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tkinter import messagebox

from .peak_fitting import (
    pseudo_voigt,
    voigt,
    calculate_fwhm,
    calculate_area,
    estimate_fwhm_robust
)
from .clustering import cluster_peaks_dbscan
from .background import fit_global_background


def fit_peaks_method(gui_instance):
    """
    Main peak fitting method - optimized group-based fitting with DBSCAN clustering.

    This is extracted as a separate function to keep the GUI class cleaner.
    It operates on the GUI instance to access and modify its state.

    Parameters
    ----------
    gui_instance : PeakFittingGUI
        The GUI instance containing all data and state

    Returns
    -------
    bool
        True if fitting succeeded, False otherwise
    """
    if len(gui_instance.selected_peaks) == 0:
        messagebox.showwarning("No Peaks", "Please select at least one peak first!")
        return False

    fit_method = gui_instance.fit_method.get()
    gui_instance.update_info(f"Fitting {len(gui_instance.selected_peaks)} peaks using {fit_method}...\n")
    gui_instance.status_label.config(text="Fitting in progress...")
    gui_instance.master.update()

    try:
        dx = np.mean(np.diff(gui_instance.x))

        # Sort peaks by position
        sorted_indices = sorted(range(len(gui_instance.selected_peaks)),
                               key=lambda i: gui_instance.x[gui_instance.selected_peaks[i]])
        sorted_peaks = [gui_instance.selected_peaks[i] for i in sorted_indices]

        # Step 1: Fit global background
        gui_instance.update_info("Fitting global background...\n")

        if len(gui_instance.bg_points) >= 2:
            # Use manually selected background points
            gui_instance.update_info("Using manually selected background points...\n")
            sorted_bg_points = sorted(gui_instance.bg_points, key=lambda p: p[0])
            bg_x = np.array([p[0] for p in sorted_bg_points])
            bg_y = np.array([p[1] for p in sorted_bg_points])

            global_bg = np.interp(gui_instance.x, bg_x, bg_y)
            global_bg_points = sorted_bg_points

            gui_instance.update_info(f"Using {len(global_bg_points)} manually selected background points\n")
        else:
            # Auto-generate background
            gui_instance.update_info("Auto-generating background from peak positions...\n")
            bg_method = 'piecewise'

            global_bg, global_bg_points = fit_global_background(
                gui_instance.x, gui_instance.y, sorted_peaks,
                method=bg_method
            )

            gui_instance.update_info(f"Piecewise linear background fitted "
                           f"with {len(global_bg_points)} anchor points\n")

        # Subtract global background
        y_nobg = gui_instance.y - global_bg

        # Step 2: Estimate FWHM for each peak
        fwhm_estimates = []
        baseline_estimates = []

        for idx in sorted_peaks:
            window_size = 50
            left = max(0, idx - window_size)
            right = min(len(gui_instance.x), idx + window_size)
            x_local = gui_instance.x[left:right]
            y_local = y_nobg[left:right]
            local_peak_idx = idx - left
            fwhm, baseline = estimate_fwhm_robust(x_local, y_local, local_peak_idx)
            fwhm_estimates.append(fwhm)
            baseline_estimates.append(baseline)

        # Step 3: Group peaks using DBSCAN clustering
        peak_positions = np.array([gui_instance.x[idx] for idx in sorted_peaks])

        avg_fwhm = np.mean(fwhm_estimates)
        eps = avg_fwhm * gui_instance.group_distance_threshold

        cluster_labels, n_clusters = cluster_peaks_dbscan(peak_positions, eps=eps)

        # Convert cluster labels to peak groups
        peak_groups = []
        for cluster_id in range(max(cluster_labels) + 1):
            group = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if group:
                peak_groups.append(group)

        peak_groups.sort(key=lambda g: gui_instance.x[sorted_peaks[g[0]]])

        gui_instance.update_info(f"DBSCAN clustering: {n_clusters} groups (eps={eps:.4f})\n")

        # Report grouped peaks
        for group in peak_groups:
            if len(group) > 1:
                original_nums = [sorted_indices[g] + 1 for g in group]
                gui_instance.update_info(f"Peaks {original_nums} grouped by DBSCAN clustering\n")

        use_voigt = (fit_method == "voigt")
        n_params_per_peak = 4 if use_voigt else 5

        # Store all results
        all_popt = {}
        group_windows = []

        # Step 4: Fit each group separately
        for g_idx, group in enumerate(peak_groups):
            gui_instance.status_label.config(text=f"Fitting group {g_idx+1}/{len(peak_groups)}...")
            gui_instance.master.update()

            group_peak_indices = [sorted_peaks[i] for i in group]
            group_fwhms = [fwhm_estimates[i] for i in group]
            is_overlapping = len(group) > 1

            # Create fitting window
            if gui_instance.overlap_mode:
                window_multiplier = 4 if is_overlapping else 2.5
            else:
                window_multiplier = 3 if is_overlapping else 2

            left_center = gui_instance.x[min(group_peak_indices)]
            right_center = gui_instance.x[max(group_peak_indices)]
            left_fwhm = group_fwhms[0]
            right_fwhm = group_fwhms[-1]

            window_left = left_center - left_fwhm * window_multiplier
            window_right = right_center + right_fwhm * window_multiplier

            left_idx = max(0, np.searchsorted(gui_instance.x, window_left))
            right_idx = min(len(gui_instance.x), np.searchsorted(gui_instance.x, window_right))
            group_windows.append((left_idx, right_idx))

            x_fit = gui_instance.x[left_idx:right_idx]
            y_fit_nobg = y_nobg[left_idx:right_idx]

            if len(x_fit) < 5:
                continue

            # Display info
            if len(group) == 1:
                gui_instance.update_info(f"Group {g_idx+1}: Peak {sorted_indices[group[0]]+1}, "
                               f"window [{window_left:.2f}, {window_right:.2f}]\n")
            else:
                peak_nums = [sorted_indices[g]+1 for g in group]
                gui_instance.update_info(f"Group {g_idx+1}: Peaks {peak_nums}, "
                               f"window [{window_left:.2f}, {window_right:.2f}]\n")

            # Build parameters
            p0 = []
            bounds_lower = []
            bounds_upper = []

            for i in group:
                idx = sorted_peaks[i]
                local_idx = idx - left_idx
                cen_guess = gui_instance.x[idx]
                fwhm_est = fwhm_estimates[i]

                sig_guess = fwhm_est / 2.355
                gam_guess = fwhm_est / 2

                peak_height = y_fit_nobg[local_idx] if local_idx < len(y_fit_nobg) else np.max(y_fit_nobg)
                if peak_height <= 0:
                    peak_height = np.max(y_fit_nobg) * 0.5

                amp_guess = peak_height * sig_guess * np.sqrt(2 * np.pi)

                y_range = np.max(y_fit_nobg) - np.min(y_fit_nobg)
                amp_lower = 0
                amp_multiplier = 10 if (is_overlapping or gui_instance.overlap_mode) else 5
                amp_upper = y_range * sig_guess * np.sqrt(2 * np.pi) * amp_multiplier

                # Center constraints
                if is_overlapping or gui_instance.overlap_mode:
                    center_tolerance = fwhm_est * 0.8
                else:
                    center_tolerance = fwhm_est * 0.5

                sig_lower = dx * 0.5
                sig_upper = fwhm_est * 3
                gam_lower = dx * 0.5
                gam_upper = fwhm_est * 3

                if use_voigt:
                    p0.extend([amp_guess, cen_guess, sig_guess, gam_guess])
                    bounds_lower.extend([amp_lower, cen_guess - center_tolerance, sig_lower, gam_lower])
                    bounds_upper.extend([amp_upper, cen_guess + center_tolerance, sig_upper, gam_upper])
                else:
                    p0.extend([amp_guess, cen_guess, sig_guess, gam_guess, 0.5])
                    bounds_lower.extend([amp_lower, cen_guess - center_tolerance, sig_lower, gam_lower, 0])
                    bounds_upper.extend([amp_upper, cen_guess + center_tolerance, sig_upper, gam_upper, 1.0])

            # Define fitting function
            n_group_peaks = len(group)
            if use_voigt:
                def make_func(n_peaks):
                    def func(x, *params):
                        y = np.zeros_like(x)
                        for i in range(n_peaks):
                            offset = i * 4
                            amp, cen, sig, gam = params[offset:offset+4]
                            y += voigt(x, amp, cen, sig, gam)
                        return y
                    return func
            else:
                def make_func(n_peaks):
                    def func(x, *params):
                        y = np.zeros_like(x)
                        for i in range(n_peaks):
                            offset = i * 5
                            amp, cen, sig, gam, eta = params[offset:offset+5]
                            y += pseudo_voigt(x, amp, cen, sig, gam, eta)
                        return y
                    return func

            multi_peak_func = make_func(n_group_peaks)

            # Perform fitting
            if is_overlapping or gui_instance.overlap_mode:
                max_iter = 30000
                ftol = 1e-9
                xtol = 1e-9
            else:
                max_iter = 10000
                ftol = 1e-8
                xtol = 1e-8

            try:
                popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_nobg,
                                   p0=p0, bounds=(bounds_lower, bounds_upper),
                                   method='trf', maxfev=max_iter,
                                   ftol=ftol, xtol=xtol)
            except Exception:
                try:
                    popt, _ = curve_fit(multi_peak_func, x_fit, y_fit_nobg,
                                       p0=p0, bounds=(bounds_lower, bounds_upper),
                                       method='dogbox', maxfev=50000)
                except Exception as e:
                    gui_instance.update_info(f"Group {g_idx+1} fit failed: {str(e)}\n")
                    continue

            # Store results
            for j, i in enumerate(group):
                offset = j * n_params_per_peak
                all_popt[i] = {
                    'params': popt[offset:offset+n_params_per_peak],
                    'group_idx': g_idx,
                    'window': (left_idx, right_idx)
                }

        # Step 5: Plot results
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_peaks)))

        # Clear manual background line if exists
        if gui_instance.bg_connect_line is not None:
            try:
                gui_instance.bg_connect_line.remove()
            except:
                pass
            gui_instance.bg_connect_line = None

        # Plot global background
        if len(global_bg_points) >= 2:
            bg_x = [p[0] for p in global_bg_points]
            bg_y = [p[1] for p in global_bg_points]
            bg_markers, = gui_instance.ax.plot(bg_x, bg_y, 'o', color='#4169E1',
                                      markersize=6, alpha=0.8, zorder=3)
            gui_instance.fit_lines.append(bg_markers)
            bg_label = 'Manual Background' if len(gui_instance.bg_points) >= 2 else 'Auto Background'
            bg_line, = gui_instance.ax.plot(gui_instance.x, global_bg, '-', color='#4169E1',
                                   linewidth=1.5, alpha=0.6,
                                   label=bg_label, zorder=3)
            gui_instance.fit_lines.append(bg_line)

        # Plot total fit for each group
        for g_idx, (left, right) in enumerate(group_windows):
            x_region = gui_instance.x[left:right]
            x_smooth = np.linspace(x_region.min(), x_region.max(), 400)

            bg_smooth = np.interp(x_smooth, gui_instance.x, global_bg)

            y_total = bg_smooth.copy()
            group = peak_groups[g_idx]

            for i in group:
                if i not in all_popt:
                    continue
                params = all_popt[i]['params']
                if use_voigt:
                    y_total += voigt(x_smooth, *params)
                else:
                    y_total += pseudo_voigt(x_smooth, *params)

            if g_idx == 0:
                line1, = gui_instance.ax.plot(x_smooth, y_total, color='#FF0000', linewidth=1.5,
                                    label='Total Fit', zorder=5, alpha=0.9)
            else:
                line1, = gui_instance.ax.plot(x_smooth, y_total, color='#FF0000', linewidth=1.5,
                                    zorder=5, alpha=0.9)
            gui_instance.fit_lines.append(line1)

        # Plot individual peak components
        for i in range(len(sorted_peaks)):
            if i not in all_popt:
                continue

            params = all_popt[i]['params']
            left, right = all_popt[i]['window']

            x_smooth = np.linspace(gui_instance.x[left], gui_instance.x[right], 400)

            if use_voigt:
                y_component = voigt(x_smooth, *params)
            else:
                y_component = pseudo_voigt(x_smooth, *params)

            bg_smooth = np.interp(x_smooth, gui_instance.x, global_bg)
            y_with_bg = y_component + bg_smooth

            original_idx = sorted_indices[i]
            line_comp, = gui_instance.ax.plot(x_smooth, y_with_bg, '--',
                                     color=colors[i], linewidth=1.2, alpha=0.7, zorder=4,
                                     label=f'Peak {original_idx+1}')
            gui_instance.fit_lines.append(line_comp)

        # Step 6: Extract results
        results = []
        info_msg = f"Fitting Results ({fit_method}):\n" + "="*50 + "\n"

        for i in range(len(sorted_peaks)):
            original_idx = sorted_indices[i]

            if i not in all_popt:
                continue

            params = all_popt[i]['params']

            if use_voigt:
                amp, cen, sig, gam = params
                fwhm = 2.355 * sig
                area = amp
                eta = "N/A"
            else:
                amp, cen, sig, gam, eta = params
                fwhm = calculate_fwhm(sig, gam, eta)
                area = calculate_area(amp, sig, gam, eta)

            results.append({
                'Peak': original_idx + 1,
                'Center_2theta': cen,
                'FWHM': fwhm,
                'Area': area,
                'Amplitude': amp,
                'Sigma': sig,
                'Gamma': gam,
                'Eta': eta
            })

            info_msg += f"Peak {original_idx+1}: 2theta={cen:.4f}, FWHM={fwhm:.5f}, Area={area:.1f}\n"

        results.sort(key=lambda r: r['Peak'])

        gui_instance.fit_results = pd.DataFrame(results)
        gui_instance.fitted = True

        # Update results table
        for item in gui_instance.results_tree.get_children():
            gui_instance.results_tree.delete(item)

        for r in results:
            eta_str = f"{r['Eta']:.3f}" if isinstance(r['Eta'], float) else r['Eta']
            gui_instance.results_tree.insert('', 'end', values=(
                f"{r['Peak']}",
                f"{r['Center_2theta']:.4f}",
                f"{r['FWHM']:.5f}",
                f"{r['Area']:.2f}",
                f"{r['Amplitude']:.2f}",
                f"{r['Sigma']:.5f}",
                f"{r['Gamma']:.5f}",
                eta_str
            ))

        gui_instance.ax.set_title(f'{gui_instance.filename} - Fit Complete ({fit_method})',
                        fontsize=14, fontweight='bold', color='#32CD32')
        gui_instance.canvas.draw()

        gui_instance.update_info(info_msg)
        gui_instance.status_label.config(text="Fitting successful!")

        gui_instance.btn_save.config(state="normal")
        gui_instance.btn_quick_save.config(state="normal")
        gui_instance.btn_clear_fit.config(state="normal")

        return True

    except Exception as e:
        import traceback
        messagebox.showerror("Fitting Error", f"Failed to fit peaks:\n{str(e)}")
        gui_instance.update_info(f"Fitting failed: {traceback.format_exc()}\n")
        gui_instance.status_label.config(text="Fitting failed")
        return False
