#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curve_fitting
==============

This module provides utilities for fitting diffraction peaks using either a Voigt
or pseudo‑Voigt profile. It includes functions for reading simple two‑column
data files (``*.xy``), detecting peaks, fitting each peak, and producing
publication‑quality plots and CSV summaries.  When run as a script, it will
process all ``*.xy`` files in a directory and write the results to an
``fit_output`` directory.

The original code was embedded directly in a README and hard‑coded file
paths.  It has been refactored into this reusable module with a command‑line
interface.  Use the provided ``requirements.txt`` to install the necessary
dependencies (NumPy, SciPy, matplotlib, and pandas).

Example usage from the command line::

    python3 curve_fitting.py /path/to/data --method pseudo --output /tmp/out

This will search ``/path/to/data`` for files ending in ``.xy``, fit the
peaks using a pseudo‑Voigt profile, and store plots/CSV summaries in
``/tmp/out``.  Omit ``--output`` to write results into a ``fit_output``
subdirectory of the input folder.

"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz

###############################################################################
# Peak profile definitions
###############################################################################

def voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float) -> np.ndarray:
    """Return a Voigt line shape.

    Parameters
    ----------
    x : ndarray
        Independent variable (e.g. two‑theta values).
    amplitude : float
        Peak amplitude.
    center : float
        Peak center.
    sigma : float
        Standard deviation of the Gaussian component.
    gamma : float
        Half‑width at half‑maximum of the Lorentzian component.

    Returns
    -------
    ndarray
        The Voigt profile evaluated at ``x``.
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def fit_voigt(xdata: np.ndarray, ydata: np.ndarray, p0: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a Voigt profile to the provided data.

    If initial parameters are not supplied, reasonable guesses are computed
    based on the maximum of ``ydata`` and the spread of ``xdata``.  Bounds
    constrain the parameters to physically sensible ranges.

    Parameters
    ----------
    xdata, ydata : ndarray
        The x and y values to fit.
    p0 : list[float], optional
        Initial guesses for ``amplitude``, ``center``, ``sigma``, and
        ``gamma``.

    Returns
    -------
    tuple
        ``(popt, pcov)`` from :func:`scipy.optimize.curve_fit`.
    """
    if p0 is None:
        amplitude_guess = np.max(ydata)
        center_guess = xdata[np.argmax(ydata)]
        sigma_guess = np.std(xdata) / 5
        gamma_guess = sigma_guess
        p0 = [amplitude_guess, center_guess, sigma_guess, gamma_guess]
    bounds = ([0, xdata.min(), 0, 0], [np.inf, xdata.max(), np.inf, np.inf])
    popt, pcov = curve_fit(voigt, xdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
    return popt, pcov


def pseudo_voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float, eta: float) -> np.ndarray:
    """Return a pseudo‑Voigt line shape (weighted sum of Gaussian and Lorentzian)."""
    gaussian = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    lorentzian = amplitude * gamma ** 2 / (((x - center) ** 2) + gamma ** 2) / (np.pi * gamma)
    return eta * lorentzian + (1 - eta) * gaussian


def fit_pseudo_voigt(xdata: np.ndarray, ydata: np.ndarray, p0: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a pseudo‑Voigt profile to the provided data."""
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


###############################################################################
# Data processing functions
###############################################################################

def _estimate_background(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a simple linear background based on the first and last points.

    Returns a slope and intercept such that ``background = intercept + slope * (x - x[0])``.
    """
    bg_left = np.mean(y[:1])
    bg_right = np.mean(y[-1:])
    slope = (bg_right - bg_left) / (x[-1] - x[0])
    intercept = bg_left
    return slope, intercept


def process_file(
    file_path: str,
    save_dir: str,
    fit_method: str = "pseudo",
    peak_distance: int = 30,
    window: int = 55,
) -> Optional[pd.DataFrame]:
    """Process a single ``.xy`` file: detect peaks, fit them, and save results.

    Parameters
    ----------
    file_path : str
        Path to the two‑column data file (x and y values).
    save_dir : str
        Directory where plots and CSV files will be saved.
    fit_method : {"voigt", "pseudo"}, default ``"pseudo"``
        Which line shape to use when fitting peaks.
    peak_distance : int, default 30
        Minimum distance between detected peaks (passed to ``find_peaks``).
    window : int, default 55
        Half‑width of the window around each detected peak used for fitting.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing the fit results for this file; returns ``None`` if no
        valid peaks were found.
    """
    try:
        # Many ``.xy`` files exported from Dioptas use latin‑1 encoding
        with open(file_path, encoding="latin1") as f:
            data = np.genfromtxt(f, comments="#")
        x = data[:, 0]
        y = data[:, 1]
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return None

    filename = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n📄 Processing file: {filename}")

    # Detect all peaks with a minimum spacing; filter them by a neighbourhood criterion
    all_peaks, _ = find_peaks(y, distance=peak_distance)
    filtered_peaks = []
    for i in all_peaks:
        # require at least one full window on either side
        if window <= i < len(y) - window:
            left_val = y[i - window]
            right_val = y[i + window]
            min_neighbor = min(left_val, right_val)
            if y[i] >= min_neighbor * 1.15:
                filtered_peaks.append(i)

    peaks = filtered_peaks
    print(f"   ➤ Detected {len(peaks)} peaks (with +15% neighbour filter)")
    if not peaks:
        print("   ⚠️ No valid peaks found, skipping this file.")
        return None

    results: List[dict] = []
    subplot_cols = 3
    subplot_rows = int(np.ceil(len(peaks) / subplot_cols))
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(5 * subplot_cols, 4 * subplot_rows))
    axs = axs.flatten()

    for idx, peak_index in enumerate(peaks):
        left = max(0, peak_index - window)
        right = min(len(x), peak_index + window)
        x_local = x[left:right]
        y_local = y[left:right]

        # Subtract background for fitting
        slope, intercept = _estimate_background(x_local, y_local)
        background = intercept + slope * (x_local - x_local[0])
        y_fit_input = y_local - background

        # High resolution x axis for smooth plot
        x_smooth = np.linspace(x_local.min(), x_local.max(), 5000)

        try:
            if fit_method == "voigt":
                popt, _ = fit_voigt(x_local, y_fit_input)
                # compute the fitted curve including background
                y_fit_corrected = pseudo_voigt(x_smooth, *popt, eta=0.5) + (intercept + slope * (x_smooth - x_local[0]))  # approximate using pseudo
                fit_label = "Voigt fit (approx)"
                results.append({
                    "Peak #": idx + 1,
                    "Center": popt[1],
                    "Amplitude": popt[0],
                    "Sigma": popt[2],
                    "Gamma": popt[3],
                    "Eta": np.nan,
                })
            else:
                popt, _ = fit_pseudo_voigt(x_local, y_fit_input)
                y_fit_corrected = pseudo_voigt(x_smooth, *popt) + (intercept + slope * (x_smooth - x_local[0]))
                fit_label = "Pseudo‑Voigt fit"
                results.append({
                    "Peak #": idx + 1,
                    "Center": popt[1],
                    "Amplitude": popt[0],
                    "Sigma": popt[2],
                    "Gamma": popt[3],
                    "Eta": popt[4],
                })

            # Plot raw data and fitted curve
            ax = axs[idx]
            ax.plot(x_local, y_local, color="black", label="Raw data")
            ax.plot(x_smooth, y_fit_corrected, color="#BA55D3", linestyle="--", linewidth=2, label=fit_label)
            ax.plot(x_smooth, intercept + slope * (x_smooth - x_local[0]), color="#FF69B4", linestyle="-", linewidth=1.5, label="Background")
            ax.set_title(f"Peak {idx + 1} @ {popt[1]:.3f}")
            ax.set_xlabel("2θ (degree)")
            ax.set_ylabel("Intensity")
            ax.legend()
            ax.set_facecolor("white")
        except RuntimeError:
            print(f"⚠️ Peak {idx + 1} fit failed 💔")
            axs[idx].text(0.3, 0.5, "Fit failed", transform=axs[idx].transAxes, fontsize=12, color="red")

    # Remove unused subplots
    for j in range(len(peaks), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"{filename} - {fit_method.capitalize()} fit", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig_path = os.path.join(save_dir, f"{filename}_fit.png")
    fig.savefig(fig_path)
    plt.close(fig)

    df = pd.DataFrame(results)
    df["File"] = filename
    csv_path = os.path.join(save_dir, f"{filename}_results.csv")
    df.to_csv(csv_path, index=False)
    return df


def process_directory(folder: str, save_dir: Optional[str] = None, fit_method: str = "pseudo") -> Optional[str]:
    """Process all ``.xy`` files in a directory.

    Parameters
    ----------
    folder : str
        Directory containing input ``.xy`` files.
    save_dir : str or None, default None
        Where to save output; if ``None``, an ``fit_output`` directory will
        be created inside ``folder``.
    fit_method : {"voigt", "pseudo"}, default "pseudo"
        Which peak profile to use for fitting.

    Returns
    -------
    str or None
        Path to the combined CSV file with all results, or ``None`` if no files
        were processed.
    """
    if save_dir is None:
        save_dir = os.path.join(folder, "fit_output")
    os.makedirs(save_dir, exist_ok=True)

    files = [f for f in os.listdir(folder) if f.lower().endswith(".xy")]
    files.sort()
    if not files:
        print(f"No .xy files found in {folder}")
        return None

    all_dfs: List[pd.DataFrame] = []
    for fname in files:
        fpath = os.path.join(folder, fname)
        df = process_file(fpath, save_dir, fit_method=fit_method)
        if df is not None:
            all_dfs.append(df)
            # Insert a blank row as a separator when concatenating
            all_dfs.append(pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns))

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv_path = os.path.join(save_dir, "all_results_with_blank_lines.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n📦 Total results saved to: {combined_csv_path}")
        return combined_csv_path
    return None


def main() -> None:
    """Entry point for command‑line execution."""
    parser = argparse.ArgumentParser(description="Batch fit diffraction peaks in .xy files using Voigt or pseudo‑Voigt profiles.")
    parser.add_argument(
        "folder",
        help="Directory containing input .xy files",
    )
    parser.add_argument(
        "--method",
        choices=["voigt", "pseudo"],
        default="pseudo",
        help="Fitting method to use (default: pseudo)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to save outputs (default: <folder>/fit_output)",
    )
    args = parser.parse_args()
    combined_csv = process_directory(args.folder, args.output, fit_method=args.method)
    if combined_csv:
        print(f"Results written to {combined_csv}")


if __name__ == "__main__":
    main()