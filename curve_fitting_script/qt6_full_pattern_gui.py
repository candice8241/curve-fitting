#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qt6 GUI for full-pattern fitting (Rietveld / Pawley / Le Bail).
Supports CIF/JCPDS peak positions, multiple phases, and adjustable profiles.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.special import wofz

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class CellParameters:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)


@dataclass
class Phase:
    name: str
    source: str
    cell: Optional[CellParameters] = None
    hkl_list: List[Tuple[int, int, int, float]] = field(default_factory=list)
    fixed_peaks: List[Tuple[float, float]] = field(default_factory=list)
    peaks: List[Tuple[float, float]] = field(default_factory=list)
    scale: float = 1.0
    color: str = "#2563eb"
    visible: bool = True
    generated_hkl: bool = False


@dataclass
class ProfileParams:
    shape: str
    sigma: float
    gamma: float
    eta: float


def parse_numeric(text: str) -> Optional[float]:
    match = FLOAT_RE.search(str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def read_xy_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[Tuple[float, float]] = []
    with open(path, "r", encoding="latin1") as handle:
        for line in handle:
            if not line.strip():
                continue
            stripped = line.lstrip()
            if stripped.startswith(("#", "!", ";", "//")):
                continue
            values = FLOAT_RE.findall(line)
            if len(values) >= 2:
                rows.append((float(values[0]), float(values[1])))
    if not rows:
        raise ValueError("No numeric data found.")
    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1]


def split_cif_tokens(line: str) -> List[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def parse_cif(
    path: str, wavelength: Optional[float] = None
) -> Tuple[Optional[CellParameters], List[Tuple[int, int, int, float]], List[Tuple[float, float]]]:
    cell_values: Dict[str, Optional[float]] = {
        "a": None,
        "b": None,
        "c": None,
        "alpha": None,
        "beta": None,
        "gamma": None,
    }

    hkl_list: List[Tuple[int, int, int, float]] = []
    fixed_peaks: List[Tuple[float, float]] = []

    with open(path, "r", encoding="latin1") as handle:
        lines = handle.readlines()

    idx = 0
    loops: List[Tuple[List[str], List[List[str]]]] = []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith("#"):
            idx += 1
            continue

        if line.startswith("_cell_length_a"):
            parts = line.split(None, 1)
            if len(parts) > 1:
                cell_values["a"] = parse_numeric(parts[1])
        elif line.startswith("_cell_length_b"):
            parts = line.split(None, 1)
            if len(parts) > 1:
                cell_values["b"] = parse_numeric(parts[1])
        elif line.startswith("_cell_length_c"):
            parts = line.split(None, 1)
            if len(parts) > 1:
                cell_values["c"] = parse_numeric(parts[1])
        elif line.startswith("_cell_angle_alpha"):
            parts = line.split(None, 1)
            if len(parts) > 1:
                cell_values["alpha"] = parse_numeric(parts[1])
        elif line.startswith("_cell_angle_beta"):
            parts = line.split(None, 1)
            if len(parts) > 1:
                cell_values["beta"] = parse_numeric(parts[1])
        elif line.startswith("_cell_angle_gamma"):
            parts = line.split(None, 1)
            if len(parts) > 1:
                cell_values["gamma"] = parse_numeric(parts[1])
        elif line.startswith("loop_"):
            idx += 1
            tags: List[str] = []
            while idx < len(lines):
                tag_line = lines[idx].strip()
                if tag_line.startswith("_"):
                    tags.append(tag_line.split()[0])
                    idx += 1
                else:
                    break
            data_rows: List[List[str]] = []
            tokens: List[str] = []
            while idx < len(lines):
                data_line = lines[idx].strip()
                if not data_line or data_line.startswith("#"):
                    idx += 1
                    continue
                if data_line.startswith("loop_") or data_line.startswith("_"):
                    break
                tokens.extend(split_cif_tokens(data_line))
                while len(tokens) >= len(tags):
                    row = tokens[: len(tags)]
                    tokens = tokens[len(tags) :]
                    data_rows.append(row)
                idx += 1
            loops.append((tags, data_rows))
            continue
        idx += 1

    for tags, data in loops:
        tag_map = {tag.lower(): index for index, tag in enumerate(tags)}

        def get_index(candidates: List[str]) -> Optional[int]:
            for name in candidates:
                if name in tag_map:
                    return tag_map[name]
            return None

        idx_h = get_index(["_refln_index_h", "_refln.index_h"])
        idx_k = get_index(["_refln_index_k", "_refln.index_k"])
        idx_l = get_index(["_refln_index_l", "_refln.index_l"])
        idx_d = get_index(["_refln_d_spacing", "_refln.d_spacing", "_refln_d_calc"])
        idx_tth = get_index(
            ["_refln_two_theta", "_refln_2theta", "_refln_2theta_calc", "_refln.2theta"]
        )
        idx_i = get_index(
            [
                "_refln_intensity_meas",
                "_refln_intensity_calc",
                "_refln_intensity",
                "_refln_f_squared_meas",
                "_refln_f_squared_calc",
                "_refln_f_squared",
                "_refln_f_meas",
                "_refln_f_calc",
            ]
        )

        if idx_h is None and idx_d is None and idx_tth is None:
            continue

        for row in data:
            intensity = parse_numeric(row[idx_i]) if idx_i is not None else None
            intensity_val = intensity if intensity is not None else 1.0

            if idx_h is not None and idx_k is not None and idx_l is not None:
                h_val = parse_numeric(row[idx_h])
                k_val = parse_numeric(row[idx_k])
                l_val = parse_numeric(row[idx_l])
                if h_val is None or k_val is None or l_val is None:
                    continue
                hkl_list.append((int(h_val), int(k_val), int(l_val), intensity_val))
            elif idx_tth is not None:
                two_theta = parse_numeric(row[idx_tth])
                if two_theta is None:
                    continue
                fixed_peaks.append((two_theta, intensity_val))
            elif idx_d is not None:
                d_val = parse_numeric(row[idx_d])
                if d_val is None:
                    continue
                if wavelength is not None:
                    two_theta = two_theta_from_d(d_val, wavelength)
                    if two_theta is None:
                        continue
                    fixed_peaks.append((two_theta, intensity_val))
                else:
                    fixed_peaks.append((d_val, intensity_val))

    cell = None
    if all(cell_values[key] is not None for key in ("a", "b", "c", "alpha", "beta", "gamma")):
        cell = CellParameters(
            a=cell_values["a"],
            b=cell_values["b"],
            c=cell_values["c"],
            alpha=cell_values["alpha"],
            beta=cell_values["beta"],
            gamma=cell_values["gamma"],
        )

    return cell, hkl_list, fixed_peaks


def parse_jcpds(path: str) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    rows: List[Tuple[float, float]] = []
    wavelength = None
    with open(path, "r", encoding="latin1") as handle:
        for line in handle:
            lower = line.lower()
            if "lambda" in lower:
                match = FLOAT_RE.search(line)
                if match:
                    wavelength = float(match.group(0))
            values = FLOAT_RE.findall(line)
            if len(values) >= 2:
                rows.append((float(values[0]), float(values[1])))
    if not rows:
        raise ValueError("No numeric data found in JCPDS file.")
    return rows, wavelength


def hkl_multiplicity(h: int, k: int, l: int) -> int:
    values = [abs(h), abs(k), abs(l)]
    nonzero = sum(1 for value in values if value != 0)
    unique_counts: Dict[int, int] = {}
    for value in values:
        unique_counts[value] = unique_counts.get(value, 0) + 1
    permutations = 6
    for count in unique_counts.values():
        permutations //= math.factorial(count)
    return permutations * (2 ** nonzero)


def cell_to_metric_inverse(cell: CellParameters) -> np.ndarray:
    a, b, c = cell.a, cell.b, cell.c
    alpha = math.radians(cell.alpha)
    beta = math.radians(cell.beta)
    gamma = math.radians(cell.gamma)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    metric = np.array(
        [
            [a * a, a * b * cos_g, a * c * cos_b],
            [a * b * cos_g, b * b, b * c * cos_a],
            [a * c * cos_b, b * c * cos_a, c * c],
        ],
        dtype=float,
    )
    return np.linalg.inv(metric)


def d_spacing_from_hkl(h: int, k: int, l: int, cell: CellParameters) -> Optional[float]:
    metric_inv = cell_to_metric_inverse(cell)
    hkl = np.array([h, k, l], dtype=float)
    inv_d2 = float(hkl @ metric_inv @ hkl)
    if inv_d2 <= 0:
        return None
    return 1.0 / math.sqrt(inv_d2)


def two_theta_from_d(d_spacing: float, wavelength: float) -> Optional[float]:
    if d_spacing <= 0:
        return None
    sin_theta = wavelength / (2.0 * d_spacing)
    if sin_theta <= 0 or sin_theta > 1:
        return None
    theta = math.asin(sin_theta)
    return math.degrees(2.0 * theta)


def merge_peaks(peaks: List[Tuple[float, float]], tolerance: float = 0.02) -> List[Tuple[float, float]]:
    if not peaks:
        return []
    peaks_sorted = sorted(peaks, key=lambda item: item[0])
    merged: List[Tuple[float, float]] = []
    current_pos, current_int = peaks_sorted[0]
    count = 1
    for pos, intensity in peaks_sorted[1:]:
        if abs(pos - current_pos) <= tolerance:
            current_pos = (current_pos * count + pos) / (count + 1)
            current_int += intensity
            count += 1
        else:
            merged.append((current_pos, current_int))
            current_pos, current_int, count = pos, intensity, 1
    merged.append((current_pos, current_int))
    return merged


def generate_hkl_list(
    cell: CellParameters,
    two_theta_min: float,
    two_theta_max: float,
    wavelength: float,
    max_index: int = 14,
) -> List[Tuple[int, int, int, float]]:
    if two_theta_max <= 0:
        return []
    theta_max = math.radians(two_theta_max / 2.0)
    sin_theta_max = math.sin(theta_max)
    if sin_theta_max <= 0:
        return []
    d_min = wavelength / (2.0 * sin_theta_max)
    max_length = max(cell.a, cell.b, cell.c)
    index_limit = int(math.ceil(max_length / d_min)) + 1
    index_limit = max(4, min(index_limit, max_index))
    peaks: List[Tuple[int, int, int, float]] = []
    for h in range(0, index_limit + 1):
        for k in range(0, index_limit + 1):
            for l in range(0, index_limit + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                d_spacing = d_spacing_from_hkl(h, k, l, cell)
                if d_spacing is None:
                    continue
                two_theta = two_theta_from_d(d_spacing, wavelength)
                if two_theta is None:
                    continue
                if two_theta < two_theta_min or two_theta > two_theta_max:
                    continue
                intensity = float(hkl_multiplicity(h, k, l))
                peaks.append((h, k, l, intensity))
    return peaks


def gaussian_profile(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def lorentzian_profile(x: np.ndarray, center: float, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return np.zeros_like(x)
    return gamma / (math.pi * ((x - center) ** 2 + gamma ** 2))


def voigt_profile(x: np.ndarray, center: float, sigma: float, gamma: float) -> np.ndarray:
    if sigma <= 0 or gamma <= 0:
        return np.zeros_like(x)
    z = ((x - center) + 1j * gamma) / (sigma * math.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * math.sqrt(2.0 * math.pi))


def profile_function(x: np.ndarray, center: float, params: ProfileParams) -> np.ndarray:
    shape = params.shape.lower()
    if shape == "gaussian":
        return gaussian_profile(x, center, params.sigma)
    if shape == "lorentzian":
        return lorentzian_profile(x, center, params.gamma)
    if shape == "pseudo-voigt":
        gauss = gaussian_profile(x, center, params.sigma)
        lorentz = lorentzian_profile(x, center, params.gamma)
        return params.eta * lorentz + (1.0 - params.eta) * gauss
    return voigt_profile(x, center, params.sigma, params.gamma)


def build_profile_matrix(
    x: np.ndarray,
    centers: List[float],
    params: ProfileParams,
    zero_shift: float,
) -> np.ndarray:
    matrix = np.zeros((len(x), len(centers)), dtype=float)
    for col_idx, center in enumerate(centers):
        matrix[:, col_idx] = profile_function(x, center + zero_shift, params)
    return matrix


def update_phase_peaks(
    phase: Phase,
    two_theta_min: float,
    two_theta_max: float,
    wavelength: float,
) -> None:
    peaks: List[Tuple[float, float]] = []
    if phase.cell:
        if not phase.hkl_list:
            phase.hkl_list = generate_hkl_list(
                phase.cell, two_theta_min, two_theta_max, wavelength
            )
            phase.generated_hkl = True
        for h, k, l, intensity in phase.hkl_list:
            d_spacing = d_spacing_from_hkl(h, k, l, phase.cell)
            if d_spacing is None:
                continue
            two_theta = two_theta_from_d(d_spacing, wavelength)
            if two_theta is None:
                continue
            if two_theta < two_theta_min or two_theta > two_theta_max:
                continue
            peaks.append((two_theta, intensity))
        peaks = merge_peaks(peaks)
    elif phase.fixed_peaks:
        for pos, intensity in phase.fixed_peaks:
            if pos < two_theta_min or pos > two_theta_max:
                continue
            peaks.append((pos, intensity))
        peaks = merge_peaks(peaks)
    phase.peaks = peaks


def compute_background(x: np.ndarray, b0: float, b1: float) -> np.ndarray:
    x_ref = float(np.mean(x))
    return b0 + b1 * (x - x_ref)


def interpolate_background(
    x: np.ndarray, points: List[Tuple[float, float]], fallback: np.ndarray
) -> np.ndarray:
    if len(points) < 2:
        return fallback
    by_x: Dict[float, List[float]] = {}
    for x_val, y_val in points:
        by_x.setdefault(float(x_val), []).append(float(y_val))
    x_vals = sorted(by_x.keys())
    if len(x_vals) < 2:
        return fallback
    y_vals = [sum(by_x[x_val]) / len(by_x[x_val]) for x_val in x_vals]
    return np.interp(x, x_vals, y_vals, left=y_vals[0], right=y_vals[-1])


def compute_pattern(
    x: np.ndarray,
    y_obs: Optional[np.ndarray],
    phases: List[Phase],
    method: str,
    profile: ProfileParams,
    background: np.ndarray,
    zero_shift: float,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    usable_phases = [phase for phase in phases if phase.visible and phase.peaks]

    if method in ("Pawley", "Le Bail") and y_obs is not None and usable_phases:
        centers: List[float] = []
        for phase_index, phase in enumerate(usable_phases):
            for center, _ in phase.peaks:
                centers.append(center)
        if not centers:
            return background, background, []
        matrix = build_profile_matrix(x, centers, profile, zero_shift)
        target = y_obs - background
        target = np.clip(target, 0.0, None)
        intensities, _ = nnls(matrix, target)
        calc = background + matrix @ intensities
        phase_patterns: List[np.ndarray] = [np.zeros_like(x) for _ in usable_phases]
        offset = 0
        for phase_index, phase in enumerate(usable_phases):
            count = len(phase.peaks)
            if count:
                phase_patterns[phase_index] = matrix[:, offset : offset + count] @ intensities[
                    offset : offset + count
                ]
            offset += count
        return calc, background, phase_patterns

    phase_patterns = []
    calc = background.copy()
    for phase in usable_phases:
        phase_calc = np.zeros_like(x)
        for center, intensity in phase.peaks:
            phase_calc += intensity * profile_function(x, center + zero_shift, profile)
        phase_calc *= phase.scale
        phase_patterns.append(phase_calc)
        calc += phase_calc
    return calc, background, phase_patterns


class FitPlotCanvas(FigureCanvas):
    def __init__(self, parent: QtWidgets.QWidget):
        fig = Figure(figsize=(8, 6), facecolor="#f8fafc")
        super().__init__(fig)
        self.setParent(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
        grid = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.05)
        self.axes_main = fig.add_subplot(grid[0])
        self.axes_diff = fig.add_subplot(grid[1], sharex=self.axes_main)
        self.axes_main.set_ylabel("Intensity")
        self.axes_diff.set_xlabel("2theta")
        self.axes_diff.set_ylabel("Diff")
        self.axes_main.set_facecolor("#f8fafc")
        self.axes_diff.set_facecolor("#f8fafc")
        self.axes_main.grid(True, color="#e2e8f0", linewidth=0.7)
        self.axes_diff.grid(True, color="#e2e8f0", linewidth=0.7)
        self.calc_line = None
        self.background_markers = None
        self.current_x = None
        self.current_calc = None
        self.current_obs = None
        self.mpl_connect("scroll_event", self.on_scroll)

    def on_scroll(self, event) -> None:
        if event.inaxes is None:
            return
        base_scale = 1.2
        if event.button == "up":
            scale_factor = 1 / base_scale
        else:
            scale_factor = base_scale

        x_min, x_max = self.axes_main.get_xlim()
        x_center = event.xdata if event.xdata is not None else 0.5 * (x_min + x_max)
        x_left = x_center - (x_center - x_min) * scale_factor
        x_right = x_center + (x_max - x_center) * scale_factor
        self.axes_main.set_xlim(x_left, x_right)
        self.axes_diff.set_xlim(x_left, x_right)

        axis = event.inaxes
        y_min, y_max = axis.get_ylim()
        y_center = event.ydata if event.ydata is not None else 0.5 * (y_min + y_max)
        y_low = y_center - (y_center - y_min) * scale_factor
        y_high = y_center + (y_max - y_center) * scale_factor
        axis.set_ylim(y_low, y_high)
        self.draw_idle()

    def is_over_calc_line(self, event) -> bool:
        if self.calc_line is None or self.current_x is None or self.current_calc is None:
            return False
        if event.xdata is None or event.ydata is None:
            return False
        if self.calc_line.contains(event)[0]:
            return True
        index = int(np.clip(np.searchsorted(self.current_x, event.xdata), 0, len(self.current_x) - 1))
        y_val = float(self.current_calc[index])
        y_range = float(np.ptp(self.current_calc)) or 1.0
        return abs(event.ydata - y_val) / y_range <= 0.03

    def render(
        self,
        x: Optional[np.ndarray],
        y_obs: Optional[np.ndarray],
        y_calc: Optional[np.ndarray],
        y_bkg: Optional[np.ndarray],
        background_points: List[Tuple[float, float]],
        phases: List[Phase],
        selected_index: Optional[int],
    ) -> None:
        self.axes_main.clear()
        self.axes_diff.clear()
        self.calc_line = None
        self.background_markers = None
        self.current_x = x
        self.current_calc = y_calc
        self.current_obs = y_obs

        if x is None or y_calc is None:
            self.draw()
            return

        if y_obs is not None:
            self.axes_main.plot(
                x,
                y_obs,
                linestyle="none",
                marker="+",
                markersize=3,
                color="#2563eb",
                label="obs",
            )

        self.calc_line = self.axes_main.plot(
            x, y_calc, color="#22c55e", linewidth=1.8, label="calc"
        )[0]
        self.calc_line.set_picker(5)
        self.calc_line.set_pickradius(8)
        if y_bkg is not None:
            self.axes_main.plot(x, y_bkg, color="#ef4444", linewidth=1.2, label="bkg")

        if background_points:
            points_x = [point[0] for point in background_points]
            points_y = [point[1] for point in background_points]
            self.background_markers = self.axes_main.scatter(
                points_x,
                points_y,
                s=28,
                color="#f59e0b",
                edgecolors="#f8fafc",
                linewidths=0.5,
                label="bkg pts",
                zorder=4,
            )

        if y_obs is not None:
            diff = y_obs - y_calc
            self.axes_diff.plot(x, diff, color="#06b6d4", linewidth=1.0, label="diff")
            self.axes_diff.axhline(0.0, color="black", linewidth=0.8)
            base_range = float(np.max(y_calc) - np.min(y_calc))
            diff_range = float(np.max(np.abs(diff)))
            target_range = max(diff_range, 0.05 * base_range)
            self.axes_diff.set_ylim(-target_range, target_range)

        # Tick marks for phase peaks (use diff axis when available)
        tick_axis = self.axes_diff if y_obs is not None else self.axes_main
        if y_obs is not None:
            diff = y_obs - y_calc
            diff_range = float(np.max(np.abs(diff)))
            base_range = diff_range if diff_range > 0 else 1.0
            tick_base = -1.2 * base_range
            tick_height = 0.25 * base_range
            row_gap = 0.35 * base_range
            self.axes_diff.set_ylim(
                min(self.axes_diff.get_ylim()[0], tick_base - row_gap * 1.4),
                self.axes_diff.get_ylim()[1],
            )
        else:
            base_range = float(np.max(y_calc) - np.min(y_calc))
            tick_base = float(np.min(y_calc)) - 0.08 * base_range
            tick_height = 0.04 * base_range
            row_gap = 0.05 * base_range

        x_min, x_max = float(np.min(x)), float(np.max(x))
        label_rows = [[], []]
        for phase_index, phase in enumerate(phases):
            if not phase.visible:
                continue
            row = phase_index % 2
            label_rows[row].append(phase.name)
            line_width = 1.6 if selected_index == phase_index else 1.0
            row_base = tick_base - row * row_gap
            for center, _ in phase.peaks:
                tick_axis.plot(
                    [center, center],
                    [row_base, row_base + tick_height],
                    color=phase.color,
                    linewidth=line_width,
                )
        for row_index, names in enumerate(label_rows):
            if not names:
                continue
            row_base = tick_base - row_index * row_gap
            tick_axis.text(
                x_min + 0.01 * (x_max - x_min),
                row_base + tick_height * 0.6,
                " | ".join(names),
                color="#334155",
                fontsize=8,
                va="center",
                ha="left",
            )

        self.axes_main.legend(loc="upper right", fontsize=9, frameon=True)
        self.axes_main.set_ylabel("Intensity")
        self.axes_diff.set_xlabel("2theta")
        self.axes_diff.set_ylabel("Diff")
        self.axes_main.grid(True, color="#e2e8f0", linewidth=0.7)
        self.axes_diff.grid(True, color="#e2e8f0", linewidth=0.7)
        self.draw()


class FullPatternFittingWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Qt6 Full-Pattern Fitting (Rietveld / Pawley / Le Bail)")
        self.resize(1200, 800)

        self.data_x: Optional[np.ndarray] = None
        self.data_y: Optional[np.ndarray] = None
        self.phases: List[Phase] = []
        self.selected_phase_index: Optional[int] = None
        self.profile_params = ProfileParams(shape="Voigt", sigma=0.05, gamma=0.05, eta=0.5)
        self.zero_shift = 0.0
        self.background_points: List[Tuple[float, float]] = []
        self.background_pick_mode = False
        self._peaks_dirty = True
        self._last_view = None
        self._last_data_range = None

        self._table_updating = False
        self._dragging_phase: Optional[int] = None
        self._drag_anchor_x: Optional[float] = None
        self._drag_start_cell: Optional[CellParameters] = None
        self._last_drag_ms = 0
        self._pending_update = False
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setInterval(120)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._apply_scheduled_update)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(splitter)

        controls = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(controls)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(12)

        data_group = QtWidgets.QGroupBox("Data")
        data_layout = QtWidgets.QVBoxLayout(data_group)
        data_layout.setSpacing(6)
        self.data_label = QtWidgets.QLabel("No data loaded.")
        self.data_label.setWordWrap(True)
        self.load_data_button = QtWidgets.QPushButton("Load Data (.dat/.txt/.chi/.xy/.fxye)")
        data_layout.addWidget(self.load_data_button)
        data_layout.addWidget(self.data_label)

        phase_group = QtWidgets.QGroupBox("Phases (CIF / JCPDS)")
        phase_layout = QtWidgets.QVBoxLayout(phase_group)
        phase_layout.setSpacing(6)
        phase_buttons = QtWidgets.QHBoxLayout()
        phase_buttons.setSpacing(6)
        self.load_phase_button = QtWidgets.QPushButton("Load CIF/JCPDS")
        self.remove_phase_button = QtWidgets.QPushButton("Remove Selected")
        self.clear_phases_button = QtWidgets.QPushButton("Clear Phases")
        phase_buttons.addWidget(self.load_phase_button)
        phase_buttons.addWidget(self.remove_phase_button)
        phase_buttons.addWidget(self.clear_phases_button)

        self.phase_table = QtWidgets.QTableWidget(0, 9)
        self.phase_table.setHorizontalHeaderLabels(
            ["Phase", "a", "b", "c", "alpha", "beta", "gamma", "Scale", "Use"]
        )
        self.phase_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.phase_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.phase_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.phase_table.setAlternatingRowColors(True)
        self.phase_table.setMinimumHeight(140)
        self.phase_table.verticalHeader().setDefaultSectionSize(26)

        phase_layout.addLayout(phase_buttons)
        phase_layout.addWidget(self.phase_table)

        fit_group = QtWidgets.QGroupBox("Fit Settings")
        fit_layout = QtWidgets.QFormLayout(fit_group)
        fit_layout.setHorizontalSpacing(10)
        fit_layout.setVerticalSpacing(8)
        fit_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        fit_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["Rietveld", "Pawley", "Le Bail"])
        self.method_combo.setCurrentText("Le Bail")
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.addItems(["Voigt", "Gaussian", "Lorentzian", "Pseudo-Voigt"])

        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_spin.setRange(1e-4, 5.0)
        self.sigma_spin.setDecimals(4)
        self.sigma_spin.setValue(0.05)
        self.sigma_spin.setSingleStep(0.005)
        self.sigma_spin.setStepType(
            QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )

        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(1e-4, 5.0)
        self.gamma_spin.setDecimals(4)
        self.gamma_spin.setValue(0.05)
        self.gamma_spin.setSingleStep(0.005)
        self.gamma_spin.setStepType(
            QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )

        self.eta_spin = QtWidgets.QDoubleSpinBox()
        self.eta_spin.setRange(0.0, 1.0)
        self.eta_spin.setDecimals(3)
        self.eta_spin.setValue(0.5)
        self.eta_spin.setSingleStep(0.01)
        self.eta_spin.setStepType(
            QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )

        self.wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.wavelength_spin.setRange(0.1, 3.0)
        self.wavelength_spin.setDecimals(5)
        self.wavelength_spin.setValue(1.5406)
        self.wavelength_spin.setSingleStep(0.0005)
        self.wavelength_spin.setStepType(
            QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )

        self.zero_shift_spin = QtWidgets.QDoubleSpinBox()
        self.zero_shift_spin.setRange(-0.5, 0.5)
        self.zero_shift_spin.setDecimals(4)
        self.zero_shift_spin.setValue(0.0)
        self.zero_shift_spin.setSingleStep(0.001)
        self.zero_shift_spin.setStepType(
            QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )

        self.b0_spin = QtWidgets.QDoubleSpinBox()
        self.b0_spin.setRange(-1e6, 1e6)
        self.b0_spin.setDecimals(3)
        self.b0_spin.setValue(0.0)
        self.b0_spin.hide()
        self.b0_spin.setSingleStep(5.0)
        self.b0_spin.setStepType(
            QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )

        self.b1_spin = QtWidgets.QDoubleSpinBox()
        self.b1_spin.setRange(-1e4, 1e4)
        self.b1_spin.setDecimals(6)
        self.b1_spin.setValue(0.0)
        self.b1_spin.hide()
        self.b1_spin.setSingleStep(0.001)
        self.b1_spin.setStepType(
            QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )

        fit_layout.addRow("Method", self.method_combo)
        fit_layout.addRow("Profile", self.profile_combo)
        fit_layout.addRow("Sigma", self.sigma_spin)
        fit_layout.addRow("Gamma", self.gamma_spin)
        fit_layout.addRow("Eta", self.eta_spin)
        fit_layout.addRow("Wavelength (A)", self.wavelength_spin)
        fit_layout.addRow("Zero shift (deg)", self.zero_shift_spin)

        background_group = QtWidgets.QGroupBox("Background")
        background_layout = QtWidgets.QVBoxLayout(background_group)
        background_layout.setSpacing(6)
        bg_buttons = QtWidgets.QHBoxLayout()
        bg_buttons.setSpacing(6)
        self.bg_pick_button = QtWidgets.QPushButton("Pick Background")
        self.bg_pick_button.setCheckable(True)
        self.bg_clear_button = QtWidgets.QPushButton("Clear Background")
        bg_buttons.addWidget(self.bg_pick_button)
        bg_buttons.addWidget(self.bg_clear_button)
        background_layout.addLayout(bg_buttons)

        control_layout.addWidget(data_group)
        control_layout.addWidget(phase_group)
        control_layout.addWidget(fit_group)
        control_layout.addWidget(background_group)
        control_layout.addStretch(1)

        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        self.plot_canvas = FitPlotCanvas(plot_container)
        self.toolbar = NavigationToolbar(self.plot_canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.plot_canvas)
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)
        splitter.addWidget(plot_container)
        splitter.setStretchFactor(1, 1)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)

        self.setStyleSheet(
            """
            QWidget { background: #f8fafc; }
            QGroupBox {
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-top: 8px;
                padding: 6px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #0f172a;
                font-weight: 600;
            }
            QLabel { color: #0f172a; }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:checked { background-color: #0ea5e9; }
            QTableWidget {
                background: #ffffff;
                gridline-color: #e2e8f0;
                border: 1px solid #e2e8f0;
                alternate-background-color: #f1f5f9;
            }
            QHeaderView::section {
                background: #f1f5f9;
                color: #334155;
                padding: 4px;
                border: 1px solid #e2e8f0;
            }
            """
        )

    def _connect_signals(self) -> None:
        self.load_data_button.clicked.connect(self.load_data)
        self.load_phase_button.clicked.connect(self.load_phase)
        self.remove_phase_button.clicked.connect(self.remove_selected_phase)
        self.clear_phases_button.clicked.connect(self.clear_phases)
        self.phase_table.itemSelectionChanged.connect(self.on_phase_selection_changed)
        self.phase_table.cellChanged.connect(self.on_phase_cell_changed)
        self.method_combo.currentTextChanged.connect(self.schedule_update)
        self.profile_combo.currentTextChanged.connect(self.update_profile)
        self.sigma_spin.valueChanged.connect(self.update_profile)
        self.gamma_spin.valueChanged.connect(self.update_profile)
        self.eta_spin.valueChanged.connect(self.update_profile)
        self.wavelength_spin.valueChanged.connect(self.mark_peaks_dirty)
        self.wavelength_spin.valueChanged.connect(self.schedule_update)
        self.zero_shift_spin.valueChanged.connect(self.schedule_update)
        self.b0_spin.valueChanged.connect(self.schedule_update)
        self.b1_spin.valueChanged.connect(self.schedule_update)
        self.bg_pick_button.toggled.connect(self.toggle_background_pick)
        self.bg_clear_button.clicked.connect(self.clear_background_points)

        self.plot_canvas.mpl_connect("button_press_event", self.on_plot_press)
        self.plot_canvas.mpl_connect("motion_notify_event", self.on_plot_motion)
        self.plot_canvas.mpl_connect("button_release_event", self.on_plot_release)

    def set_status(self, message: str) -> None:
        self.status_bar.showMessage(message, 5000)

    def schedule_update(self) -> None:
        self._pending_update = True
        if not self._update_timer.isActive():
            self._update_timer.start()

    def mark_peaks_dirty(self) -> None:
        self._peaks_dirty = True

    def _flash_cursor(self) -> None:
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QtCore.QTimer.singleShot(120, QtWidgets.QApplication.restoreOverrideCursor)

    def _apply_scheduled_update(self) -> None:
        if not self._pending_update:
            return
        self._pending_update = False
        self.update_pattern()

    def toggle_background_pick(self, enabled: bool) -> None:
        self.background_pick_mode = enabled
        if enabled:
            self.plot_canvas.setCursor(Qt.CursorShape.CrossCursor)
            self.set_status("Background pick: left-click add, right-click remove.")
        else:
            self.plot_canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def clear_background_points(self) -> None:
        self.background_points.clear()
        self._update_background_controls()
        self.set_status("Background points cleared.")
        self.schedule_update()

    def _update_background_controls(self) -> None:
        manual = len(self.background_points) >= 2
        self.b0_spin.setEnabled(not manual)
        self.b1_spin.setEnabled(not manual)

    def _add_background_point(self, x_val: float, y_val: float) -> None:
        self.background_points.append((float(x_val), float(y_val)))
        self.background_points.sort(key=lambda item: item[0])
        self._update_background_controls()
        self.set_status(f"Background points: {len(self.background_points)}")
        self._flash_cursor()
        self.schedule_update()

    def _remove_background_point(self, x_val: float, y_val: float) -> None:
        if not self.background_points or self.data_x is None or self.data_y is None:
            return
        x_range = float(np.max(self.data_x) - np.min(self.data_x))
        y_range = float(np.max(self.data_y) - np.min(self.data_y))
        x_range = max(x_range, 1e-6)
        y_range = max(y_range, 1e-6)
        best_index = None
        best_score = float("inf")
        for idx, (px, py) in enumerate(self.background_points):
            score = ((px - x_val) / x_range) ** 2 + ((py - y_val) / y_range) ** 2
            if score < best_score:
                best_score = score
                best_index = idx
        if best_index is not None and best_score < 0.02**2:
            self.background_points.pop(best_index)
            self._update_background_controls()
            self.set_status(f"Background points: {len(self.background_points)}")
            self._flash_cursor()
            self.schedule_update()

    def load_data(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Data",
            "",
            "XRD Data (*.dat *.txt *.chi *.xy *.fxye);;All Files (*)",
        )
        if not file_path:
            return
        try:
            x, y = read_xy_data(file_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to read data:\n{exc}")
            return
        self.data_x, self.data_y = x, y
        self.data_label.setText(file_path)
        self.b0_spin.setValue(float(np.min(y)))
        self.b1_spin.setValue(0.0)
        self.background_points.clear()
        self._peaks_dirty = True
        self._last_view = None
        self._last_data_range = None
        self._update_background_controls()
        self.set_status("Data loaded.")
        self.refresh_phase_peaks()
        self.update_pattern()

    def load_phase(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load CIF/JCPDS",
            "",
            "Structure Files (*.cif *.CIF *.jcpds *.pdf *.txt);;All Files (*)",
        )
        if not file_path:
            return

        extension = os.path.splitext(file_path)[1].lower()
        phase_name = os.path.basename(file_path)
        color_cycle = ["#2563eb", "#ef4444", "#22c55e", "#a855f7", "#f97316"]
        color = color_cycle[len(self.phases) % len(color_cycle)]

        try:
            if extension == ".cif":
                wavelength = self.wavelength_spin.value()
                cell, hkl_list, fixed_peaks = parse_cif(file_path, wavelength=wavelength)
                phase = Phase(
                    name=phase_name,
                    source=file_path,
                    cell=cell,
                    hkl_list=hkl_list,
                    fixed_peaks=fixed_peaks,
                    color=color,
                )
            else:
                rows, wavelength = parse_jcpds(file_path)
                column_mode, ok = QtWidgets.QInputDialog.getItem(
                    self,
                    "JCPDS Column Type",
                    "Column 1 interpretation:",
                    ["Auto", "d-spacing", "2theta"],
                    0,
                    False,
                )
                if not ok:
                    return
                fixed_peaks = self.convert_jcpds_rows(rows, column_mode, wavelength)
                phase = Phase(
                    name=phase_name,
                    source=file_path,
                    cell=None,
                    fixed_peaks=fixed_peaks,
                    color=color,
                )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to parse phase:\n{exc}")
            return

        self.phases.append(phase)
        self._peaks_dirty = True
        self.update_phase_table()
        self.refresh_phase_peaks()
        self.update_pattern()
        self.set_status("Phase loaded.")

    def convert_jcpds_rows(
        self, rows: List[Tuple[float, float]], mode: str, wavelength: Optional[float]
    ) -> List[Tuple[float, float]]:
        if wavelength is None:
            wavelength = self.wavelength_spin.value()
        peaks: List[Tuple[float, float]] = []
        first_values = [row[0] for row in rows]
        auto_is_twotheta = max(first_values) > 10.0
        for value, intensity in rows:
            if mode == "2theta" or (mode == "Auto" and auto_is_twotheta):
                peaks.append((value, intensity))
            else:
                two_theta = two_theta_from_d(value, wavelength)
                if two_theta is not None:
                    peaks.append((two_theta, intensity))
        return peaks

    def remove_selected_phase(self) -> None:
        if self.selected_phase_index is None:
            return
        if 0 <= self.selected_phase_index < len(self.phases):
            self.phases.pop(self.selected_phase_index)
        self.selected_phase_index = None
        self._peaks_dirty = True
        self.update_phase_table()
        self.refresh_phase_peaks()
        self.update_pattern()

    def clear_phases(self) -> None:
        self.phases.clear()
        self.selected_phase_index = None
        self._peaks_dirty = True
        self.update_phase_table()
        self.update_pattern()

    def update_phase_table(self) -> None:
        self._table_updating = True
        self.phase_table.setRowCount(len(self.phases))
        for row, phase in enumerate(self.phases):
            name_item = QtWidgets.QTableWidgetItem(phase.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            name_item.setForeground(QtGui.QBrush(QtGui.QColor(phase.color)))
            self.phase_table.setItem(row, 0, name_item)

            def set_cell(column: int, value: Optional[float]) -> None:
                item = QtWidgets.QTableWidgetItem("N/A" if value is None else f"{value:.4f}")
                if value is None:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.phase_table.setItem(row, column, item)

            if phase.cell:
                set_cell(1, phase.cell.a)
                set_cell(2, phase.cell.b)
                set_cell(3, phase.cell.c)
                set_cell(4, phase.cell.alpha)
                set_cell(5, phase.cell.beta)
                set_cell(6, phase.cell.gamma)
            else:
                for col in range(1, 7):
                    set_cell(col, None)

            scale_item = QtWidgets.QTableWidgetItem(f"{phase.scale:.4f}")
            self.phase_table.setItem(row, 7, scale_item)

            use_item = QtWidgets.QTableWidgetItem("")
            use_item.setFlags(
                (use_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            use_item.setCheckState(
                Qt.CheckState.Checked if phase.visible else Qt.CheckState.Unchecked
            )
            self.phase_table.setItem(row, 8, use_item)

            tint = QtGui.QColor(phase.color)
            tint.setAlpha(40 if phase.visible else 0)
            for col in range(self.phase_table.columnCount()):
                item = self.phase_table.item(row, col)
                if item is None:
                    continue
                if phase.visible:
                    item.setBackground(QtGui.QBrush(tint))
                else:
                    item.setBackground(QtGui.QBrush())

        self._table_updating = False

    def on_phase_selection_changed(self) -> None:
        selected = self.phase_table.selectedItems()
        if not selected:
            self.selected_phase_index = None
        else:
            self.selected_phase_index = selected[0].row()
        self.schedule_update()

    def on_phase_cell_changed(self, row: int, column: int) -> None:
        if self._table_updating:
            return
        if row >= len(self.phases):
            return
        phase = self.phases[row]
        item = self.phase_table.item(row, column)
        if item is None:
            return

        if column == 8:
            phase.visible = item.checkState() == Qt.CheckState.Checked
            self.update_phase_table()
            self.schedule_update()
            return

        if column == 7:
            value = parse_numeric(item.text())
            if value is None:
                self.set_status("Invalid scale value.")
                self.update_phase_table()
                return
            phase.scale = max(0.0, float(value))
            self.schedule_update()
            return

        if phase.cell is None:
            return

        value = parse_numeric(item.text())
        if value is None:
            self.set_status("Invalid cell parameter.")
            self.update_phase_table()
            return

        if column == 1:
            phase.cell.a = float(value)
        elif column == 2:
            phase.cell.b = float(value)
        elif column == 3:
            phase.cell.c = float(value)
        elif column == 4:
            phase.cell.alpha = float(value)
        elif column == 5:
            phase.cell.beta = float(value)
        elif column == 6:
            phase.cell.gamma = float(value)
        self._peaks_dirty = True
        self.refresh_phase_peaks()
        self.schedule_update()

    def refresh_phase_peaks(self) -> None:
        if self.data_x is None or not self._peaks_dirty:
            return
        two_theta_min = float(np.min(self.data_x))
        two_theta_max = float(np.max(self.data_x))
        wavelength = self.wavelength_spin.value()
        for phase in self.phases:
            if phase.generated_hkl:
                phase.hkl_list = []
            update_phase_peaks(phase, two_theta_min, two_theta_max, wavelength)
        self._peaks_dirty = False

    def update_profile(self) -> None:
        shape = self.profile_combo.currentText()
        self.profile_params = ProfileParams(
            shape=shape,
            sigma=self.sigma_spin.value(),
            gamma=self.gamma_spin.value(),
            eta=self.eta_spin.value(),
        )
        self.schedule_update()

    def update_pattern(self) -> None:
        if self.data_x is None:
            self.plot_canvas.render(
                None, None, None, None, [], self.phases, self.selected_phase_index
            )
            return
        self._pending_update = False
        self._update_background_controls()
        preserve_view = False
        view_state = None
        current_range = (float(np.min(self.data_x)), float(np.max(self.data_x)))
        if (
            self.plot_canvas.current_x is not None
            and self._last_data_range == current_range
            and self.plot_canvas.axes_main.has_data()
        ):
            preserve_view = True
            view_state = (
                self.plot_canvas.axes_main.get_xlim(),
                self.plot_canvas.axes_main.get_ylim(),
                self.plot_canvas.axes_diff.get_ylim(),
            )
        self.profile_params = ProfileParams(
            shape=self.profile_combo.currentText(),
            sigma=self.sigma_spin.value(),
            gamma=self.gamma_spin.value(),
            eta=self.eta_spin.value(),
        )
        self.zero_shift = self.zero_shift_spin.value()
        self.refresh_phase_peaks()
        method = self.method_combo.currentText()
        if self.background_points:
            fallback = compute_background(self.data_x, self.b0_spin.value(), self.b1_spin.value())
        else:
            y0 = float(self.data_y[0])
            y1 = float(self.data_y[-1])
            x0 = float(self.data_x[0])
            x1 = float(self.data_x[-1])
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
            fallback = y0 + slope * (self.data_x - x0)
        background = interpolate_background(self.data_x, self.background_points, fallback)
        calc, bkg, _ = compute_pattern(
            self.data_x,
            self.data_y,
            self.phases,
            method,
            self.profile_params,
            background,
            self.zero_shift,
        )
        self.plot_canvas.render(
            self.data_x,
            self.data_y,
            calc,
            bkg,
            self.background_points,
            self.phases,
            self.selected_phase_index,
        )
        self._last_data_range = current_range
        if preserve_view and view_state is not None:
            self.plot_canvas.axes_main.set_xlim(view_state[0])
            self.plot_canvas.axes_main.set_ylim(view_state[1])
            self.plot_canvas.axes_diff.set_ylim(view_state[2])
            self.plot_canvas.draw_idle()
        self._last_view = (
            self.plot_canvas.axes_main.get_xlim(),
            self.plot_canvas.axes_main.get_ylim(),
            self.plot_canvas.axes_diff.get_ylim(),
        )

    def perform_fit(self) -> None:
        if self.data_x is None or self.data_y is None or not self.phases:
            return

        method = self.method_combo.currentText()
        self.profile_params = ProfileParams(
            shape=self.profile_combo.currentText(),
            sigma=self.sigma_spin.value(),
            gamma=self.gamma_spin.value(),
            eta=self.eta_spin.value(),
        )
        x = self.data_x
        y = self.data_y
        use_manual_background = len(self.background_points) >= 2

        param_map: List[Tuple[str, int, Optional[str]]] = []
        params: List[float] = []
        lower: List[float] = []
        upper: List[float] = []

        for phase_index, phase in enumerate(self.phases):
            if phase.cell:
                params.extend([phase.cell.a, phase.cell.b, phase.cell.c])
                lower.extend([phase.cell.a * 0.95, phase.cell.b * 0.95, phase.cell.c * 0.95])
                upper.extend([phase.cell.a * 1.05, phase.cell.b * 1.05, phase.cell.c * 1.05])
                param_map.extend([("cell", phase_index, "a"), ("cell", phase_index, "b"), ("cell", phase_index, "c")])
            if method == "Rietveld":
                params.append(phase.scale)
                lower.append(0.0)
                upper.append(max(phase.scale * 3.0, 1.0))
                param_map.append(("scale", phase_index, None))

        params.append(self.zero_shift_spin.value())
        lower.append(-0.5)
        upper.append(0.5)
        param_map.append(("zero_shift", -1, None))

        if not use_manual_background:
            params.append(self.b0_spin.value())
            lower.append(-1e6)
            upper.append(1e6)
            param_map.append(("b0", -1, None))

            params.append(self.b1_spin.value())
            lower.append(-1e4)
            upper.append(1e4)
            param_map.append(("b1", -1, None))

        def apply_params(vector: np.ndarray) -> Tuple[float, np.ndarray, float, float]:
            zero_shift = self.zero_shift_spin.value()
            b0 = self.b0_spin.value()
            b1 = self.b1_spin.value()
            for value, mapping in zip(vector, param_map):
                kind, phase_index, attr = mapping
                if kind == "cell" and attr and self.phases[phase_index].cell:
                    setattr(self.phases[phase_index].cell, attr, float(value))
                elif kind == "scale":
                    self.phases[phase_index].scale = float(value)
                elif kind == "zero_shift":
                    zero_shift = float(value)
                elif kind == "b0":
                    b0 = float(value)
                elif kind == "b1":
                    b1 = float(value)
            fallback = compute_background(x, b0, b1)
            background = interpolate_background(x, self.background_points, fallback)
            return zero_shift, background, b0, b1

        def residuals(vector: np.ndarray) -> np.ndarray:
            zero_shift, background, _, _ = apply_params(vector)
            self.refresh_phase_peaks()
            calc, _, _ = compute_pattern(
                x,
                y,
                self.phases,
                method,
                self.profile_params,
                background,
                zero_shift,
            )
            return calc - y

        self.set_status("Fitting... please wait.")
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = least_squares(
                residuals,
                np.array(params, dtype=float),
                bounds=(np.array(lower, dtype=float), np.array(upper, dtype=float)),
                max_nfev=20,
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        _, _, b0_value, b1_value = apply_params(result.x)
        if ("zero_shift", -1, None) in param_map:
            self.zero_shift_spin.setValue(float(result.x[param_map.index(("zero_shift", -1, None))]))
        if ("b0", -1, None) in param_map:
            self.b0_spin.setValue(float(result.x[param_map.index(("b0", -1, None))]))
        else:
            self.b0_spin.setValue(b0_value)
        if ("b1", -1, None) in param_map:
            self.b1_spin.setValue(float(result.x[param_map.index(("b1", -1, None))]))
        else:
            self.b1_spin.setValue(b1_value)
        self.update_phase_table()
        self.refresh_phase_peaks()
        self._update_background_controls()
        self.update_pattern()
        self.set_status("Fit complete.")

    def save_plot(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Image (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not file_path:
            return
        self.plot_canvas.figure.savefig(file_path, dpi=300)
        self.set_status(f"Plot saved to {file_path}")

    def on_plot_press(self, event) -> None:
        if event.inaxes != self.plot_canvas.axes_main:
            return
        if self.background_pick_mode:
            if event.xdata is None or event.ydata is None:
                return
            if event.button == 1:
                self._add_background_point(event.xdata, event.ydata)
            elif event.button == 3:
                self._remove_background_point(event.xdata, event.ydata)
            return
        if self.selected_phase_index is None and len(self.phases) == 1:
            self.phase_table.selectRow(0)
            self.selected_phase_index = 0
        if self.selected_phase_index is None:
            self.set_status("Select a phase to drag its lattice parameters.")
            return
        if not (0 <= self.selected_phase_index < len(self.phases)):
            return
        phase = self.phases[self.selected_phase_index]
        if phase.cell is None:
            self.set_status("Selected phase has no lattice parameters to adjust.")
            return
        if event.xdata is None:
            return
        if event.button != 1:
            return
        if not self.plot_canvas.is_over_calc_line(event):
            return
        self._dragging_phase = self.selected_phase_index
        self._drag_anchor_x = float(event.xdata)
        self._drag_start_cell = CellParameters(*phase.cell.as_tuple())

    def on_plot_motion(self, event) -> None:
        if self._dragging_phase is None or self._drag_anchor_x is None:
            return
        if event.inaxes != self.plot_canvas.axes_main or event.xdata is None:
            return
        current_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        if current_time - self._last_drag_ms < 30:
            return
        self._last_drag_ms = current_time
        delta = float(event.xdata) - self._drag_anchor_x
        if abs(delta) < 1e-6:
            return
        phase = self.phases[self._dragging_phase]
        if phase.cell is None or self._drag_start_cell is None:
            return
        anchor_theta = math.radians(self._drag_anchor_x / 2.0)
        shifted_theta = math.radians((self._drag_anchor_x + delta) / 2.0)
        if math.sin(shifted_theta) <= 0:
            return
        scale = math.sin(anchor_theta) / math.sin(shifted_theta)
        phase.cell.a = self._drag_start_cell.a * scale
        phase.cell.b = self._drag_start_cell.b * scale
        phase.cell.c = self._drag_start_cell.c * scale
        self.update_phase_table()
        self.refresh_phase_peaks()
        self.schedule_update()

    def on_plot_release(self, event) -> None:
        self._dragging_phase = None
        self._drag_anchor_x = None
        self._drag_start_cell = None
        self.schedule_update()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = FullPatternFittingWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
