# -*- coding: utf-8 -*-
"""
Complete XRD Batch Processing Suite - Fully Integrated Version
Integrates all functions from three files
Created on Mon Nov 10 15:11:31 2025

Features:
1. Batch Integration (HDF5 → .xy)
2. Peak Fitting (Pseudo-Voigt/Voigt)
3. Full Pipeline (Integration + Fitting)
4. Phase Transition Detection & Analysis
5. Lattice Parameter Fitting (8 crystal systems)
6. Birch-Murnaghan EOS Fitting (2nd & 3rd order)

@author: 16961
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
from batch_integration import BatchIntegrator
from peak_fitting import BatchFitter
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, curve_fit
import re
import warnings
import math
import random

warnings.filterwarnings('ignore')


# ==================== XRayDiffractionAnalyzer Class ====================
class XRayDiffractionAnalyzer:
    """
    X-ray Diffraction Analysis Tool for phase transition identification and lattice parameter fitting.
    Fully preserves all features from the second file
    """

    CRYSTAL_SYSTEMS = {
        'cubic_FCC': {
            'name': 'FCC',
            'min_peaks': 1,
            'atoms_per_cell': 4,
            'hkl_list': [
                (1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2),
                (4,0,0), (3,3,1), (4,2,0), (4,2,2), (3,3,3),
                (5,1,1), (4,4,0), (5,3,1), (6,0,0), (6,2,0),
                (5,3,3), (6,2,2), (4,4,4), (5,5,1), (6,4,0)
            ]
        },
        'cubic_BCC': {
            'name': 'BCC',
            'min_peaks': 1,
            'atoms_per_cell': 2,
            'hkl_list': [
                (1,1,0), (2,0,0), (2,1,1), (2,2,0), (3,1,0),
                (2,2,2), (3,2,1), (4,0,0), (3,3,0), (4,1,1),
                (3,3,2), (4,2,0), (4,2,2), (3,3,3), (5,1,0),
                (4,3,1), (5,2,1), (4,4,0), (5,3,0), (6,0,0)
            ]
        },
        'cubic_SC': {
            'name': 'SC',
            'min_peaks': 1,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0),
                (2,1,1), (2,2,0), (2,2,1), (3,0,0), (3,1,0),
                (3,1,1), (2,2,2), (3,2,0), (3,2,1), (4,0,0),
                (4,1,0), (3,3,0), (4,1,1), (3,3,1), (4,2,0)
            ]
        },
        'Hexagonal': {
            'name': 'HCP',
            'min_peaks': 2,
            'atoms_per_cell': 2,
            'hkl_list': [
                (1,0,0), (1,0,1), (1,0,2), (1,1,0),
                (1,0,3), (2,0,0), (1,1,2), (2,0,1), (0,0,4),
                (2,0,2), (1,0,4), (2,0,3), (2,1,0), (2,1,1),
                (2,0,4), (2,1,2), (3,0,0), (2,1,3), (2,2,0)
            ]
        },
        'Tetragonal': {
            'name': 'Tetragonal',
            'min_peaks': 2,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (0,0,1), (1,1,0), (1,0,1), (1,1,1),
                (2,0,0), (2,1,0), (0,0,2), (2,1,1), (2,0,1),
                (2,2,0), (2,1,2), (3,0,0), (2,2,1), (3,1,0),
                (2,0,2), (3,1,1), (2,2,2), (3,2,0), (3,0,1)
            ]
        },
        'Orthorhombic': {
            'name': 'Orthorhombic',
            'min_peaks': 3,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
                (0,1,1), (1,1,1), (2,0,0), (2,1,0), (2,0,1),
                (1,2,0), (0,2,0), (1,2,1), (0,2,1), (2,1,1),
                (2,2,0), (2,0,2), (0,0,2), (2,2,1), (3,0,0)
            ]
        },
        'Monoclinic': {
            'name': 'Monoclinic',
            'min_peaks': 4,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
                (0,1,1), (1,-1,0), (1,0,-1), (1,1,1), (2,0,0),
                (1,-1,1), (2,1,0), (0,2,0), (2,0,1), (1,2,0),
                (0,0,2), (2,1,1), (1,1,-1), (2,-1,0), (2,0,-1)
            ]
        },
        'Triclinic': {
            'name': 'Triclinic',
            'min_peaks': 6,
            'atoms_per_cell': 1,
            'hkl_list': [
                (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
                (0,1,1), (1,-1,0), (1,0,-1), (0,1,-1), (1,1,1),
                (1,-1,1), (1,1,-1), (2,0,0), (0,2,0), (0,0,2),
                (2,1,0), (2,0,1), (1,2,0), (0,2,1), (1,0,2)
            ]
        }
    }

    def __init__(self, wavelength=0.4133, peak_tolerance_1=0.3,
                 peak_tolerance_2=0.4, peak_tolerance_3=0.01,
                 n_pressure_points=4):
        self.wavelength = wavelength
        self.peak_tolerance_1 = peak_tolerance_1
        self.peak_tolerance_2 = peak_tolerance_2
        self.peak_tolerance_3 = peak_tolerance_3
        self.n_pressure_points = n_pressure_points

        self.pressure_data = None
        self.transition_pressure = None
        self.before_pressures = []
        self.after_pressures = []
        self.original_peak_dataset = None
        self.tracked_new_peaks = None
        self.original_results = None
        self.new_results = None

    @staticmethod
    def two_theta_to_d(two_theta, wavelength):
        theta_rad = np.deg2rad(two_theta / 2.0)
        return wavelength / (2.0 * np.sin(theta_rad))

    @staticmethod
    def d_to_two_theta(d, wavelength):
        sin_theta = wavelength / (2.0 * d)
        if sin_theta > 1.0 or sin_theta < -1.0:
            return None
        theta_rad = np.arcsin(sin_theta)
        return np.rad2deg(2.0 * theta_rad)

    @staticmethod
    def calculate_d_cubic(hkl, a):
        h, k, l = hkl
        return a / np.sqrt(h**2 + k**2 + l**2)

    @staticmethod
    def calculate_d_hexagonal(hkl, a, c):
        h, k, l = hkl
        return 1.0 / np.sqrt(4.0/3.0 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)

    @staticmethod
    def calculate_d_tetragonal(hkl, a, c):
        h, k, l = hkl
        return 1.0 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)

    @staticmethod
    def calculate_d_orthorhombic(hkl, a, b, c):
        h, k, l = hkl
        return 1.0 / np.sqrt(h**2 / a**2 + k**2 / b**2 + l**2 / c**2)

    @staticmethod
    def calculate_d_monoclinic(hkl, a, b, c, beta):
        h, k, l = hkl
        beta_rad = np.deg2rad(beta)
        sin_beta = np.sin(beta_rad)
        cos_beta = np.cos(beta_rad)
        term = (h**2 / a**2 + k**2 * sin_beta**2 / b**2 + l**2 / c**2
                - 2*h*l*cos_beta / (a*c)) / sin_beta**2
        return 1.0 / np.sqrt(term)

    @staticmethod
    def calculate_cell_volume_cubic(a):
        return a**3

    @staticmethod
    def calculate_cell_volume_hexagonal(a, c):
        return np.sqrt(3) / 2 * a**2 * c

    @staticmethod
    def calculate_cell_volume_tetragonal(a, c):
        return a**2 * c

    @staticmethod
    def calculate_cell_volume_orthorhombic(a, b, c):
        return a * b * c

    @staticmethod
    def calculate_cell_volume_monoclinic(a, b, c, beta):
        beta_rad = np.deg2rad(beta)
        return a * b * c * np.sin(beta_rad)

    def read_pressure_peak_data(self, csv_path):
        df = pd.read_csv(csv_path)

        if 'File' not in df.columns or 'Center' not in df.columns:
            raise ValueError("CSV file must contain 'File' and 'Center' columns")

        pressure_data = {}

        for idx, row in df.iterrows():
            if pd.isna(row['File']) or row['File'] == '':
                continue

            try:
                file_str = str(row['File'])
                numbers = re.findall(r'[-+]?\d*\.?\d+', file_str)
                if numbers:
                    pressure = float(numbers[0])
                else:
                    pressure = float(file_str)
            except:
                continue

            try:
                peak_position = float(row['Center'])
            except:
                continue

            if pressure not in pressure_data:
                pressure_data[pressure] = []
            pressure_data[pressure].append(peak_position)

        for pressure in pressure_data:
            pressure_data[pressure] = sorted(pressure_data[pressure])

        self.pressure_data = pressure_data
        return pressure_data

    def find_phase_transition_point(self, pressure_data=None, tolerance=None):
        if pressure_data is None:
            pressure_data = self.pressure_data
        if tolerance is None:
            tolerance = self.peak_tolerance_1

        sorted_pressures = sorted(pressure_data.keys())

        if len(sorted_pressures) < 2:
            return None, sorted_pressures, []

        for i in range(1, len(sorted_pressures)):
            prev_pressure = sorted_pressures[i - 1]
            curr_pressure = sorted_pressures[i]

            prev_peaks = pressure_data[prev_pressure]
            curr_peaks = pressure_data[curr_pressure]

            tolerance_windows = [(p - tolerance, p + tolerance) for p in prev_peaks]

            has_new_peak = False
            for peak in curr_peaks:
                in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
                if not in_any_window:
                    has_new_peak = True
                    break

            if has_new_peak:
                self.transition_pressure = curr_pressure
                self.before_pressures = sorted_pressures[:i]
                self.after_pressures = sorted_pressures[i:]
                return curr_pressure, self.before_pressures, self.after_pressures

        self.transition_pressure = None
        self.before_pressures = sorted_pressures
        self.after_pressures = []
        return None, sorted_pressures, []

    def collect_tracked_new_peaks(self, pressure_data=None, transition_pressure=None,
                                   after_pressures=None, new_peaks_ref=None, tolerance=None):
        if pressure_data is None:
            pressure_data = self.pressure_data
        if tolerance is None:
            tolerance = self.peak_tolerance_2

        tracked_peaks_dict = {}
        peak_occurrences = {peak: 0 for peak in new_peaks_ref}

        for pressure in after_pressures:
            current_peaks = pressure_data[pressure]
            matched_peaks = []

            for new_peak in new_peaks_ref:
                lower = new_peak - tolerance
                upper = new_peak + tolerance

                matches = [p for p in current_peaks if lower <= p <= upper]
                if matches:
                    mean_match = np.mean(matches)
                    matched_peaks.append(mean_match)
                    peak_occurrences[new_peak] += 1

            if matched_peaks:
                tracked_peaks_dict[pressure] = matched_peaks

        stable_count = sum(1 for count in peak_occurrences.values()
                          if count >= self.n_pressure_points)

        self.tracked_new_peaks = tracked_peaks_dict
        return stable_count, tracked_peaks_dict

    def build_original_peak_dataset(self, pressure_data=None, tracked_new_peak_dataset=None,
                                     tolerance=None):
        if pressure_data is None:
            pressure_data = self.pressure_data
        if tracked_new_peak_dataset is None:
            tracked_new_peak_dataset = self.tracked_new_peaks
        if tolerance is None:
            tolerance = self.peak_tolerance_3

        original_peak_dataset = {}

        for pressure, all_peaks in pressure_data.items():
            new_peaks = tracked_new_peak_dataset.get(pressure, [])
            original_peaks = []

            for peak in all_peaks:
                is_new = any(abs(peak - new_peak) <= tolerance for new_peak in new_peaks)
                if not is_new:
                    original_peaks.append(peak)

            original_peak_dataset[pressure] = {
                'original_peaks': original_peaks,
                'count': len(original_peaks)
            }

        self.original_peak_dataset = original_peak_dataset
        return original_peak_dataset

    def fit_lattice_parameters_cubic(self, peak_dataset, crystal_system_key):
        results = {}
        hkl_list = self.CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
        atoms_per_cell = self.CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

        for pressure, data in peak_dataset.items():
            if isinstance(data, dict):
                peaks = data.get('original_peaks', data.get('new_peaks', []))
            else:
                peaks = data

            if len(peaks) == 0:
                continue

            d_obs = [self.two_theta_to_d(peak, self.wavelength) for peak in peaks]

            num_peaks = min(len(peaks), len(hkl_list))
            matched_hkl = hkl_list[:num_peaks]

            def residuals(params):
                a = params[0]
                errors = []
                for i, hkl in enumerate(matched_hkl):
                    d_calc = self.calculate_d_cubic(hkl, a)
                    errors.append(d_obs[i] - d_calc)
                return errors

            a_init = d_obs[0] * np.sqrt(sum(x**2 for x in matched_hkl[0]))
            result = least_squares(residuals, [a_init], bounds=([0], [np.inf]))
            a_fitted = result.x[0]

            V_cell = self.calculate_cell_volume_cubic(a_fitted)
            V_atomic = V_cell / atoms_per_cell

            results[pressure] = {
                'a': a_fitted,
                'V_cell': V_cell,
                'V_atomic': V_atomic,
                'num_peaks_used': num_peaks
            }

        return results

    def fit_lattice_parameters_hexagonal(self, peak_dataset, crystal_system_key):
        results = {}
        hkl_list = self.CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
        atoms_per_cell = self.CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

        for pressure, data in peak_dataset.items():
            if isinstance(data, dict):
                peaks = data.get('original_peaks', data.get('new_peaks', []))
            else:
                peaks = data

            if len(peaks) < 2:
                continue

            d_obs = [self.two_theta_to_d(peak, self.wavelength) for peak in peaks]

            num_peaks = min(len(peaks), len(hkl_list))
            matched_hkl = hkl_list[:num_peaks]

            def residuals(params):
                a, c = params
                errors = []
                for i, hkl in enumerate(matched_hkl):
                    d_calc = self.calculate_d_hexagonal(hkl, a, c)
                    errors.append(d_obs[i] - d_calc)

                target_ratio = 1.633
                ratio = c / a
                penalty_weight = 0.1
                penalty = penalty_weight * (ratio - target_ratio)
                errors.append(penalty)

                return errors

            a_init = 3.0
            c_init = 5.0

            result = least_squares(residuals, [a_init, c_init],
                                  bounds=([0, 0], [np.inf, np.inf]))
            a_fitted, c_fitted = result.x

            V_cell = self.calculate_cell_volume_hexagonal(a_fitted, c_fitted)
            V_atomic = V_cell / atoms_per_cell

            results[pressure] = {
                'a': a_fitted,
                'c': c_fitted,
                'c/a': c_fitted / a_fitted,
                'V_cell': V_cell,
                'V_atomic': V_atomic,
                'num_peaks_used': num_peaks
            }

        return results

    def fit_lattice_parameters_tetragonal(self, peak_dataset, crystal_system_key):
        results = {}
        hkl_list = self.CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
        atoms_per_cell = self.CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

        for pressure, data in peak_dataset.items():
            if isinstance(data, dict):
                peaks = data.get('original_peaks', data.get('new_peaks', []))
            else:
                peaks = data

            if len(peaks) < 2:
                continue

            d_obs = [self.two_theta_to_d(peak, self.wavelength) for peak in peaks]

            num_peaks = min(len(peaks), len(hkl_list))
            matched_hkl = hkl_list[:num_peaks]

            def residuals(params):
                a, c = params
                errors = []
                for i, hkl in enumerate(matched_hkl):
                    d_calc = self.calculate_d_tetragonal(hkl, a, c)
                    errors.append(d_obs[i] - d_calc)
                return errors

            a_init = 3.0
            c_init = 4.0

            result = least_squares(residuals, [a_init, c_init],
                                  bounds=([0, 0], [np.inf, np.inf]))
            a_fitted, c_fitted = result.x

            V_cell = self.calculate_cell_volume_tetragonal(a_fitted, c_fitted)
            V_atomic = V_cell / atoms_per_cell

            results[pressure] = {
                'a': a_fitted,
                'c': c_fitted,
                'c/a': c_fitted / a_fitted,
                'V_cell': V_cell,
                'V_atomic': V_atomic,
                'num_peaks_used': num_peaks
            }

        return results

    def fit_lattice_parameters_orthorhombic(self, peak_dataset, crystal_system_key):
        results = {}
        hkl_list = self.CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
        atoms_per_cell = self.CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

        for pressure, data in peak_dataset.items():
            if isinstance(data, dict):
                peaks = data.get('original_peaks', data.get('new_peaks', []))
            else:
                peaks = data

            if len(peaks) < 3:
                continue

            d_obs = [self.two_theta_to_d(peak, self.wavelength) for peak in peaks]

            num_peaks = min(len(peaks), len(hkl_list))
            matched_hkl = hkl_list[:num_peaks]

            def residuals(params):
                a, b, c = params
                errors = []
                for i, hkl in enumerate(matched_hkl):
                    d_calc = self.calculate_d_orthorhombic(hkl, a, b, c)
                    errors.append(d_obs[i] - d_calc)
                return errors

            a_init = 3.0
            b_init = 4.0
            c_init = 5.0

            result = least_squares(residuals, [a_init, b_init, c_init],
                                  bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            a_fitted, b_fitted, c_fitted = result.x

            V_cell = self.calculate_cell_volume_orthorhombic(a_fitted, b_fitted, c_fitted)
            V_atomic = V_cell / atoms_per_cell

            results[pressure] = {
                'a': a_fitted,
                'b': b_fitted,
                'c': c_fitted,
                'V_cell': V_cell,
                'V_atomic': V_atomic,
                'num_peaks_used': num_peaks
            }

        return results

    def fit_lattice_parameters(self, peak_dataset, crystal_system_key):
        system_type = crystal_system_key.split('_')[0]

        if system_type == 'cubic':
            return self.fit_lattice_parameters_cubic(peak_dataset, crystal_system_key)
        elif crystal_system_key == 'Hexagonal':
            return self.fit_lattice_parameters_hexagonal(peak_dataset, crystal_system_key)
        elif crystal_system_key == 'Tetragonal':
            return self.fit_lattice_parameters_tetragonal(peak_dataset, crystal_system_key)
        elif crystal_system_key == 'Orthorhombic':
            return self.fit_lattice_parameters_orthorhombic(peak_dataset, crystal_system_key)
        else:
            return {}

    @staticmethod
    def save_lattice_results_to_csv(results, filename, crystal_system_key):
        if not results:
            return

        data_rows = []
        for pressure, params in sorted(results.items()):
            row = {'Pressure (GPa)': pressure}
            row.update(params)
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)

    def analyze(self, csv_path, original_system='cubic_FCC', new_system='cubic_FCC',
                auto_mode=True, log_callback=None):
        def log(msg):
            if log_callback:
                log_callback(msg)

        log("="*60)
        log("Phase Transition Analysis")
        log("="*60)

        try:
            self.read_pressure_peak_data(csv_path)
            log(f"\n✓ Data loaded successfully")
            log(f"  Pressure points: {len(self.pressure_data)}")
        except Exception as e:
            log(f"❌ Error: {e}")
            return None

        log("\n" + "="*60)
        log("Phase Transition Detection")
        log("="*60)

        transition_pressure, before_pressures, after_pressures = self.find_phase_transition_point()

        if transition_pressure is None:
            log("\nNo phase transition detected, single phase analysis...")

            all_data_dict = {p: peaks for p, peaks in self.pressure_data.items()}
            results = self.fit_lattice_parameters(all_data_dict, original_system)

            output_filename = csv_path.replace('.csv', '_lattice_results.csv')
            self.save_lattice_results_to_csv(results, output_filename, original_system)

            return {'single_phase_results': results}

        log(f"\n>>> Phase transition detected at: {transition_pressure:.2f} GPa")

        log("\n" + "="*60)
        log("New Peaks Collection")
        log("="*60)

        transition_peaks = self.pressure_data[transition_pressure]
        prev_pressure = before_pressures[-1]
        prev_peaks = self.pressure_data[prev_pressure]

        tolerance_windows = [(p - self.peak_tolerance_1, p + self.peak_tolerance_1)
                            for p in prev_peaks]
        new_peaks_at_transition = []

        for peak in transition_peaks:
            in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
            if not in_any_window:
                new_peaks_at_transition.append(peak)

        log(f"\nNew peaks count: {len(new_peaks_at_transition)}")

        stable_count, tracked_new_peaks = self.collect_tracked_new_peaks(
            self.pressure_data, transition_pressure, after_pressures,
            new_peaks_at_transition, self.peak_tolerance_2
        )

        log(f"Stable new peaks: {stable_count}")

        original_peak_dataset = self.build_original_peak_dataset(
            self.pressure_data, tracked_new_peaks, self.peak_tolerance_3
        )

        log(f"Original peaks dataset: {len(original_peak_dataset)} points")

        log("\n" + "="*60)
        log("Lattice Parameter Fitting")
        log("="*60)

        log(f"\n>>> Original peaks system: {self.CRYSTAL_SYSTEMS[original_system]['name']}")
        self.original_results = self.fit_lattice_parameters(original_peak_dataset, original_system)

        log(f"\n>>> New peaks system: {self.CRYSTAL_SYSTEMS[new_system]['name']}")
        self.new_results = self.fit_lattice_parameters(tracked_new_peaks, new_system)

        base_filename = csv_path.replace('.csv', '')

        original_output = f"{base_filename}_original_peaks_lattice.csv"
        self.save_lattice_results_to_csv(self.original_results, original_output, original_system)

        new_output = f"{base_filename}_new_peaks_lattice.csv"
        self.save_lattice_results_to_csv(self.new_results, new_output, new_system)

        log("\n" + "="*60)
        log("Analysis Complete")
        log("="*60)
        log(f"\nTransition pressure: {transition_pressure:.2f} GPa")
        log(f"Original peaks results: {original_output}")
        log(f"New peaks results: {new_output}")

        return {
            'original_results': self.original_results,
            'new_results': self.new_results,
            'transition_pressure': transition_pressure
        }


# ==================== Birch-Murnaghan Fitter Class ====================
class BirchMurnaghanFitter:
    """
    Birch-Murnaghan Equation of State Fitter
    Fully preserves features from the third file
    """

    def __init__(self, data_file, output_dir, order=3):
        self.data_file = data_file
        self.output_dir = output_dir
        self.order = order

        self.data = pd.read_csv(data_file)
        os.makedirs(output_dir, exist_ok=True)

    def birch_murnaghan_2nd(self, V, V0, K0):
        """2nd order BM equation (K0' = 4)"""
        x = (V0 / V)**(2/3)
        return (3/2) * K0 * (x - 1)

    def birch_murnaghan_3rd(self, V, V0, K0, K0_prime):
        """3rd order BM equation"""
        x = (V0 / V)**(2/3)
        return (3/2) * K0 * (x - 1) * (1 + (3/4) * (K0_prime - 4) * (x - 1))

    def fit(self, initial_guess=None):
        P = self.data['Pressure (GPa)'].values
        V = self.data['V_atomic'].values

        if self.order == 2:
            if initial_guess is None or len(initial_guess) != 2:
                initial_guess = [V[0], 180.0]

            popt, pcov = curve_fit(self.birch_murnaghan_2nd, V, P, p0=initial_guess)
            V0, K0 = popt
            K0_prime = 4.0

            perr = np.sqrt(np.diag(pcov))
            V0_err, K0_err = perr
            K0_prime_err = 0.0

            P_fit = self.birch_murnaghan_2nd(V, V0, K0)

        else:
            if initial_guess is None or len(initial_guess) != 3:
                initial_guess = [V[0], 180.0, 4.0]

            popt, pcov = curve_fit(self.birch_murnaghan_3rd, V, P, p0=initial_guess)
            V0, K0, K0_prime = popt

            perr = np.sqrt(np.diag(pcov))
            V0_err, K0_err, K0_prime_err = perr

            P_fit = self.birch_murnaghan_3rd(V, V0, K0, K0_prime)

        ss_res = np.sum((P - P_fit)**2)
        ss_tot = np.sum((P - np.mean(P))**2)
        r_squared = 1 - (ss_res / ss_tot)

        results = {
            'V0': V0,
            'V0_err': V0_err,
            'K0': K0,
            'K0_err': K0_err,
            'K0_prime': K0_prime,
            'K0_prime_err': K0_prime_err,
            'r_squared': r_squared,
            'order': self.order
        }

        results_df = pd.DataFrame([results])
        results_file = os.path.join(self.output_dir, 'bm_fit_results.csv')
        results_df.to_csv(results_file, index=False)

        fitted_data = pd.DataFrame({
            'Pressure (GPa)': P,
            'V_atomic': V,
            'P_fit': P_fit
        })
        fitted_file = os.path.join(self.output_dir, 'bm_fitted_data.csv')
        fitted_data.to_csv(fitted_file, index=False)

        V_plot = np.linspace(V.min() * 0.9, V.max() * 1.1, 100)
        if self.order == 2:
            P_plot = self.birch_murnaghan_2nd(V_plot, V0, K0)
        else:
            P_plot = self.birch_murnaghan_3rd(V_plot, V0, K0, K0_prime)

        plot_data = pd.DataFrame({
            'V_plot': V_plot,
            'P_plot': P_plot
        })
        plot_file = os.path.join(self.output_dir, 'bm_plot_data.csv')
        plot_data.to_csv(plot_file, index=False)

        return results


# ==================== GUI Components ====================
class ModernButton(tk.Canvas):
    """Modern button component"""
    def __init__(self, parent, text, command, icon="", bg_color="#9D4EDD",
                 hover_color="#C77DFF", text_color="white", width=200, height=40, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'],
                        highlightthickness=0, **kwargs)

        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color

        self.rect = self.create_rounded_rectangle(0, 0, width, height, radius=10,
                                                   fill=bg_color, outline="")

        display_text = f"{icon}  {text}" if icon else text
        self.text_id = self.create_text(width//2, height//2, text=display_text,
                                       fill=text_color, font=('Comic Sans MS', 11, 'bold'))

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.config(cursor="hand2")

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1, x2, y1,
                  x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius, x2, y2,
                  x2-radius, y2, x2-radius, y2, x1+radius, y2, x1+radius, y2, x1, y2,
                  x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_enter(self, e):
        self.itemconfig(self.rect, fill=self.hover_color)

    def on_leave(self, e):
        self.itemconfig(self.rect, fill=self.bg_color)

    def on_click(self, e):
        if self.command:
            self.command()


class ModernTab(tk.Frame):
    """Modern tab component"""
    def __init__(self, parent, text, command, is_active=False, **kwargs):
        super().__init__(parent, bg=parent['bg'], **kwargs)
        self.command = command
        self.is_active = is_active
        self.parent_widget = parent

        self.active_color = "#9D4EDD"
        self.inactive_color = "#8B8BA7"
        self.hover_color = "#C77DFF"

        self.label = tk.Label(self, text=text,
                             fg=self.active_color if is_active else self.inactive_color,
                             bg=parent['bg'], font=('Comic Sans MS', 11, 'bold'),
                             cursor="hand2", padx=20, pady=10)
        self.label.pack()

        self.underline = tk.Frame(self, bg=self.active_color if is_active else parent['bg'],
                                 height=3)
        self.underline.pack(fill=tk.X)

        self.label.bind("<Enter>", self.on_enter)
        self.label.bind("<Leave>", self.on_leave)
        self.label.bind("<Button-1>", self.on_click)

    def on_enter(self, e):
        if not self.is_active:
            self.label.config(fg=self.hover_color)

    def on_leave(self, e):
        if not self.is_active:
            self.label.config(fg=self.inactive_color)

    def on_click(self, e):
        if self.command:
            self.command()

    def set_active(self, active):
        self.is_active = active
        self.label.config(fg=self.active_color if active else self.inactive_color)
        self.underline.config(bg=self.active_color if active else self.parent_widget['bg'])


class CuteSheepProgressBar(tk.Canvas):
    """Cute sheep progress bar animation"""
    def __init__(self, parent, width=700, height=80, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'],
                        highlightthickness=0, **kwargs)

        self.width = width
        self.height = height
        self.sheep = []
        self.is_animating = False
        self.frame_count = 0

    def draw_adorable_sheep(self, x, y, jump_phase):
        jump = -abs(math.sin(jump_phase) * 20)
        y = y + jump

        # Shadow
        self.create_oval(x-15, y+25, x+15, y+28, fill="#E8E4F3", outline="")
        # Body
        self.create_oval(x-20, y-15, x+20, y+15, fill="#FFFFFF", outline="#FFB6D9", width=3)
        self.create_oval(x-18, y-10, x-10, y-2, fill="#FFF5FF", outline="")
        self.create_oval(x+10, y-8, x+18, y, fill="#FFF5FF", outline="")
        self.create_oval(x-5, y+8, x+5, y+15, fill="#FFF5FF", outline="")
        # Head
        self.create_oval(x+15, y-12, x+35, y+8, fill="#FFE4F0", outline="#FFB6D9", width=3)
        # Ears
        self.create_polygon(x+17, y-10, x+20, y-18, x+23, y-10,
                           fill="#FFB6D9", outline="#FF6B9D", width=2, smooth=True)
        self.create_polygon(x+27, y-10, x+30, y-18, x+33, y-10,
                           fill="#FFB6D9", outline="#FF6B9D", width=2, smooth=True)
        # Eyes
        self.create_oval(x+19, y-6, x+24, y-1, fill="#FFFFFF")
        self.create_oval(x+20, y-5, x+23, y-2, fill="#2B2D42")
        self.create_oval(x+21, y-4, x+22, y-3, fill="#FFFFFF")
        self.create_oval(x+26, y-6, x+31, y-1, fill="#FFFFFF")
        self.create_oval(x+27, y-5, x+30, y-2, fill="#2B2D42")
        self.create_oval(x+28, y-4, x+29, y-3, fill="#FFFFFF")
        # Nose and mouth
        self.create_oval(x+23, y+2, x+27, y+6, fill="#FFB6D9", outline="#FF6B9D", width=2)
        self.create_arc(x+20, y+3, x+30, y+9, start=0, extent=-180,
                       outline="#FF6B9D", width=3, style="arc")
        # Cheeks
        self.create_oval(x+16, y+1, x+19, y+4, fill="#FFD4E5", outline="")
        self.create_oval(x+31, y+1, x+34, y+4, fill="#FFD4E5", outline="")

        # Legs with animation
        leg_offset = abs(math.sin(jump_phase) * 3)
        self.create_line(x-12, y+15, x-12, y+24-leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        self.create_line(x-4, y+15, x-4, y+24+leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        self.create_line(x+6, y+15, x+6, y+24-leg_offset, fill="#FFB6D9", width=5, capstyle="round")
        self.create_line(x+14, y+15, x+14, y+24+leg_offset, fill="#FFB6D9", width=5, capstyle="round")

        # Hooves
        self.create_oval(x-14, y+22-leg_offset, x-10, y+25-leg_offset, fill="#D4BBFF")
        self.create_oval(x-6, y+22+leg_offset, x-2, y+25+leg_offset, fill="#D4BBFF")
        self.create_oval(x+4, y+22-leg_offset, x+8, y+25-leg_offset, fill="#D4BBFF")
        self.create_oval(x+12, y+22+leg_offset, x+16, y+25+leg_offset, fill="#D4BBFF")

        # Tail
        self.create_oval(x-22, y+5, x-16, y+11, fill="#FFFFFF", outline="#FFB6D9", width=2)

    def start(self):
        self.is_animating = True
        self.frame_count = 0
        self.sheep = []
        self._animate()

    def stop(self):
        self.is_animating = False
        self.delete("all")
        self.sheep = []
        self.frame_count = 0

    def _animate(self):
        if not self.is_animating:
            return

        self.delete("all")

        # Spawn new sheep periodically
        if self.frame_count % 35 == 0:
            self.sheep.append({'x': -40, 'phase': 0})

        new_sheep = []
        for sheep_data in self.sheep:
            sheep_data['x'] += 3.5
            sheep_data['phase'] += 0.25

            if sheep_data['x'] < self.width + 50:
                self.draw_adorable_sheep(sheep_data['x'], self.height // 2, sheep_data['phase'])
                new_sheep.append(sheep_data)

        self.sheep = new_sheep
        self.frame_count += 1

        self.after(35, self._animate)


# ==================== Main GUI Application Class ====================
class XRDProcessingGUI:
    """
    Main GUI Application Class
    Integrates all features:
    1. Batch Integration
    2. Peak Fitting
    3. Full Pipeline
    4. Phase Transition Analysis
    5. Birch-Murnaghan EOS Fitting
    """

    def __init__(self, root):
        self.root = root
        self.root.title("XRD Complete Processing Suite | XRD完整处理套件")
        self.root.geometry("1100x950")
        self.root.resizable(True, True)

        # Color scheme
        self.colors = {
            'bg': '#F8F7FF',
            'card_bg': '#FFFFFF',
            'primary': '#B794F6',
            'primary_hover': '#D4BBFF',
            'secondary': '#E0AAFF',
            'accent': '#FF6B9D',
            'text_dark': '#2B2D42',
            'text_light': '#8B8BA7',
            'border': '#E8E4F3',
            'success': '#06D6A0',
            'error': '#EF476F'
        }

        self.root.configure(bg=self.colors['bg'])

        # Integration and fitting variables
        self.poni_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.input_pattern = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.dataset_path = tk.StringVar(value="entry/data/data")
        self.npt = tk.IntVar(value=4000)
        self.unit = tk.StringVar(value='2th_deg')
        self.fit_method = tk.StringVar(value='pseudo')

        # Phase analysis variables
        self.phase_input_file = tk.StringVar()
        self.phase_output_dir = tk.StringVar()
        self.phase_wavelength = tk.DoubleVar(value=0.4133)
        self.phase_tolerance_1 = tk.DoubleVar(value=0.3)
        self.phase_tolerance_2 = tk.DoubleVar(value=0.4)
        self.phase_tolerance_3 = tk.DoubleVar(value=0.01)
        self.phase_n_points = tk.IntVar(value=4)
        self.phase_original_system = tk.StringVar(value='FCC')
        self.phase_new_system = tk.StringVar(value='FCC')

        # Birch-Murnaghan variables
        self.bm_input_file = tk.StringVar()
        self.bm_output_dir = tk.StringVar()
        self.bm_order = tk.StringVar(value='3')
        self.v0_guess = tk.DoubleVar(value=10.0)
        self.k0_guess = tk.DoubleVar(value=180.0)
        self.k0_prime_guess = tk.DoubleVar(value=4.0)

        self.current_tab = "powder"

        self.setup_ui()

    def setup_ui(self):
        """Create UI layout"""
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['card_bg'], height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        header_content = tk.Frame(header_frame, bg=self.colors['card_bg'])
        header_content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Label(header_content, text="🐱", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 32)).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(header_content, text="XRD Complete Suite | XRD完整套件",
                bg=self.colors['card_bg'], fg=self.colors['text_dark'],
                font=('Comic Sans MS', 20, 'bold')).pack(side=tk.LEFT)

        # Tab bar
        tab_frame = tk.Frame(self.root, bg=self.colors['bg'], height=50)
        tab_frame.pack(fill=tk.X, padx=30, pady=(10, 0))

        tabs_container = tk.Frame(tab_frame, bg=self.colors['bg'])
        tabs_container.pack(side=tk.LEFT)

        self.powder_tab = ModernTab(tabs_container, "Powder XRD | 粉末",
                                    lambda: self.switch_tab("powder"), is_active=True)
        self.powder_tab.pack(side=tk.LEFT, padx=(0, 5))

        self.single_tab = ModernTab(tabs_container, "Single Crystal | 单晶",
                                   lambda: self.switch_tab("single"))
        self.single_tab.pack(side=tk.LEFT, padx=5)

        self.radial_tab = ModernTab(tabs_container, "Radial | 径向",
                                   lambda: self.switch_tab("radial"))
        self.radial_tab.pack(side=tk.LEFT, padx=5)

        # Scrollable container
        container = tk.Frame(self.root, bg=self.colors['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        canvas = tk.Canvas(container, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        self.scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.root.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas
        self.setup_powder_content()

    def setup_powder_content(self):
        """Setup complete content for powder XRD"""
        main_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Section 1: Integration Settings
        integration_card = self.create_card_frame(main_frame)
        integration_card.pack(fill=tk.X, pady=(0, 15))

        content1 = tk.Frame(integration_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content1.pack(fill=tk.BOTH, expand=True)

        header1 = tk.Frame(content1, bg=self.colors['card_bg'])
        header1.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header1, text="🦊", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header1, text="Integration Settings | 积分设置",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.create_file_picker(content1, "PONI File | PONI文件", self.poni_path,
                               [("PONI files", "*.poni"), ("All files", "*.*")])
        self.create_file_picker(content1, "Mask File | 掩码文件", self.mask_path,
                               [("EDF files", "*.edf"), ("All files", "*.*")])
        self.create_file_picker(content1, "Input Pattern | 输入模式", self.input_pattern,
                               [("HDF5 files", "*.h5"), ("All files", "*.*")], pattern=True)
        self.create_folder_picker(content1, "Output Directory | 输出目录", self.output_dir)
        self.create_entry(content1, "Dataset Path | 数据集路径", self.dataset_path)

        param_frame = tk.Frame(content1, bg=self.colors['card_bg'])
        param_frame.pack(fill=tk.X, pady=(5, 0))

        npt_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        npt_cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(npt_cont, text="Number of Points | 点数", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Spinbox(npt_cont, from_=500, to=10000, textvariable=self.npt,
                   width=18, font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        unit_cont = tk.Frame(param_frame, bg=self.colors['card_bg'])
        unit_cont.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(unit_cont, text="Unit | 单位", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(unit_cont, textvariable=self.unit,
                    values=['2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'],
                    width=16, state='readonly', font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # Section 2: Fitting Settings
        fitting_card = self.create_card_frame(main_frame)
        fitting_card.pack(fill=tk.X, pady=(0, 15))

        content2 = tk.Frame(fitting_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content2.pack(fill=tk.BOTH, expand=True)

        header2 = tk.Frame(content2, bg=self.colors['card_bg'])
        header2.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header2, text="🐹", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header2, text="Peak Fitting Settings | 峰拟合设置",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        fit_cont = tk.Frame(content2, bg=self.colors['card_bg'])
        fit_cont.pack(fill=tk.X)
        tk.Label(fit_cont, text="Fitting Method | 拟合方法", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(fit_cont, textvariable=self.fit_method,
                    values=['pseudo', 'voigt'], width=22, state='readonly',
                    font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # Integration and fitting buttons
        btn_frame1 = tk.Frame(main_frame, bg=self.colors['bg'])
        btn_frame1.pack(fill=tk.X, pady=(0, 15))

        btn_cont1 = tk.Frame(btn_frame1, bg=self.colors['bg'])
        btn_cont1.pack(expand=True)

        btns1 = tk.Frame(btn_cont1, bg=self.colors['bg'])
        btns1.pack()

        ModernButton(btns1, "Run Integration | 积分", self.run_integration, icon="🐿️",
                    bg_color=self.colors['secondary'], hover_color=self.colors['primary_hover'],
                    width=220, height=45).pack(side=tk.LEFT, padx=8)

        ModernButton(btns1, "Run Fitting | 拟合", self.run_fitting, icon="🐻",
                    bg_color=self.colors['secondary'], hover_color=self.colors['primary_hover'],
                    width=220, height=45).pack(side=tk.LEFT, padx=8)

        ModernButton(btns1, "Full Pipeline | 完整流程", self.run_full_pipeline, icon="🦔",
                    bg_color=self.colors['primary'], hover_color=self.colors['accent'],
                    width=220, height=45).pack(side=tk.LEFT, padx=8)

        # Section 3: Phase Transition Analysis
        phase_card = self.create_card_frame(main_frame)
        phase_card.pack(fill=tk.X, pady=(0, 15))

        content3 = tk.Frame(phase_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content3.pack(fill=tk.BOTH, expand=True)

        header3 = tk.Frame(content3, bg=self.colors['card_bg'])
        header3.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header3, text="🔬", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header3, text="Phase Transition Analysis | 相变与晶格参数分析",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.create_file_picker(content3, "Input CSV (Peak Data) | 输入CSV(峰数据)",
                               self.phase_input_file, [("CSV files", "*.csv"), ("All files", "*.*")])
        self.create_folder_picker(content3, "Output Directory | 输出目录", self.phase_output_dir)

        # Parameter row 1
        phase_p1 = tk.Frame(content3, bg=self.colors['card_bg'])
        phase_p1.pack(fill=tk.X, pady=(5, 0))

        self.create_param_entry(phase_p1, "Wavelength (Å) | 波长", self.phase_wavelength, True)
        self.create_param_entry(phase_p1, "Tolerance 1 | 容差1", self.phase_tolerance_1, True)
        self.create_param_entry(phase_p1, "Tolerance 2 | 容差2", self.phase_tolerance_2, False)

        # Parameter row 2
        phase_p2 = tk.Frame(content3, bg=self.colors['card_bg'])
        phase_p2.pack(fill=tk.X, pady=(5, 0))

        self.create_param_entry(phase_p2, "Tolerance 3 | 容差3", self.phase_tolerance_3, True)

        npts = tk.Frame(phase_p2, bg=self.colors['card_bg'])
        npts.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(npts, text="N Pressure Points | 压力点数", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Spinbox(npts, from_=1, to=20, textvariable=self.phase_n_points,
                   width=13, font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # Crystal system selection
        phase_p3 = tk.Frame(content3, bg=self.colors['card_bg'])
        phase_p3.pack(fill=tk.X, pady=(5, 0))

        orig_sys = tk.Frame(phase_p3, bg=self.colors['card_bg'])
        orig_sys.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        tk.Label(orig_sys, text="Original System | 原始晶系", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(orig_sys, textvariable=self.phase_original_system,
                    values=['FCC', 'BCC', 'SC', 'Hexagonal', 'Tetragonal',
                           'Orthorhombic', 'Monoclinic', 'Triclinic'],
                    width=16, state='readonly', font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        new_sys = tk.Frame(phase_p3, bg=self.colors['card_bg'])
        new_sys.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(new_sys, text="New System | 新晶系", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Combobox(new_sys, textvariable=self.phase_new_system,
                    values=['FCC', 'BCC', 'SC', 'Hexagonal', 'Tetragonal',
                           'Orthorhombic', 'Monoclinic', 'Triclinic'],
                    width=16, state='readonly', font=('Comic Sans MS', 9)).pack(anchor=tk.W)

        # Phase analysis button
        btn_frame2 = tk.Frame(main_frame, bg=self.colors['bg'])
        btn_frame2.pack(fill=tk.X, pady=(0, 15))

        btn_cont2 = tk.Frame(btn_frame2, bg=self.colors['bg'])
        btn_cont2.pack(expand=True)

        ModernButton(btn_cont2, "Analyze Phase Transition | 相变分析",
                    self.run_phase_analysis, icon="🔬",
                    bg_color="#06D6A0", hover_color="#05B88A",
                    width=280, height=45).pack()

        # Section 4: Birch-Murnaghan
        bm_card = self.create_card_frame(main_frame)
        bm_card.pack(fill=tk.X, pady=(0, 15))

        content4 = tk.Frame(bm_card, bg=self.colors['card_bg'], padx=20, pady=12)
        content4.pack(fill=tk.BOTH, expand=True)

        header4 = tk.Frame(content4, bg=self.colors['card_bg'])
        header4.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(header4, text="⚗️", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(header4, text="Birch-Murnaghan EOS | BM状态方程",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.create_file_picker(content4, "Input CSV (P-V Data) | 输入CSV(P-V数据)",
                               self.bm_input_file, [("CSV files", "*.csv"), ("All files", "*.*")])
        self.create_folder_picker(content4, "Output Directory | 输出目录", self.bm_output_dir)

        order_cont = tk.Frame(content4, bg=self.colors['card_bg'])
        order_cont.pack(fill=tk.X, pady=(5, 0))
        tk.Label(order_cont, text="BM Order | BM阶数", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        order_combo = ttk.Combobox(order_cont, textvariable=self.bm_order,
                                  values=['2', '3'], width=18, state='readonly',
                                  font=('Comic Sans MS', 9))
        order_combo.pack(anchor=tk.W)
        order_combo.bind('<<ComboboxSelected>>', self.on_bm_order_change)

        bm_params = tk.Frame(content4, bg=self.colors['card_bg'])
        bm_params.pack(fill=tk.X, pady=(5, 0))

        self.create_param_entry(bm_params, "V₀ Initial | V₀初值", self.v0_guess, True)
        self.create_param_entry(bm_params, "K₀ Initial | K₀初值", self.k0_guess, True)

        k0p = tk.Frame(bm_params, bg=self.colors['card_bg'])
        k0p.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(k0p, text="K₀' Initial | K₀'初值", bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.k0p_entry = tk.Entry(k0p, textvariable=self.k0_prime_guess,
                                  font=('Comic Sans MS', 9), width=15)
        self.k0p_entry.pack(anchor=tk.W)

        # BM fitting button
        btn_frame3 = tk.Frame(main_frame, bg=self.colors['bg'])
        btn_frame3.pack(fill=tk.X, pady=(0, 15))

        btn_cont3 = tk.Frame(btn_frame3, bg=self.colors['bg'])
        btn_cont3.pack(expand=True)

        ModernButton(btn_cont3, "Birch-Murnaghan Fit | BM拟合",
                    self.run_birch_murnaghan, icon="⚗️",
                    bg_color="#FF6B9D", hover_color="#FF8FB3",
                    width=250, height=45).pack()

        # Progress bar
        prog_cont = tk.Frame(main_frame, bg=self.colors['bg'])
        prog_cont.pack(fill=tk.X, pady=(0, 15))

        prog_inner = tk.Frame(prog_cont, bg=self.colors['bg'])
        prog_inner.pack(expand=True)

        self.progress = CuteSheepProgressBar(prog_inner, width=780, height=80)
        self.progress.pack()

        # Log area
        log_card = self.create_card_frame(main_frame)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_content = tk.Frame(log_card, bg=self.colors['card_bg'], padx=20, pady=12)
        log_content.pack(fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_content, bg=self.colors['card_bg'])
        log_header.pack(anchor=tk.W, pady=(0, 8))

        tk.Label(log_header, text="🐰", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 14)).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(log_header, text="Process Log | 处理日志",
                bg=self.colors['card_bg'], fg=self.colors['primary'],
                font=('Comic Sans MS', 11, 'bold')).pack(side=tk.LEFT)

        self.log_text = scrolledtext.ScrolledText(log_content, height=10, wrap=tk.WORD,
                                                  font=('Comic Sans MS', 10),
                                                  bg='#FAFAFA', fg='#B794F6',
                                                  relief='flat', borderwidth=0, padx=10, pady=10,
                                                  spacing1=3, spacing2=2, spacing3=3)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # Helper UI methods

    def create_card_frame(self, parent, **kwargs):
        card = tk.Frame(parent, bg=self.colors['card_bg'],
                       relief='flat', borderwidth=0, **kwargs)
        card.configure(highlightbackground=self.colors['border'],
                      highlightthickness=1)
        return card

    def create_file_picker(self, parent, label, variable, filetypes, pattern=False):
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        entry = tk.Entry(input_frame, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1, highlightthickness=1,
                        highlightbackground=self.colors['border'],
                        highlightcolor=self.colors['primary'])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        if pattern:
            btn = ModernButton(input_frame, "Browse",
                             lambda: self.browse_pattern(variable, filetypes),
                             bg_color=self.colors['secondary'],
                             hover_color=self.colors['primary'],
                             width=75, height=28)
        else:
            btn = ModernButton(input_frame, "Browse",
                             lambda: self.browse_file(variable, filetypes),
                             bg_color=self.colors['secondary'],
                             hover_color=self.colors['primary'],
                             width=75, height=28)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_folder_picker(self, parent, label, variable):
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        input_frame = tk.Frame(container, bg=self.colors['card_bg'])
        input_frame.pack(fill=tk.X)

        entry = tk.Entry(input_frame, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1, highlightthickness=1,
                        highlightbackground=self.colors['border'],
                        highlightcolor=self.colors['primary'])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        btn = ModernButton(input_frame, "Browse",
                         lambda: self.browse_folder(variable),
                         bg_color=self.colors['secondary'],
                         hover_color=self.colors['primary'],
                         width=75, height=28)
        btn.pack(side=tk.LEFT, padx=(5, 0))

    def create_entry(self, parent, label, variable):
        container = tk.Frame(parent, bg=self.colors['card_bg'])
        container.pack(fill=tk.X, pady=(0, 4))

        tk.Label(container, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        entry = tk.Entry(container, textvariable=variable, font=('Comic Sans MS', 9),
                        bg='white', fg=self.colors['text_dark'], relief='solid',
                        borderwidth=1, highlightthickness=1,
                        highlightbackground=self.colors['border'],
                        highlightcolor=self.colors['primary'])
        entry.pack(fill=tk.X, ipady=3)

    def create_param_entry(self, parent, label, variable, has_margin):
        cont = tk.Frame(parent, bg=self.colors['card_bg'])
        if has_margin:
            cont.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        else:
            cont.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(cont, text=label, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        tk.Entry(cont, textvariable=variable,
                font=('Comic Sans MS', 9), width=15).pack(anchor=tk.W)

    def browse_file(self, variable, filetypes):
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            variable.set(filename)

    def browse_pattern(self, variable, filetypes):
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            folder = os.path.dirname(filename)
            ext = os.path.splitext(filename)[1]
            pattern = os.path.join(folder, f"*{ext}")
            variable.set(pattern)

    def browse_folder(self, variable):
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)

    def log(self, message):
        self.log_text.config(state='normal')
        if "🎀" in message and "🎀 " * 15 in message:
            self.log_text.insert(tk.END, "🎀 " * 10 + "\n")
            self.log_text.insert(tk.END, "🎀 " * 10 + "\n")
        else:
            self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()

    def on_bm_order_change(self, event=None):
        if self.bm_order.get() == '2':
            self.k0p_entry.config(state='disabled')
            self.k0_prime_guess.set(4.0)
        else:
            self.k0p_entry.config(state='normal')

    def switch_tab(self, tab_name):
        self.current_tab = tab_name
        self.powder_tab.set_active(tab_name == "powder")
        self.single_tab.set_active(tab_name == "single")
        self.radial_tab.set_active(tab_name == "radial")

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if tab_name == "powder":
            self.setup_powder_content()
        elif tab_name == "single":
            self.setup_placeholder("Single Crystal XRD", "Coming soon...")
        else:
            self.setup_placeholder("Radial XRD", "Coming soon...")

    def setup_placeholder(self, title, message):
        main_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, pady=50)

        card = self.create_card_frame(main_frame)
        card.pack(fill=tk.BOTH, expand=True)

        content = tk.Frame(card, bg=self.colors['card_bg'], padx=50, pady=50)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="🔬", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 48)).pack(pady=(0, 20))

        tk.Label(content, text=title, bg=self.colors['card_bg'],
                fg=self.colors['text_dark'], font=('Comic Sans MS', 20, 'bold')).pack(pady=(0, 10))

        tk.Label(content, text=message, bg=self.colors['card_bg'],
                fg=self.colors['text_light'], font=('Comic Sans MS', 12)).pack()

    def show_success(self, message):
        dialog = tk.Toplevel(self.root)
        dialog.title("Success | 成功")
        dialog.geometry("450x300")
        dialog.configure(bg=self.colors['card_bg'])
        dialog.resizable(False, False)

        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="✅", bg=self.colors['card_bg'],
                font=('Segoe UI Emoji', 64)).pack(pady=(30, 20))

        tk.Label(dialog, text=message, bg=self.colors['card_bg'],
                fg=self.colors['primary'], font=('Comic Sans MS', 13, 'bold'),
                wraplength=400).pack(pady=(10, 30))

        ModernButton(dialog, "OK", dialog.destroy,
                    bg_color=self.colors['primary'],
                    hover_color=self.colors['primary_hover'],
                    width=120, height=40).pack()

    # Validation methods

    def validate_integration_inputs(self):
        if not self.poni_path.get():
            messagebox.showerror("Error | 错误", "Please select a PONI file | 请选择PONI文件")
            return False
        if not self.mask_path.get():
            messagebox.showerror("Error | 错误", "Please select a mask file | 请选择掩码文件")
            return False
        if not self.input_pattern.get():
            messagebox.showerror("Error | 错误", "Please select input files | 请选择输入文件")
            return False
        if not self.output_dir.get():
            messagebox.showerror("Error | 错误", "Please select output directory | 请选择输出目录")
            return False
        return True

    def validate_fitting_inputs(self):
        if not self.output_dir.get():
            messagebox.showerror("Error | 错误", "Please select output directory | 请选择输出目录")
            return False
        return True

    def validate_phase_inputs(self):
        if not self.phase_input_file.get():
            messagebox.showerror("Error | 错误", "Please select input CSV | 请选择输入CSV文件")
            return False
        if not self.phase_output_dir.get():
            messagebox.showerror("Error | 错误", "Please select output directory | 请选择输出目录")
            return False
        return True

    def validate_birch_murnaghan_inputs(self):
        if not self.bm_input_file.get():
            messagebox.showerror("Error | 错误", "Please select input CSV | 请选择输入CSV文件")
            return False
        if not self.bm_output_dir.get():
            messagebox.showerror("Error | 错误", "Please select output directory | 请选择输出目录")
            return False
        return True

    # Feature 1: Batch Integration

    def run_integration(self):
        if not self.validate_integration_inputs():
            return
        thread = threading.Thread(target=self._run_integration_thread)
        thread.daemon = True
        thread.start()

    def _run_integration_thread(self):
        try:
            self.progress.start()
            self.log("🎀 " * 15)
            self.log("🔁 Starting Batch Integration (HDF5 ➜ .xy)")

            integrator = BatchIntegrator(self.poni_path.get(), self.mask_path.get())
            dataset = self.dataset_path.get() if self.dataset_path.get() else None

            integrator.batch_integrate(
                input_pattern=self.input_pattern.get(),
                output_dir=self.output_dir.get(),
                npt=self.npt.get(),
                unit=self.unit.get(),
                dataset_path=dataset,
                correctSolidAngle=True,
                polarization_factor=None,
                method='csr',
                safe=True,
                normalization_factor=1.0
            )

            self.log("✅ Integration completed successfully!")
            self.show_success("Integration completed! | 积分完成!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error | 错误", f"Integration failed | 积分失败:\n{str(e)}")
        finally:
            self.progress.stop()

    # Feature 2: Peak Fitting

    def run_fitting(self):
        if not self.validate_fitting_inputs():
            return
        thread = threading.Thread(target=self._run_fitting_thread)
        thread.daemon = True
        thread.start()

    def _run_fitting_thread(self):
        try:
            self.progress.start()
            self.log("🎀 " * 15)
            self.log("📈 Starting Batch Fitting (.xy ➜ fitted peaks)")

            fitter = BatchFitter(
                folder=self.output_dir.get(),
                fit_method=self.fit_method.get()
            )
            fitter.run_batch_fitting()

            self.log("✅ Fitting completed successfully!")
            self.show_success("Fitting completed! | 拟合完成!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error | 错误", f"Fitting failed | 拟合失败:\n{str(e)}")
        finally:
            self.progress.stop()

    # Feature 3: Full Pipeline

    def run_full_pipeline(self):
        if not self.validate_integration_inputs():
            return
        thread = threading.Thread(target=self._run_full_pipeline_thread)
        thread.daemon = True
        thread.start()

    def _run_full_pipeline_thread(self):
        try:
            self.progress.start()

            self.log("🎀 " * 15)
            self.log("🔁 Step 1/2: Running Batch Integration")

            integrator = BatchIntegrator(self.poni_path.get(), self.mask_path.get())
            dataset = self.dataset_path.get() if self.dataset_path.get() else None

            integrator.batch_integrate(
                input_pattern=self.input_pattern.get(),
                output_dir=self.output_dir.get(),
                npt=self.npt.get(),
                unit=self.unit.get(),
                dataset_path=dataset,
                correctSolidAngle=True,
                polarization_factor=None,
                method='csr',
                safe=True,
                normalization_factor=1.0
            )

            self.log("✅ Integration completed!")

            self.log("🎀 " * 15)
            self.log("📈 Step 2/2: Running Batch Fitting")

            fitter = BatchFitter(
                folder=self.output_dir.get(),
                fit_method=self.fit_method.get()
            )
            fitter.run_batch_fitting()

            self.log("✅ Fitting completed!")
            self.log("🎀 " * 15)
            self.log("🎉 Full pipeline completed!")

            self.show_success("Full pipeline completed! | 完整流程完成!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error | 错误", f"Pipeline failed | 流程失败:\n{str(e)}")
        finally:
            self.progress.stop()

    # Feature 4: Phase Transition Analysis

    def run_phase_analysis(self):
        if not self.validate_phase_inputs():
            return
        thread = threading.Thread(target=self._run_phase_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_phase_analysis_thread(self):
        try:
            self.progress.start()
            self.log("🎀 " * 15)
            self.log("🔬 Starting Phase Transition Analysis")

            # Map crystal system names
            system_map = {
                'FCC': 'cubic_FCC',
                'BCC': 'cubic_BCC',
                'SC': 'cubic_SC',
                'Hexagonal': 'Hexagonal',
                'Tetragonal': 'Tetragonal',
                'Orthorhombic': 'Orthorhombic',
                'Monoclinic': 'Monoclinic',
                'Triclinic': 'Triclinic'
            }

            original_sys = system_map[self.phase_original_system.get()]
            new_sys = system_map[self.phase_new_system.get()]

            analyzer = XRayDiffractionAnalyzer(
                wavelength=self.phase_wavelength.get(),
                peak_tolerance_1=self.phase_tolerance_1.get(),
                peak_tolerance_2=self.phase_tolerance_2.get(),
                peak_tolerance_3=self.phase_tolerance_3.get(),
                n_pressure_points=self.phase_n_points.get()
            )

            results = analyzer.analyze(
                csv_path=self.phase_input_file.get(),
                original_system=original_sys,
                new_system=new_sys,
                auto_mode=True,
                log_callback=self.log
            )

            if results:
                self.log("\n✅ Phase analysis completed!")
                self.show_success("Phase analysis completed! | 相变分析完成!")
            else:
                self.log("❌ Analysis failed")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error | 错误", f"Analysis failed | 分析失败:\n{str(e)}")
        finally:
            self.progress.stop()

    # Feature 5: Birch-Murnaghan EOS Fitting

    def run_birch_murnaghan(self):
        if not self.validate_birch_murnaghan_inputs():
            return
        thread = threading.Thread(target=self._run_birch_murnaghan_thread)
        thread.daemon = True
        thread.start()

    def _run_birch_murnaghan_thread(self):
        try:
            self.progress.start()
            self.log("🎀 " * 15)
            order_str = "2nd order" if self.bm_order.get() == '2' else "3rd order"
            self.log(f"⚗️ Starting {order_str} Birch-Murnaghan Fitting")

            fitter = BirchMurnaghanFitter(
                data_file=self.bm_input_file.get(),
                output_dir=self.bm_output_dir.get(),
                order=int(self.bm_order.get())
            )

            if self.bm_order.get() == '2':
                initial_guess = [
                    self.v0_guess.get(),
                    self.k0_guess.get()
                ]
            else:
                initial_guess = [
                    self.v0_guess.get(),
                    self.k0_guess.get(),
                    self.k0_prime_guess.get()
                ]

            results = fitter.fit(initial_guess=initial_guess)

            self.log("\n✅ Birch-Murnaghan fitting completed!")
            self.log(f"📊 V₀ = {results['V0']:.4f} ± {results['V0_err']:.4f} Å³")
            self.log(f"📊 K₀ = {results['K0']:.4f} ± {results['K0_err']:.4f} GPa")

            if self.bm_order.get() == '3':
                self.log(f"📊 K₀' = {results['K0_prime']:.4f} ± {results['K0_prime_err']:.4f}")
            else:
                self.log(f"📊 K₀' = 4.0 (fixed)")

            self.log(f"📈 R² = {results['r_squared']:.6f}")
            self.log(f"\n💾 Results saved to: {self.bm_output_dir.get()}")

            self.show_success(f"{order_str} BM fitting completed! | BM拟合完成!")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error | 错误", f"BM fitting failed | BM拟合失败:\n{str(e)}")
        finally:
            self.progress.stop()


# Main Entry Point
def main():
    """Main program entry point"""
    root = tk.Tk()
    app = XRDProcessingGUI(root)

    # Set window icon (if available)
    # root.iconbitmap('icon.ico')

    # Run main loop
    root.mainloop()


if __name__ == "__main__":
    main()
