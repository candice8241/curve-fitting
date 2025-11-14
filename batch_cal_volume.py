# -*- coding: utf-8 -*-
"""
XRD Phase Analysis and Volume Calculation Module
Adapted to read CSV format with pressure and peak columns
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class XRayDiffractionAnalyzer:
    """
    Analyzer for XRD data to identify phase transitions and calculate volumes

    CSV Format Expected:
    - Column: pressure (gpa) or pressure(gpa) - pressure values
    - Columns: peak_1_2theta, peak_2_2theta, ... - peak positions
    - Column: number of peaks - count of peaks (optional, for reference)
    """

    def __init__(self, wavelength=0.4133, peak_tolerance_1=0.3,
                 peak_tolerance_2=0.4, peak_tolerance_3=0.01, n_pressure_points=4):
        """
        Initialize the analyzer

        Parameters:
        -----------
        wavelength : float
            X-ray wavelength in Angstroms
        peak_tolerance_1 : float
            Tolerance for initial phase transition detection
        peak_tolerance_2 : float
            Tolerance for tracking new peaks
        peak_tolerance_3 : float
            Tolerance for matching original peaks
        n_pressure_points : int
            Number of consecutive pressure points to confirm phase transition
        """
        self.wavelength = wavelength
        self.peak_tolerance_1 = peak_tolerance_1
        self.peak_tolerance_2 = peak_tolerance_2
        self.peak_tolerance_3 = peak_tolerance_3
        self.n_pressure_points = n_pressure_points
        self.pressure_data = {}

    def read_pressure_peak_data(self, csv_path):
        """
        Read CSV file with pressure and peak data

        Expected columns:
        - pressure (gpa) or similar: pressure values
        - peak_1_2theta, peak_2_2theta, ...: peak positions
        - number of peaks: peak count (optional)

        Returns:
        --------
        dict: {pressure: [peak1, peak2, ...]}
        """
        print(f"Reading CSV file: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        # Find pressure column (case-insensitive, flexible naming)
        pressure_col = None
        for col in df.columns:
            col_lower = col.lower().replace(' ', '').replace('(', '').replace(')', '')
            if 'pressure' in col_lower or 'gpa' in col_lower:
                pressure_col = col
                break

        if pressure_col is None:
            raise ValueError(
                "CSV file must contain a pressure column (e.g., 'pressure (gpa)', 'Pressure', 'pressure(gpa)')\n"
                f"Found columns: {list(df.columns)}"
            )

        # Find all peak columns (looking for peak_X_2theta or similar)
        peak_cols = []
        for col in df.columns:
            col_lower = col.lower().replace(' ', '')
            if 'peak' in col_lower and '2theta' in col_lower:
                peak_cols.append(col)

        # Sort peak columns by number
        def extract_peak_number(col_name):
            import re
            match = re.search(r'peak[_\s]*(\d+)', col_name, re.IGNORECASE)
            return int(match.group(1)) if match else 999

        peak_cols.sort(key=extract_peak_number)

        if not peak_cols:
            raise ValueError(
                "CSV file must contain peak position columns (e.g., 'peak_1_2theta', 'peak_2_2theta', ...)\n"
                f"Found columns: {list(df.columns)}"
            )

        print(f"✓ Found pressure column: {pressure_col}")
        print(f"✓ Found {len(peak_cols)} peak columns: {peak_cols[:5]}..." if len(peak_cols) > 5 else f"✓ Found peak columns: {peak_cols}")

        # Build pressure-peak dictionary
        pressure_data = {}
        for _, row in df.iterrows():
            pressure = float(row[pressure_col])
            peaks = []

            for peak_col in peak_cols:
                peak_value = row[peak_col]
                # Only add non-null, non-zero, and positive peaks
                if pd.notna(peak_value) and peak_value > 0:
                    peaks.append(float(peak_value))

            if peaks:  # Only add if there are valid peaks
                pressure_data[pressure] = sorted(peaks)

        self.pressure_data = pressure_data
        print(f"✓ Loaded data for {len(pressure_data)} pressure points")
        print(f"  Pressure range: {min(pressure_data.keys()):.2f} - {max(pressure_data.keys()):.2f} GPa")

        return pressure_data

    def find_phase_transition_point(self):
        """
        Identify the phase transition pressure point

        Returns:
        --------
        tuple: (transition_pressure, before_pressures, after_pressures)
        """
        if not self.pressure_data:
            raise ValueError("No pressure data loaded. Call read_pressure_peak_data() first.")

        sorted_pressures = sorted(self.pressure_data.keys())

        for i in range(len(sorted_pressures) - self.n_pressure_points):
            current_pressure = sorted_pressures[i]
            next_pressure = sorted_pressures[i + 1]

            current_peaks = self.pressure_data[current_pressure]
            next_peaks = self.pressure_data[next_pressure]

            # Check if new peaks appear
            new_peaks_found = False
            for peak in next_peaks:
                # Check if this peak exists in current pressure within tolerance
                in_current = any(abs(peak - cp) <= self.peak_tolerance_1 for cp in current_peaks)
                if not in_current:
                    new_peaks_found = True
                    break

            if new_peaks_found:
                # Verify this continues for n_pressure_points
                stable = True
                for j in range(1, min(self.n_pressure_points, len(sorted_pressures) - i - 1)):
                    check_pressure = sorted_pressures[i + j + 1]
                    check_peaks = self.pressure_data[check_pressure]

                    # Check if similar new peaks persist
                    new_in_check = False
                    for peak in check_peaks:
                        in_original = any(abs(peak - cp) <= self.peak_tolerance_1 for cp in current_peaks)
                        if not in_original:
                            new_in_check = True
                            break

                    if not new_in_check:
                        stable = False
                        break

                if stable:
                    transition_pressure = next_pressure
                    before_pressures = sorted_pressures[:i+1]
                    after_pressures = sorted_pressures[i+1:]
                    return transition_pressure, before_pressures, after_pressures

        return None, [], []

    def collect_tracked_new_peaks(self, pressure_data, transition_pressure,
                                   after_pressures, new_peaks_at_transition,
                                   tolerance, output_csv=None):
        """
        Track new peaks across pressure points after phase transition

        Returns:
        --------
        tuple: (stable_peak_count, tracked_new_peaks_dict)
        """
        tracked_new_peaks = {p: [] for p in after_pressures}

        # Track each new peak
        for new_peak in new_peaks_at_transition:
            trajectory = []

            for pressure in after_pressures:
                peaks_at_pressure = pressure_data[pressure]
                # Find closest match
                matches = [p for p in peaks_at_pressure if abs(p - new_peak) <= tolerance]

                if matches:
                    closest = min(matches, key=lambda p: abs(p - new_peak))
                    trajectory.append(closest)
                    tracked_new_peaks[pressure].append(closest)
                else:
                    trajectory.append(None)

        # Count stable peaks (present at transition point)
        stable_count = len(new_peaks_at_transition)

        # Export to CSV if requested
        if output_csv:
            df_data = {'Pressure (GPa)': after_pressures}
            for pressure in after_pressures:
                df_data[pressure] = tracked_new_peaks.get(pressure, [])

            # Convert to DataFrame format
            max_peaks = max(len(peaks) for peaks in tracked_new_peaks.values())
            export_data = {'Pressure (GPa)': after_pressures}

            for i in range(max_peaks):
                peak_values = []
                for pressure in after_pressures:
                    peaks = tracked_new_peaks[pressure]
                    peak_values.append(peaks[i] if i < len(peaks) else np.nan)
                export_data[f'peak_{i+1}_2theta'] = peak_values

            df = pd.DataFrame(export_data)
            df.to_csv(output_csv, index=False)
            print(f"✓ New peaks dataset saved to: {output_csv}")

        return stable_count, tracked_new_peaks

    def build_original_peak_dataset(self, pressure_data, tracked_new_peaks,
                                     tolerance, output_csv=None):
        """
        Build dataset of original peaks (excluding new peaks)

        Returns:
        --------
        dict: {pressure: [original_peaks]}
        """
        original_peak_dataset = {}

        for pressure in sorted(pressure_data.keys()):
            all_peaks = pressure_data[pressure]
            new_peaks = tracked_new_peaks.get(pressure, [])

            # Exclude new peaks
            original_peaks = []
            for peak in all_peaks:
                is_new = any(abs(peak - np) <= tolerance for np in new_peaks)
                if not is_new:
                    original_peaks.append(peak)

            original_peak_dataset[pressure] = original_peaks

        # Export to CSV if requested
        if output_csv:
            pressures = sorted(original_peak_dataset.keys())
            max_peaks = max(len(peaks) for peaks in original_peak_dataset.values())

            export_data = {'Pressure (GPa)': pressures}
            for i in range(max_peaks):
                peak_values = []
                for pressure in pressures:
                    peaks = original_peak_dataset[pressure]
                    peak_values.append(peaks[i] if i < len(peaks) else np.nan)
                export_data[f'peak_{i+1}_2theta'] = peak_values

            df = pd.DataFrame(export_data)
            df.to_csv(output_csv, index=False)
            print(f"✓ Original peaks dataset saved to: {output_csv}")

        return original_peak_dataset

    def analyze(self, csv_path, original_system='cubic_FCC', new_system='cubic_FCC',
                auto_mode=True):
        """
        Complete analysis workflow

        Parameters:
        -----------
        csv_path : str
            Path to input CSV file
        original_system : str
            Crystal system for original phase
        new_system : str
            Crystal system for new phase
        auto_mode : bool
            If True, automatically process everything

        Returns:
        --------
        dict: Analysis results
        """
        # Read data
        pressure_data = self.read_pressure_peak_data(csv_path)

        # Find phase transition
        transition_pressure, before_pressures, after_pressures = self.find_phase_transition_point()

        results = {
            'transition_pressure': transition_pressure,
            'before_pressures': before_pressures,
            'after_pressures': after_pressures,
            'pressure_data': pressure_data
        }

        if transition_pressure is None:
            print("⚠️ No phase transition detected")
            results['phase_transition_detected'] = False
            return results

        results['phase_transition_detected'] = True
        print(f"✓ Phase transition detected at {transition_pressure:.2f} GPa")

        return results
