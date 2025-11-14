# -*- coding: utf-8 -*-
"""
X-Ray Diffraction Analysis Module
Calculates unit cell volumes from peak positions
Analyzes phase transitions and separates original/new peaks
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


class XRayDiffractionAnalyzer:
    """Analyzer for XRD data to calculate volumes and detect phase transitions"""

    def __init__(self, wavelength=0.4133, peak_tolerance_1=0.3,
                 peak_tolerance_2=0.4, peak_tolerance_3=0.01,
                 n_pressure_points=4, csv_path=None):
        """
        Initialize the XRD analyzer

        Parameters:
        -----------
        wavelength : float
            X-ray wavelength in Angstroms
        peak_tolerance_1 : float
            Tolerance for phase transition detection
        peak_tolerance_2 : float
            Tolerance for new peak tracking
        peak_tolerance_3 : float
            Tolerance for original peak identification
        n_pressure_points : int
            Minimum number of pressure points required
        csv_path : str
            Path to CSV file with peak data
        """
        self.wavelength = wavelength
        self.peak_tolerance_1 = peak_tolerance_1
        self.peak_tolerance_2 = peak_tolerance_2
        self.peak_tolerance_3 = peak_tolerance_3
        self.n_pressure_points = n_pressure_points
        self.csv_path = csv_path

    @staticmethod
    def bragg_to_d_spacing(two_theta_deg, wavelength):
        """
        Convert 2θ to d-spacing using Bragg's law

        Parameters:
        -----------
        two_theta_deg : float or array
            2θ angle in degrees
        wavelength : float
            X-ray wavelength in Angstroms

        Returns:
        --------
        d : float or array
            d-spacing in Angstroms
        """
        theta_rad = np.radians(two_theta_deg / 2)
        d = wavelength / (2 * np.sin(theta_rad))
        return d

    @staticmethod
    def calculate_volume(d_spacings, crystal_system='FCC', hkl_indices=None):
        """
        Calculate unit cell volume from d-spacings

        Parameters:
        -----------
        d_spacings : array-like
            d-spacing values
        crystal_system : str
            Crystal system type (FCC, BCC, SC, Hexagonal, etc.)
        hkl_indices : list of tuples
            Miller indices for each peak

        Returns:
        --------
        volume : float
            Unit cell volume in Å³
        """
        if crystal_system == 'FCC':
            # For FCC, (111) is typically the strongest peak
            # a = d * sqrt(h^2 + k^2 + l^2)
            # For (111): a = d * sqrt(3)
            a = np.mean(d_spacings) * np.sqrt(3)
            volume = a ** 3

        elif crystal_system == 'BCC':
            # For BCC, (110) is typically the strongest peak
            # For (110): a = d * sqrt(2)
            a = np.mean(d_spacings) * np.sqrt(2)
            volume = a ** 3

        elif crystal_system == 'SC':
            # For simple cubic
            a = np.mean(d_spacings)
            volume = a ** 3

        elif crystal_system == 'Hexagonal':
            # For hexagonal: need (100) and (001) peaks
            # Assuming first peak is (100)
            a = d_spacings[0] * 2 / np.sqrt(3)
            # Assuming ratio c/a ~ 1.633 (ideal)
            c = a * 1.633
            volume = a ** 2 * c * np.sqrt(3) / 2

        else:
            # Generic cubic
            a = np.mean(d_spacings)
            volume = a ** 3

        return volume

    def read_pressure_peak_data(self, csv_path):
        """
        Read peak data from CSV file organized by pressure

        Expected CSV format:
        File, Pressure, Peak Index, Peak #, Center, Amplitude, ...
        OR
        File, Peak #, Center, ... (pressure extracted from filename)

        Returns:
        --------
        pressure_data : dict
            Dictionary with pressure as key and list of peak positions as value
        """
        df = pd.read_csv(csv_path)

        # Debug: print columns
        print(f"CSV columns: {df.columns.tolist()}")

        pressure_data = defaultdict(list)

        # Group by file/pressure
        for idx, row in df.iterrows():
            try:
                pressure = None

                # Method 1: Check for 'Pressure' column
                if 'Pressure' in df.columns:
                    try:
                        pressure = float(row['Pressure'])
                    except (ValueError, TypeError):
                        pass

                # Method 2: Extract from 'File' column
                if pressure is None and 'File' in df.columns:
                    import re
                    file_str = str(row['File'])
                    # Try various patterns: "10GPa", "10 GPa", "10.5GPa", etc.
                    match = re.search(r'(\d+\.?\d*)\s*[Gg][Pp][Aa]', file_str)
                    if match:
                        pressure = float(match.group(1))
                    else:
                        # Try pattern like "file_10_" where 10 is pressure
                        match = re.search(r'[_\-](\d+\.?\d*)[_\-]', file_str)
                        if match:
                            pressure = float(match.group(1))

                # Method 3: Use row index as pressure (sequential)
                if pressure is None:
                    pressure = float(idx)

                # Get peak position - try multiple column names
                peak_position = None
                for col_name in ['Center', 'center', 'Peak Position', 'Position', '2theta', 'Two-Theta']:
                    if col_name in df.columns:
                        try:
                            peak_position = float(row[col_name])
                            break
                        except (ValueError, TypeError):
                            continue

                # If still no peak position, use first numeric column after file info
                if peak_position is None:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        peak_position = float(row[numeric_cols[0]])

                if peak_position is not None:
                    pressure_data[pressure].append(peak_position)
                    print(f"Row {idx}: Pressure={pressure:.2f}, Peak={peak_position:.4f}")

            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Skipping row {idx}: {e}")
                continue

        if not pressure_data:
            raise ValueError("No valid pressure-peak data found in CSV file. Please check the file format.")

        print(f"\nSuccessfully loaded data for {len(pressure_data)} pressure points")
        for p in sorted(pressure_data.keys())[:3]:  # Show first 3
            print(f"  Pressure {p:.2f}: {len(pressure_data[p])} peaks")

        return dict(pressure_data)

    def find_phase_transition_point(self, pressure_data, tolerance=0.3):
        """
        Detect phase transition by finding when new peaks appear

        Parameters:
        -----------
        pressure_data : dict
            Dictionary with pressure as key and list of peaks as value
        tolerance : float
            Tolerance for considering peaks as "new"

        Returns:
        --------
        transition_pressure : float
            Pressure at which phase transition occurs
        before_pressures : list
            Pressures before transition
        after_pressures : list
            Pressures after transition
        """
        pressures = sorted(pressure_data.keys())

        if len(pressures) < 2:
            return None, [], []

        # Count peaks at each pressure
        peak_counts = [len(pressure_data[p]) for p in pressures]

        # Find where peak count significantly increases
        for i in range(1, len(pressures)):
            if peak_counts[i] > peak_counts[i-1] + 1:  # New peaks appeared
                transition_pressure = pressures[i]
                before_pressures = pressures[:i]
                after_pressures = pressures[i:]
                return transition_pressure, before_pressures, after_pressures

        # No clear transition found
        return None, pressures, []

    def build_original_peak_dataset(self, pressure_data, tracked_new_peak_dataset=None,
                                   tolerance=0.01):
        """
        Identify original peaks (present throughout pressure range)

        Parameters:
        -----------
        pressure_data : dict
            Dictionary with pressure as key and list of peaks as value
        tracked_new_peak_dataset : set
            Set of peak indices identified as new peaks
        tolerance : float
            Tolerance for peak position matching

        Returns:
        --------
        original_peaks : set
            Set of peak indices identified as original peaks
        """
        if not pressure_data:
            return set()

        pressures = sorted(pressure_data.keys())
        reference_peaks = pressure_data[pressures[0]]

        original_peaks = set(range(len(reference_peaks)))

        # Exclude tracked new peaks if provided
        if tracked_new_peak_dataset:
            original_peaks -= tracked_new_peak_dataset

        return original_peaks

    def collect_tracked_new_peaks(self, pressure_data, transition_pressure,
                                 after_pressures, new_peaks_ref=None, tolerance=0.4):
        """
        Collect and track new peaks that appear after phase transition

        Parameters:
        -----------
        pressure_data : dict
            Dictionary with pressure as key and list of peaks as value
        transition_pressure : float
            Pressure at which phase transition occurs
        after_pressures : list
            Pressures after transition
        new_peaks_ref : list
            Reference new peak positions
        tolerance : float
            Tolerance for peak tracking

        Returns:
        --------
        new_peaks : set
            Set of peak indices identified as new peaks
        """
        new_peaks = set()

        if not after_pressures or transition_pressure is None:
            return new_peaks

        # Find peaks that appear at transition
        before_peaks = set()
        for p in sorted(pressure_data.keys()):
            if p < transition_pressure:
                before_peaks.update(pressure_data[p])
            else:
                after_peaks = set(pressure_data[p])
                # New peaks are those not in before_peaks
                potential_new = after_peaks - before_peaks
                new_peaks.update(potential_new)
                break

        return new_peaks

    def calculate_volumes_from_peaks(self, pressure_data, crystal_system='FCC',
                                   output_dir=None):
        """
        Calculate unit cell volumes for each pressure point

        Parameters:
        -----------
        pressure_data : dict
            Dictionary with pressure as key and list of 2θ values as value
        crystal_system : str
            Crystal system type
        output_dir : str
            Output directory for results

        Returns:
        --------
        results_df : DataFrame
            DataFrame with columns [Pressure, Volume, d_spacing_avg]
        """
        results = []

        for pressure in sorted(pressure_data.keys()):
            two_theta_values = pressure_data[pressure]

            # Convert to d-spacings
            d_spacings = [self.bragg_to_d_spacing(tt, self.wavelength)
                         for tt in two_theta_values]

            # Calculate volume
            volume = self.calculate_volume(d_spacings, crystal_system)

            results.append({
                'Pressure (GPa)': pressure,
                'Volume (Å³)': volume,
                'd_spacing_avg (Å)': np.mean(d_spacings)
            })

        results_df = pd.DataFrame(results)

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'pressure_volume_results.csv')
            results_df.to_csv(output_file, index=False)
            print(f"✅ Results saved to {output_file}")

            # Plot P-V curve
            self._plot_pv_curve(results_df, output_dir)

        return results_df

    def _plot_pv_curve(self, results_df, output_dir):
        """Plot pressure-volume curve"""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(results_df['Volume (Å³)'], results_df['Pressure (GPa)'],
               'ko-', markersize=8, linewidth=2)
        ax.set_xlabel('Volume (Å³)', fontsize=12)
        ax.set_ylabel('Pressure (GPa)', fontsize=12)
        ax.set_title('Pressure-Volume Relationship', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'pressure_volume_plot.png')
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"✅ Plot saved to {plot_file}")
