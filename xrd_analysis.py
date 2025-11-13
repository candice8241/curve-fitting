# -*- coding: utf-8 -*-
"""
X-ray Diffraction Analysis Tool
Created on Thu Nov 13 14:30:34 2025

@author: 16961
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration Parameters ====================
WAVELENGTH = 0.4133  # X-ray wavelength (Å), modify according to actual conditions
PEAK_TOLERANCE_1 = 0.3  # Tolerance for phase transition identification (degrees)
PEAK_TOLERANCE_2 = 0.42  # Tolerance for determining new peak count (degrees)
PEAK_TOLERANCE_3 = 0.02  # Tolerance for tracking new peaks at subsequent pressures (degrees)
N_PRESSURE_POINTS = 4  # Number of pressure points required for stable new peak determination

# ==================== Crystal System HKL Order Definitions ====================
# Define hkl order for each crystal system sorted by 2theta (first 20 peaks)

CRYSTAL_SYSTEMS = {
    'cubic_FCC': {
        'name': 'FCC',
        'min_peaks': 1,
        'atoms_per_cell': 4,  # FCC has 4 atoms per unit cell
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
        'atoms_per_cell': 2,  # BCC has 2 atoms per unit cell
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
        'atoms_per_cell': 1,  # Simple cubic has 1 atom per unit cell
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
        'atoms_per_cell': 2,  # HCP has 2 atoms per unit cell
        'hkl_list': [
            (1,0,0), (0,0,2), (1,0,1), (1,0,2), (1,1,0),
            (1,0,3), (2,0,0), (1,1,2), (2,0,1), (0,0,4),
            (2,0,2), (1,0,4), (2,0,3), (2,1,0), (2,1,1),
            (2,0,4), (2,1,2), (3,0,0), (2,1,3), (2,2,0)
        ]
    },
    'Tetragonal': {
        'name': 'Tetragonal',
        'min_peaks': 2,
        'atoms_per_cell': 1,  # Default value, can be adjusted
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
        'atoms_per_cell': 1,  # Default value, can be adjusted
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
        'atoms_per_cell': 1,  # Default value, can be adjusted
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
        'atoms_per_cell': 1,  # Default value, can be adjusted
        'hkl_list': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (1,-1,0), (1,0,-1), (0,1,-1), (1,1,1),
            (1,-1,1), (1,1,-1), (2,0,0), (0,2,0), (0,0,2),
            (2,1,0), (2,0,1), (1,2,0), (0,2,1), (1,0,2)
        ]
    }
}

# ==================== Utility Functions ====================

def two_theta_to_d(two_theta, wavelength=WAVELENGTH):
    """
    Convert 2theta angle to d-spacing

    Parameters:
        two_theta: 2theta angle (degrees)
        wavelength: X-ray wavelength (Å)

    Returns:
        d-spacing (Å)
    """
    theta_rad = np.deg2rad(two_theta / 2.0)
    return wavelength / (2.0 * np.sin(theta_rad))

def d_to_two_theta(d, wavelength=WAVELENGTH):
    """
    Convert d-spacing to 2theta angle

    Parameters:
        d: d-spacing (Å)
        wavelength: X-ray wavelength (Å)

    Returns:
        2theta angle (degrees)
    """
    sin_theta = wavelength / (2.0 * d)
    if sin_theta > 1.0 or sin_theta < -1.0:
        return None
    theta_rad = np.arcsin(sin_theta)
    return np.rad2deg(2.0 * theta_rad)

def calculate_d_cubic(hkl, a):
    """Calculate d-spacing for cubic crystal system"""
    h, k, l = hkl
    return a / np.sqrt(h**2 + k**2 + l**2)

def calculate_d_hexagonal(hkl, a, c):
    """Calculate d-spacing for hexagonal crystal system"""
    h, k, l = hkl
    return 1.0 / np.sqrt(4.0/3.0 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)

def calculate_d_tetragonal(hkl, a, c):
    """Calculate d-spacing for tetragonal crystal system"""
    h, k, l = hkl
    return 1.0 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)

def calculate_d_orthorhombic(hkl, a, b, c):
    """Calculate d-spacing for orthorhombic crystal system"""
    h, k, l = hkl
    return 1.0 / np.sqrt(h**2 / a**2 + k**2 / b**2 + l**2 / c**2)

def calculate_d_monoclinic(hkl, a, b, c, beta):
    """Calculate d-spacing for monoclinic crystal system"""
    h, k, l = hkl
    beta_rad = np.deg2rad(beta)
    sin_beta = np.sin(beta_rad)
    cos_beta = np.cos(beta_rad)

    term = (h**2 / a**2 + k**2 * sin_beta**2 / b**2 + l**2 / c**2
            - 2*h*l*cos_beta / (a*c)) / sin_beta**2
    return 1.0 / np.sqrt(term)

# ==================== Unit Cell Volume Calculation ====================

def calculate_cell_volume_cubic(a):
    """Calculate unit cell volume for cubic system"""
    return a**3

def calculate_cell_volume_hexagonal(a, c):
    """Calculate unit cell volume for hexagonal system"""
    return np.sqrt(3) / 2 * a**2 * c

def calculate_cell_volume_tetragonal(a, c):
    """Calculate unit cell volume for tetragonal system"""
    return a**2 * c

def calculate_cell_volume_orthorhombic(a, b, c):
    """Calculate unit cell volume for orthorhombic system"""
    return a * b * c

def calculate_cell_volume_monoclinic(a, b, c, beta):
    """Calculate unit cell volume for monoclinic system"""
    beta_rad = np.deg2rad(beta)
    return a * b * c * np.sin(beta_rad)

# ==================== CSV Reading and Data Preprocessing ====================

def read_pressure_peak_data(csv_path):
    """
    Read CSV file and extract pressure points and peak positions

    Parameters:
        csv_path: Path to CSV file

    Returns:
        pressure_data: Dictionary with pressure values (GPa) as keys and peak position lists (2theta) as values
    """
    df = pd.read_csv(csv_path)

    # Check for required columns
    if 'File' not in df.columns or 'Center' not in df.columns:
        raise ValueError("CSV file must contain 'File' and 'Center' columns")

    pressure_data = {}
    current_pressure = None

    for idx, row in df.iterrows():
        # Check if this is a blank row (separator)
        if pd.isna(row['File']) or row['File'] == '':
            current_pressure = None
            continue

        # Extract pressure value
        try:
            # Assume File column contains pressure info, format may be "filename_XXGPa" or direct number
            file_str = str(row['File'])
            # Try to extract number
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', file_str)
            if numbers:
                pressure = float(numbers[0])
            else:
                pressure = float(file_str)
        except:
            print(f"Warning: Cannot parse pressure value: {row['File']}")
            continue

        # Extract peak position
        try:
            peak_position = float(row['Center'])
        except:
            print(f"Warning: Cannot parse peak position: {row['Center']}")
            continue

        # Add to dictionary
        if pressure not in pressure_data:
            pressure_data[pressure] = []
        pressure_data[pressure].append(peak_position)

    # Sort peak positions for each pressure point
    for pressure in pressure_data:
        pressure_data[pressure] = sorted(pressure_data[pressure])

    return pressure_data

# ==================== Phase Transition Identification ====================

def find_phase_transition_point(pressure_data, tolerance=PEAK_TOLERANCE_1):
    """
    Identify phase transition point (interval method): Starting from minimum pressure,
    compare adjacent pressure points. If a peak doesn't fall within tolerance window
    of any peak from previous pressure point, it's a new peak.

    Parameters:
        pressure_data: Pressure-peak data dictionary
        tolerance: Peak position tolerance (degrees)

    Returns:
        transition_pressure: Phase transition pressure point (GPa)
        before_pressures: List of pressure points before transition
        after_pressures: List of pressure points after transition
    """
    sorted_pressures = sorted(pressure_data.keys())

    if len(sorted_pressures) < 2:
        print("Warning: Less than 2 pressure points, cannot identify phase transition")
        return None, sorted_pressures, []

    for i in range(1, len(sorted_pressures)):
        prev_pressure = sorted_pressures[i - 1]
        curr_pressure = sorted_pressures[i]

        prev_peaks = pressure_data[prev_pressure]
        curr_peaks = pressure_data[curr_pressure]

        # Build tolerance windows for previous pressure point peaks
        tolerance_windows = [(p - tolerance, p + tolerance) for p in prev_peaks]

        # Check if current point has new peaks
        has_new_peak = False
        for peak in curr_peaks:
            # Check if this peak falls in any tolerance window
            in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
            if not in_any_window:
                has_new_peak = True
                break

        if has_new_peak:
            print(f"\n>>> Phase transition detected at: {curr_pressure:.2f} GPa")
            return curr_pressure, sorted_pressures[:i], sorted_pressures[i:]

    print("\n>>> No obvious phase transition detected")
    return None, sorted_pressures, []



# ==================== New Peak Statistics ====================

def build_original_peak_dataset(pressure_data, tracked_new_peak_dataset, tolerance=PEAK_TOLERANCE_3):
    """
    Build original peak dataset (all pressure points) based on new peak dataset

    Parameters:
        pressure_data: dict, {pressure: [all peak positions]}
        tracked_new_peak_dataset: dict, {pressure: [new peak positions]} from collect_tracked_new_peaks
        tolerance: Tolerance range for determining if peak belongs to new peaks

    Returns:
        original_peak_dataset: dict, {pressure: {'original_peaks': [...], 'count': x}}
    """
    original_peak_dataset = {}

    for pressure, all_peaks in pressure_data.items():
        new_peaks = tracked_new_peak_dataset.get(pressure, [])
        original_peaks = []

        for peak in all_peaks:
            # Check if peak matches any new peak (within ±tolerance)
            is_new = any(abs(peak - new_peak) <= tolerance for new_peak in new_peaks)
            if not is_new:
                original_peaks.append(peak)

        original_peak_dataset[pressure] = {
            'original_peaks': original_peaks,
            'count': len(original_peaks)
        }

    return original_peak_dataset



def collect_tracked_new_peaks(pressure_data, transition_pressure,
                               after_pressures, new_peaks_ref,
                               tolerance=PEAK_TOLERANCE_2):
    """
    Track specified new peaks starting from phase transition pressure (based on ±tolerance matching)

    Parameters:
        pressure_data: dict, pressure → peak position list
        transition_pressure: Phase transition point, used for new peak reference
        after_pressures: List of pressure points after transition
        new_peaks_ref: List of new peaks found at transition point
        tolerance: Tolerance range for matching

    Returns:
        stable_count: Number of stable new peaks (appearing ≥ N times based on N_PRESSURE_POINTS)
        tracked_peaks_dict: dict, keys are pressure points, values are lists of matched new peaks
    """
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

    # Count "stable" new peaks (appearing ≥ N_PRESSURE_POINTS times)
    stable_count = sum(1 for count in peak_occurrences.values() if count >= N_PRESSURE_POINTS)

    return stable_count, tracked_peaks_dict


# ==================== User Interaction for Crystal System Selection ====================
def select_crystal_system(label=""):
    print(f"\nSelect crystal system for {label}:")
    print("[1] Face-Centered Cubic (FCC)")
    print("[2] Body-Centered Cubic (BCC)")
    print("[3] Simple Cubic (SC)")
    print("[4] Hexagonal Close-Packed (HCP)")
    print("[5] Tetragonal")
    print("[6] Orthorhombic")
    print("[7] Monoclinic")
    print("[8] Triclinic")

    choice = input("Enter your choice (1-8): ").strip()

    mapping = {
        "1": "cubic_FCC",
        "2": "cubic_BCC",
        "3": "cubic_SC",
        "4": "Hexagonal",
        "5": "Tetragonal",
        "6": "Orthorhombic",
        "7": "Monoclinic",
        "8": "Triclinic"
    }

    selected = mapping.get(choice)
    if selected:
        print(f"✓ Selected crystal system: {CRYSTAL_SYSTEMS[selected]['name']}")
        return selected
    else:
        print("⚠️ Invalid choice, defaulting to 'cubic_FCC'")
        return "cubic_FCC"

# ==================== Lattice Parameter Fitting ====================

def fit_lattice_parameters_cubic(peak_dataset, crystal_system_key):
    """
    Fit lattice parameters for cubic crystal systems

    Parameters:
        peak_dataset: dict, {pressure: {'original_peaks': [...]} or [peak_list]}
        crystal_system_key: Crystal system key

    Returns:
        results: dict containing lattice parameters and volumes for each pressure
    """
    results = {}
    hkl_list = CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
    atoms_per_cell = CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

    for pressure, data in peak_dataset.items():
        # Extract peak list
        if isinstance(data, dict):
            peaks = data.get('original_peaks', data.get('new_peaks', []))
        else:
            peaks = data

        if len(peaks) == 0:
            continue

        # Convert 2theta to d-spacing
        d_obs = [two_theta_to_d(peak) for peak in peaks]

        # Match peaks to hkl indices
        num_peaks = min(len(peaks), len(hkl_list))
        matched_hkl = hkl_list[:num_peaks]

        # Fitting function
        def residuals(params):
            a = params[0]
            errors = []
            for i, hkl in enumerate(matched_hkl):
                d_calc = calculate_d_cubic(hkl, a)
                errors.append(d_obs[i] - d_calc)
            return errors

        # Initial guess for lattice parameter
        a_init = d_obs[0] * np.sqrt(sum(x**2 for x in matched_hkl[0]))

        # Perform least squares fitting
        result = least_squares(residuals, [a_init], bounds=([0], [np.inf]))
        a_fitted = result.x[0]

        # Calculate unit cell volume
        V_cell = calculate_cell_volume_cubic(a_fitted)

        # Calculate average atomic volume
        V_atomic = V_cell / atoms_per_cell

        results[pressure] = {
            'a': a_fitted,
            'V_cell': V_cell,
            'V_atomic': V_atomic,
            'num_peaks_used': num_peaks
        }

        print(f"Pressure: {pressure:.2f} GPa")
        print(f"  Lattice parameter a = {a_fitted:.6f} Å")
        print(f"  Unit cell volume V = {V_cell:.6f} Å³")
        print(f"  Average atomic volume = {V_atomic:.6f} Å³/atom")

    return results


def fit_lattice_parameters_hexagonal(peak_dataset, crystal_system_key):
    """
    Fit lattice parameters for hexagonal crystal systems

    Parameters:
        peak_dataset: dict, {pressure: {'original_peaks': [...]} or [peak_list]}
        crystal_system_key: Crystal system key

    Returns:
        results: dict containing lattice parameters and volumes for each pressure
    """
    results = {}
    hkl_list = CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
    atoms_per_cell = CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

    for pressure, data in peak_dataset.items():
        # Extract peak list
        if isinstance(data, dict):
            peaks = data.get('original_peaks', data.get('new_peaks', []))
        else:
            peaks = data

        if len(peaks) < 2:  # Hexagonal requires at least 2 peaks
            continue

        # Convert 2theta to d-spacing
        d_obs = [two_theta_to_d(peak) for peak in peaks]

        # Match peaks to hkl indices
        num_peaks = min(len(peaks), len(hkl_list))
        matched_hkl = hkl_list[:num_peaks]

        # Fitting function
        def residuals(params):
            a, c = params
            errors = []
            for i, hkl in enumerate(matched_hkl):
                d_calc = calculate_d_hexagonal(hkl, a, c)
                errors.append(d_obs[i] - d_calc)
            return errors

        # Initial guess for lattice parameters
        a_init = 3.0
        c_init = 5.0

        # Perform least squares fitting
        result = least_squares(residuals, [a_init, c_init], bounds=([0, 0], [np.inf, np.inf]))
        a_fitted, c_fitted = result.x

        # Calculate unit cell volume
        V_cell = calculate_cell_volume_hexagonal(a_fitted, c_fitted)

        # Calculate average atomic volume
        V_atomic = V_cell / atoms_per_cell

        results[pressure] = {
            'a': a_fitted,
            'c': c_fitted,
            'c/a': c_fitted / a_fitted,
            'V_cell': V_cell,
            'V_atomic': V_atomic,
            'num_peaks_used': num_peaks
        }

        print(f"Pressure: {pressure:.2f} GPa")
        print(f"  Lattice parameter a = {a_fitted:.6f} Å")
        print(f"  Lattice parameter c = {c_fitted:.6f} Å")
        print(f"  c/a ratio = {c_fitted/a_fitted:.6f}")
        print(f"  Unit cell volume V = {V_cell:.6f} Å³")
        print(f"  Average atomic volume = {V_atomic:.6f} Å³/atom")

    return results


def fit_lattice_parameters_tetragonal(peak_dataset, crystal_system_key):
    """
    Fit lattice parameters for tetragonal crystal systems

    Parameters:
        peak_dataset: dict, {pressure: {'original_peaks': [...]} or [peak_list]}
        crystal_system_key: Crystal system key

    Returns:
        results: dict containing lattice parameters and volumes for each pressure
    """
    results = {}
    hkl_list = CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
    atoms_per_cell = CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

    for pressure, data in peak_dataset.items():
        # Extract peak list
        if isinstance(data, dict):
            peaks = data.get('original_peaks', data.get('new_peaks', []))
        else:
            peaks = data

        if len(peaks) < 2:  # Tetragonal requires at least 2 peaks
            continue

        # Convert 2theta to d-spacing
        d_obs = [two_theta_to_d(peak) for peak in peaks]

        # Match peaks to hkl indices
        num_peaks = min(len(peaks), len(hkl_list))
        matched_hkl = hkl_list[:num_peaks]

        # Fitting function
        def residuals(params):
            a, c = params
            errors = []
            for i, hkl in enumerate(matched_hkl):
                d_calc = calculate_d_tetragonal(hkl, a, c)
                errors.append(d_obs[i] - d_calc)
            return errors

        # Initial guess
        a_init = 3.0
        c_init = 4.0

        # Perform least squares fitting
        result = least_squares(residuals, [a_init, c_init], bounds=([0, 0], [np.inf, np.inf]))
        a_fitted, c_fitted = result.x

        # Calculate unit cell volume
        V_cell = calculate_cell_volume_tetragonal(a_fitted, c_fitted)

        # Calculate average atomic volume
        V_atomic = V_cell / atoms_per_cell

        results[pressure] = {
            'a': a_fitted,
            'c': c_fitted,
            'c/a': c_fitted / a_fitted,
            'V_cell': V_cell,
            'V_atomic': V_atomic,
            'num_peaks_used': num_peaks
        }

        print(f"Pressure: {pressure:.2f} GPa")
        print(f"  Lattice parameter a = {a_fitted:.6f} Å")
        print(f"  Lattice parameter c = {c_fitted:.6f} Å")
        print(f"  c/a ratio = {c_fitted/a_fitted:.6f}")
        print(f"  Unit cell volume V = {V_cell:.6f} Å³")
        print(f"  Average atomic volume = {V_atomic:.6f} Å³/atom")

    return results


def fit_lattice_parameters_orthorhombic(peak_dataset, crystal_system_key):
    """
    Fit lattice parameters for orthorhombic crystal systems

    Parameters:
        peak_dataset: dict, {pressure: {'original_peaks': [...]} or [peak_list]}
        crystal_system_key: Crystal system key

    Returns:
        results: dict containing lattice parameters and volumes for each pressure
    """
    results = {}
    hkl_list = CRYSTAL_SYSTEMS[crystal_system_key]['hkl_list']
    atoms_per_cell = CRYSTAL_SYSTEMS[crystal_system_key]['atoms_per_cell']

    for pressure, data in peak_dataset.items():
        # Extract peak list
        if isinstance(data, dict):
            peaks = data.get('original_peaks', data.get('new_peaks', []))
        else:
            peaks = data

        if len(peaks) < 3:  # Orthorhombic requires at least 3 peaks
            continue

        # Convert 2theta to d-spacing
        d_obs = [two_theta_to_d(peak) for peak in peaks]

        # Match peaks to hkl indices
        num_peaks = min(len(peaks), len(hkl_list))
        matched_hkl = hkl_list[:num_peaks]

        # Fitting function
        def residuals(params):
            a, b, c = params
            errors = []
            for i, hkl in enumerate(matched_hkl):
                d_calc = calculate_d_orthorhombic(hkl, a, b, c)
                errors.append(d_obs[i] - d_calc)
            return errors

        # Initial guess
        a_init = 3.0
        b_init = 4.0
        c_init = 5.0

        # Perform least squares fitting
        result = least_squares(residuals, [a_init, b_init, c_init],
                             bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        a_fitted, b_fitted, c_fitted = result.x

        # Calculate unit cell volume
        V_cell = calculate_cell_volume_orthorhombic(a_fitted, b_fitted, c_fitted)

        # Calculate average atomic volume
        V_atomic = V_cell / atoms_per_cell

        results[pressure] = {
            'a': a_fitted,
            'b': b_fitted,
            'c': c_fitted,
            'V_cell': V_cell,
            'V_atomic': V_atomic,
            'num_peaks_used': num_peaks
        }

        print(f"Pressure: {pressure:.2f} GPa")
        print(f"  Lattice parameter a = {a_fitted:.6f} Å")
        print(f"  Lattice parameter b = {b_fitted:.6f} Å")
        print(f"  Lattice parameter c = {c_fitted:.6f} Å")
        print(f"  Unit cell volume V = {V_cell:.6f} Å³")
        print(f"  Average atomic volume = {V_atomic:.6f} Å³/atom")

    return results


# ==================== Main Fitting Function ====================

def fit_lattice_parameters(peak_dataset, crystal_system_key):
    """
    Main function to fit lattice parameters based on crystal system

    Parameters:
        peak_dataset: Peak dataset dictionary
        crystal_system_key: Crystal system key

    Returns:
        results: Fitting results including lattice parameters and volumes
    """
    system_type = crystal_system_key.split('_')[0]  # Extract system type (cubic, hexagonal, etc.)

    print(f"\n{'='*60}")
    print(f"Fitting Lattice Parameters for {CRYSTAL_SYSTEMS[crystal_system_key]['name']}")
    print(f"{'='*60}\n")

    if system_type == 'cubic':
        return fit_lattice_parameters_cubic(peak_dataset, crystal_system_key)
    elif crystal_system_key == 'Hexagonal':
        return fit_lattice_parameters_hexagonal(peak_dataset, crystal_system_key)
    elif crystal_system_key == 'Tetragonal':
        return fit_lattice_parameters_tetragonal(peak_dataset, crystal_system_key)
    elif crystal_system_key == 'Orthorhombic':
        return fit_lattice_parameters_orthorhombic(peak_dataset, crystal_system_key)
    else:
        print(f"Warning: Fitting for {crystal_system_key} not yet implemented")
        return {}


# ==================== Save Results to CSV ====================

def save_lattice_results_to_csv(results, filename, crystal_system_key):
    """
    Save lattice parameter fitting results to CSV file

    Parameters:
        results: Fitting results dictionary
        filename: Output CSV filename
        crystal_system_key: Crystal system key
    """
    if not results:
        print("No results to save.")
        return

    # Prepare data for DataFrame
    data_rows = []
    for pressure, params in sorted(results.items()):
        row = {'Pressure (GPa)': pressure}
        row.update(params)
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"\n✓ Results saved to: {filename}")


# ==================== Example Main Function ====================

def main():
    """
    Main function demonstrating the complete workflow:
    1. Read pressure-peak data from CSV
    2. Identify phase transition point
    3. Separate new peaks and original peaks
    4. Allow user to select crystal systems for both datasets
    5. Fit lattice parameters and calculate atomic volumes
    6. Save results
    """

    # Step 1: Read data from CSV
    print("\n" + "="*60)
    print("X-RAY DIFFRACTION ANALYSIS - PHASE TRANSITION & LATTICE FITTING")
    print("="*60)

    csv_path = r'D:\HEPS\ID31\dioptas_data\Al0\fit_output\all_results.csv'

    try:
        pressure_data = read_pressure_peak_data(csv_path)
        print(f"\n✓ Successfully read data from {csv_path}")
        print(f"  Total pressure points: {len(pressure_data)}")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    # Step 2: Identify phase transition
    print("\n" + "="*60)
    print("PHASE TRANSITION IDENTIFICATION")
    print("="*60)

    transition_pressure, before_pressures, after_pressures = find_phase_transition_point(
        pressure_data, tolerance=PEAK_TOLERANCE_1
    )

    if transition_pressure is None:
        print("\nNo phase transition detected. Analyzing as single phase...")
        print("\nPlease select crystal system for the entire dataset:")
        system_key = select_crystal_system()

        # Fit lattice parameters for all data
        all_data_dict = {p: peaks for p, peaks in pressure_data.items()}
        results = fit_lattice_parameters(all_data_dict, system_key)

        # Save results
        output_filename = csv_path.replace('.csv', '_lattice_results.csv')
        save_lattice_results_to_csv(results, output_filename, system_key)

        return

    # Step 3: Collect new peaks and original peaks
    print("\n" + "="*60)
    print("COLLECTING NEW PEAKS AND ORIGINAL PEAKS")
    print("="*60)

    # Get new peaks from transition point
    transition_peaks = pressure_data[transition_pressure]
    prev_pressure = before_pressures[-1]
    prev_peaks = pressure_data[prev_pressure]

    # Identify new peaks at transition point
    tolerance_windows = [(p - PEAK_TOLERANCE_1, p + PEAK_TOLERANCE_1) for p in prev_peaks]
    new_peaks_at_transition = []

    for peak in transition_peaks:
        in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
        if not in_any_window:
            new_peaks_at_transition.append(peak)

    print(f"\nNew peaks detected at transition: {len(new_peaks_at_transition)}")
    print(f"Positions: {[f'{p:.3f}' for p in new_peaks_at_transition]}")

    # Track new peaks across all after-transition pressures
    stable_count, tracked_new_peaks = collect_tracked_new_peaks(
        pressure_data, transition_pressure, after_pressures,
        new_peaks_at_transition, tolerance=PEAK_TOLERANCE_2
    )

    print(f"\nStable new peaks (appearing in ≥{N_PRESSURE_POINTS} pressure points): {stable_count}")

    # Build original peak dataset
    original_peak_dataset = build_original_peak_dataset(
        pressure_data, tracked_new_peaks, tolerance=PEAK_TOLERANCE_3
    )

    print(f"\nOriginal peaks dataset constructed for {len(original_peak_dataset)} pressure points")

    # Step 4: Select crystal systems
    print("\n" + "="*60)
    print("CRYSTAL SYSTEM SELECTION")
    print("="*60)

    original_system = select_crystal_system("ORIGINAL PEAKS (before transition)")
    new_system = select_crystal_system("NEW PEAKS (after transition)")

    # Step 5: Fit lattice parameters
    print("\n" + "="*60)
    print("FITTING LATTICE PARAMETERS")
    print("="*60)

    print("\n>>> FITTING ORIGINAL PEAKS <<<")
    original_results = fit_lattice_parameters(original_peak_dataset, original_system)

    print("\n>>> FITTING NEW PEAKS <<<")
    new_results = fit_lattice_parameters(tracked_new_peaks, new_system)

    # Step 6: Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    base_filename = csv_path.replace('.csv', '')

    original_output = f"{base_filename}_original_peaks_lattice.csv"
    save_lattice_results_to_csv(original_results, original_output, original_system)

    new_output = f"{base_filename}_new_peaks_lattice.csv"
    save_lattice_results_to_csv(new_results, new_output, new_system)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nSummary:")
    print(f"  - Phase transition pressure: {transition_pressure:.2f} GPa")
    print(f"  - Original peaks crystal system: {CRYSTAL_SYSTEMS[original_system]['name']}")
    print(f"  - New peaks crystal system: {CRYSTAL_SYSTEMS[new_system]['name']}")
    print(f"  - Original peaks results saved to: {original_output}")
    print(f"  - New peaks results saved to: {new_output}")
    print("\n" + "="*60 + "\n")

    #r'D:\HEPS\ID31\dioptas_data\Al0\fit_output\all_results'

if __name__ == "__main__":
    main()
