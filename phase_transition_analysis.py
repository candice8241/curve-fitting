# -*- coding: utf-8 -*-
"""
Phase Transition Analysis Script
Created for crystal structure identification and EOS fitting
Author: candicewang928@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
PEAK_TOLERANCE_1 = 0.3  # GPa tolerance for initial new peak detection
PEAK_TOLERANCE_2 = 0.2  # GPa tolerance for stable new peak counting
PEAK_TOLERANCE_3 = 0.15  # GPa tolerance for tracking new peaks after transition
NUM_STABLE_POINTS = 4  # Number of pressure points to confirm stable new peaks
WAVELENGTH = 0.6199  # X-ray wavelength in Angstroms (adjust as needed)

# ==================== CRYSTAL SYSTEM DATABASE ====================
# Each crystal system with corresponding hkl indices ordered by increasing 2theta
CRYSTAL_SYSTEMS = {
    'cubic': {
        'min_peaks': 1,
        'atoms_per_cell': {
            'fcc': 4,
            'bcc': 2,
            'sc': 1
        },
        'hkl_sequence': [
            (1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2),
            (4,0,0), (3,3,1), (4,2,0), (4,2,2), (5,1,1)
        ]
    },
    'hexagonal': {
        'min_peaks': 2,
        'atoms_per_cell': {
            'hcp': 2,
            'hex': 1
        },
        'hkl_sequence': [
            (1,0,0), (0,0,2), (1,0,1), (1,0,2), (1,1,0),
            (1,0,3), (2,0,0), (1,1,2), (2,0,1), (0,0,4),
            (2,0,2), (1,0,4), (2,0,3), (2,1,0), (2,1,1)
        ]
    },
    'tetragonal': {
        'min_peaks': 2,
        'atoms_per_cell': {
            'primitive': 1,
            'body_centered': 2
        },
        'hkl_sequence': [
            (1,0,0), (1,1,0), (0,0,1), (1,0,1), (1,1,1),
            (2,0,0), (2,1,0), (0,0,2), (2,1,1), (2,2,0),
            (1,0,2), (2,0,1), (3,1,0), (2,2,1), (3,1,1)
        ]
    },
    'orthorhombic': {
        'min_peaks': 3,
        'atoms_per_cell': {
            'primitive': 1,
            'base_centered': 2,
            'body_centered': 2,
            'face_centered': 4
        },
        'hkl_sequence': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (2,0,0), (1,1,1), (2,1,0), (2,0,1),
            (0,2,0), (1,2,0), (0,0,2), (2,1,1), (1,0,2)
        ]
    },
    'rhombohedral': {
        'min_peaks': 2,
        'atoms_per_cell': {
            'primitive': 1
        },
        'hkl_sequence': [
            (1,0,0), (1,0,1), (1,1,0), (0,0,3), (1,1,1),
            (2,0,0), (2,0,1), (1,0,2), (2,1,0), (2,1,1),
            (3,0,0), (1,1,2), (3,0,1), (2,0,2), (2,2,0)
        ]
    },
    'monoclinic': {
        'min_peaks': 4,
        'atoms_per_cell': {
            'primitive': 1,
            'base_centered': 2
        },
        'hkl_sequence': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (-1,0,1), (0,1,1), (1,1,1), (2,0,0), (1,-1,1),
            (2,1,0), (0,2,0), (2,0,1), (1,2,0), (2,1,1)
        ]
    },
    'triclinic': {
        'min_peaks': 6,
        'atoms_per_cell': {
            'primitive': 1
        },
        'hkl_sequence': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (1,1,1), (-1,1,0), (1,-1,0), (-1,0,1),
            (1,0,-1), (0,-1,1), (0,1,-1), (2,0,0), (0,2,0)
        ]
    }
}

# ==================== DATA STRUCTURES ====================
@dataclass
class PressurePoint:
    """Data structure for a single pressure point"""
    pressure: float
    peak_positions: np.ndarray

@dataclass
class PhaseInfo:
    """Information about a crystallographic phase"""
    crystal_system: str
    lattice_params: Dict[str, float]
    pressure_range: Tuple[float, float]
    peak_positions: List[np.ndarray]
    pressures: List[float]
    volumes: List[float]
    is_new_phase: bool = False

# ==================== UTILITY FUNCTIONS ====================
def twotheta_to_d(twotheta_deg: float, wavelength: float = WAVELENGTH) -> float:
    """Convert 2theta (degrees) to d-spacing (Angstroms)"""
    theta_rad = np.radians(twotheta_deg / 2)
    return wavelength / (2 * np.sin(theta_rad))

def d_to_twotheta(d: float, wavelength: float = WAVELENGTH) -> float:
    """Convert d-spacing to 2theta (degrees)"""
    return 2 * np.degrees(np.arcsin(wavelength / (2 * d)))

# ==================== PEAK MATCHING FUNCTIONS ====================
def find_matching_peaks(peaks1: np.ndarray, peaks2: np.ndarray,
                       tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find matching and non-matching peaks between two sets

    Returns:
        matched_indices: indices in peaks1 that match peaks2
        new_indices: indices in peaks2 that don't match peaks1
    """
    matched = []
    new_peaks = []

    for i, p2 in enumerate(peaks2):
        found = False
        for p1 in peaks1:
            if abs(p2 - p1) <= tolerance:
                found = True
                break
        if found:
            matched.append(i)
        else:
            new_peaks.append(i)

    return np.array(matched), np.array(new_peaks)

def identify_phase_transition(pressure_data: List[PressurePoint],
                              tolerance1: float = PEAK_TOLERANCE_1,
                              tolerance2: float = PEAK_TOLERANCE_2,
                              num_stable: int = NUM_STABLE_POINTS) -> Optional[int]:
    """
    Identify the pressure point where phase transition occurs

    Returns:
        Index of transition point, or None if no transition detected
    """
    for i in range(1, len(pressure_data)):
        prev_peaks = pressure_data[i-1].peak_positions
        curr_peaks = pressure_data[i].peak_positions

        _, new_peak_indices = find_matching_peaks(prev_peaks, curr_peaks, tolerance1)

        if len(new_peak_indices) > 0:
            # Check stability in next few points
            if i + num_stable <= len(pressure_data):
                new_peak_counts = [len(new_peak_indices)]

                for j in range(i+1, min(i+num_stable, len(pressure_data))):
                    _, new_peaks_j = find_matching_peaks(
                        pressure_data[i].peak_positions,
                        pressure_data[j].peak_positions,
                        tolerance2
                    )
                    new_peak_counts.append(len(new_peaks_j))

                # Check if new peak count stabilizes
                if len(set(new_peak_counts[-3:])) == 1:  # Last 3 counts are same
                    print(f"✓ Phase transition detected at P = {pressure_data[i].pressure:.2f} GPa")
                    print(f"  New peak evolution: {new_peak_counts}")
                    return i

    return None

# ==================== LATTICE PARAMETER CALCULATION ====================
def calculate_cubic_lattice(d_spacings: np.ndarray, hkl_list: List[Tuple]) -> Optional[float]:
    """Calculate cubic lattice parameter a"""
    if len(d_spacings) == 0:
        return None

    a_values = []
    for d, (h, k, l) in zip(d_spacings, hkl_list):
        hkl_sum = h**2 + k**2 + l**2
        if hkl_sum > 0:
            a = d * np.sqrt(hkl_sum)
            a_values.append(a)

    if len(a_values) == 0:
        return None

    return np.mean(a_values)

def calculate_hexagonal_lattice(d_spacings: np.ndarray,
                                hkl_list: List[Tuple]) -> Optional[Dict[str, float]]:
    """Calculate hexagonal lattice parameters a and c"""
    if len(d_spacings) < 2:
        return None

    # For hexagonal: 1/d² = 4/3 * (h²+hk+k²)/a² + l²/c²
    # Need at least 2 peaks to solve for a and c

    def residual(params):
        a, c = params
        if a <= 0 or c <= 0:
            return 1e10
        error = 0
        for d, (h, k, l) in zip(d_spacings, hkl_list):
            d_calc_sq_inv = (4/3) * (h**2 + h*k + k**2) / a**2 + l**2 / c**2
            d_calc = 1 / np.sqrt(d_calc_sq_inv)
            error += (d - d_calc)**2
        return error

    # Initial guess
    a0 = d_spacings[0] * 2
    c0 = d_spacings[-1] * 2

    result = minimize(residual, [a0, c0], method='Nelder-Mead')

    if result.success:
        a, c = result.x
        return {'a': a, 'c': c, 'c/a': c/a}

    return None

def calculate_tetragonal_lattice(d_spacings: np.ndarray,
                                 hkl_list: List[Tuple]) -> Optional[Dict[str, float]]:
    """Calculate tetragonal lattice parameters a and c"""
    if len(d_spacings) < 2:
        return None

    # For tetragonal: 1/d² = (h²+k²)/a² + l²/c²

    def residual(params):
        a, c = params
        if a <= 0 or c <= 0:
            return 1e10
        error = 0
        for d, (h, k, l) in zip(d_spacings, hkl_list):
            d_calc_sq_inv = (h**2 + k**2) / a**2 + l**2 / c**2
            d_calc = 1 / np.sqrt(d_calc_sq_inv)
            error += (d - d_calc)**2
        return error

    a0 = d_spacings[0] * 2
    c0 = d_spacings[-1] * 2

    result = minimize(residual, [a0, c0], method='Nelder-Mead')

    if result.success:
        a, c = result.x
        return {'a': a, 'c': c, 'c/a': c/a}

    return None

def calculate_orthorhombic_lattice(d_spacings: np.ndarray,
                                   hkl_list: List[Tuple]) -> Optional[Dict[str, float]]:
    """Calculate orthorhombic lattice parameters a, b, c"""
    if len(d_spacings) < 3:
        return None

    # For orthorhombic: 1/d² = h²/a² + k²/b² + l²/c²

    def residual(params):
        a, b, c = params
        if a <= 0 or b <= 0 or c <= 0:
            return 1e10
        error = 0
        for d, (h, k, l) in zip(d_spacings, hkl_list):
            d_calc_sq_inv = h**2/a**2 + k**2/b**2 + l**2/c**2
            d_calc = 1 / np.sqrt(d_calc_sq_inv)
            error += (d - d_calc)**2
        return error

    a0, b0, c0 = d_spacings[0]*2, d_spacings[1]*2, d_spacings[2]*2

    result = minimize(residual, [a0, b0, c0], method='Nelder-Mead')

    if result.success:
        a, b, c = result.x
        return {'a': a, 'b': b, 'c': c}

    return None

def calculate_rhombohedral_lattice(d_spacings: np.ndarray,
                                   hkl_list: List[Tuple]) -> Optional[Dict[str, float]]:
    """Calculate rhombohedral lattice parameters a and alpha"""
    if len(d_spacings) < 2:
        return None

    # Simplified calculation - assume hexagonal setting
    # More complex calculation would require alpha angle

    def residual(params):
        a, alpha_deg = params
        if a <= 0 or alpha_deg <= 0 or alpha_deg >= 180:
            return 1e10
        alpha = np.radians(alpha_deg)
        error = 0
        for d, (h, k, l) in zip(d_spacings, hkl_list):
            # Rhombohedral metric
            sin_alpha = np.sin(alpha)
            cos_alpha = np.cos(alpha)

            term1 = (h**2 + k**2 + l**2) * sin_alpha**2
            term2 = 2 * (h*k + k*l + h*l) * (cos_alpha**2 - cos_alpha)
            d_calc_sq_inv = (term1 + term2) / (a**2 * (1 - 3*cos_alpha**2 + 2*cos_alpha**3))

            if d_calc_sq_inv <= 0:
                return 1e10

            d_calc = 1 / np.sqrt(d_calc_sq_inv)
            error += (d - d_calc)**2
        return error

    a0 = d_spacings[0] * 2
    alpha0 = 60  # Common for rhombohedral

    result = minimize(residual, [a0, alpha0], method='Nelder-Mead')

    if result.success:
        a, alpha = result.x
        return {'a': a, 'alpha': alpha}

    return None

# ==================== CRYSTAL SYSTEM IDENTIFICATION ====================
def identify_crystal_system(peak_positions: np.ndarray,
                           num_new_peaks: int = None,
                           wavelength: float = WAVELENGTH) -> Optional[PhaseInfo]:
    """
    Identify crystal system from peak positions

    Args:
        peak_positions: Array of 2theta positions
        num_new_peaks: If specified, only consider systems that can be solved with this many peaks
        wavelength: X-ray wavelength

    Returns:
        PhaseInfo object with best matching crystal system
    """
    d_spacings = np.array([twotheta_to_d(pos, wavelength) for pos in peak_positions])
    d_spacings = np.sort(d_spacings)[::-1]  # Sort in descending order

    best_match = None
    best_error = float('inf')

    for system_name, system_info in CRYSTAL_SYSTEMS.items():
        min_peaks = system_info['min_peaks']

        # Skip if we don't have enough peaks
        if len(peak_positions) < min_peaks:
            continue

        # If num_new_peaks specified, only consider systems that can be solved
        if num_new_peaks is not None:
            if min_peaks > num_new_peaks:
                continue

        hkl_sequence = system_info['hkl_sequence'][:len(d_spacings)]

        # Try to calculate lattice parameters
        lattice_params = None
        error = float('inf')

        if system_name == 'cubic':
            a = calculate_cubic_lattice(d_spacings, hkl_sequence)
            if a is not None:
                lattice_params = {'a': a}
                # Calculate error
                error = 0
                for d, (h, k, l) in zip(d_spacings, hkl_sequence):
                    d_calc = a / np.sqrt(h**2 + k**2 + l**2)
                    error += abs(d - d_calc) / d
                error /= len(d_spacings)

        elif system_name == 'hexagonal':
            params = calculate_hexagonal_lattice(d_spacings, hkl_sequence)
            if params is not None:
                lattice_params = params
                a, c = params['a'], params['c']
                error = 0
                for d, (h, k, l) in zip(d_spacings, hkl_sequence):
                    d_calc_sq_inv = (4/3) * (h**2 + h*k + k**2) / a**2 + l**2 / c**2
                    d_calc = 1 / np.sqrt(d_calc_sq_inv)
                    error += abs(d - d_calc) / d
                error /= len(d_spacings)

        elif system_name == 'tetragonal':
            params = calculate_tetragonal_lattice(d_spacings, hkl_sequence)
            if params is not None:
                lattice_params = params
                a, c = params['a'], params['c']
                error = 0
                for d, (h, k, l) in zip(d_spacings, hkl_sequence):
                    d_calc_sq_inv = (h**2 + k**2) / a**2 + l**2 / c**2
                    d_calc = 1 / np.sqrt(d_calc_sq_inv)
                    error += abs(d - d_calc) / d
                error /= len(d_spacings)

        elif system_name == 'orthorhombic':
            params = calculate_orthorhombic_lattice(d_spacings, hkl_sequence)
            if params is not None:
                lattice_params = params
                a, b, c = params['a'], params['b'], params['c']
                error = 0
                for d, (h, k, l) in zip(d_spacings, hkl_sequence):
                    d_calc_sq_inv = h**2/a**2 + k**2/b**2 + l**2/c**2
                    d_calc = 1 / np.sqrt(d_calc_sq_inv)
                    error += abs(d - d_calc) / d
                error /= len(d_spacings)

        elif system_name == 'rhombohedral':
            params = calculate_rhombohedral_lattice(d_spacings, hkl_sequence)
            if params is not None:
                lattice_params = params
                # Error calculation for rhombohedral
                error = 0.1  # Placeholder

        # Check if this is the best match
        if lattice_params is not None and error < best_error:
            best_error = error
            best_match = PhaseInfo(
                crystal_system=system_name,
                lattice_params=lattice_params,
                pressure_range=(0, 0),
                peak_positions=[],
                pressures=[],
                volumes=[]
            )

    return best_match

# ==================== VOLUME CALCULATIONS ====================
def calculate_unit_cell_volume(lattice_params: Dict[str, float],
                               crystal_system: str) -> float:
    """Calculate unit cell volume based on crystal system"""
    if crystal_system == 'cubic':
        a = lattice_params['a']
        return a**3

    elif crystal_system == 'hexagonal':
        a = lattice_params['a']
        c = lattice_params['c']
        return np.sqrt(3) / 2 * a**2 * c

    elif crystal_system == 'tetragonal':
        a = lattice_params['a']
        c = lattice_params['c']
        return a**2 * c

    elif crystal_system == 'orthorhombic':
        a = lattice_params['a']
        b = lattice_params['b']
        c = lattice_params['c']
        return a * b * c

    elif crystal_system == 'rhombohedral':
        a = lattice_params['a']
        alpha = np.radians(lattice_params.get('alpha', 60))
        return a**3 * np.sqrt(1 - 3*np.cos(alpha)**2 + 2*np.cos(alpha)**3)

    else:
        return 0.0

def calculate_volume_per_atom(volume: float, crystal_system: str,
                              structure_type: str = 'primitive') -> float:
    """Calculate volume per atom"""
    system_info = CRYSTAL_SYSTEMS.get(crystal_system, {})
    atoms_dict = system_info.get('atoms_per_cell', {'primitive': 1})

    atoms_per_cell = atoms_dict.get(structure_type, 1)

    return volume / atoms_per_cell

# ==================== EQUATION OF STATE (EOS) ====================
def birch_murnaghan_2nd(V: np.ndarray, V0: float, K0: float) -> np.ndarray:
    """
    2nd order Birch-Murnaghan equation of state

    P = (3/2) * K0 * [(V0/V)^(7/3) - (V0/V)^(5/3)]
    """
    eta = (V0 / V)**(2/3)
    P = (3/2) * K0 * (eta**(7/2) - eta**(5/2))
    return P

def birch_murnaghan_3rd(V: np.ndarray, V0: float, K0: float, K0p: float) -> np.ndarray:
    """
    3rd order Birch-Murnaghan equation of state

    P = (3/2) * K0 * [(V0/V)^(7/3) - (V0/V)^(5/3)] *
        {1 + (3/4) * (K0' - 4) * [(V0/V)^(2/3) - 1]}
    """
    eta = (V0 / V)**(2/3)
    P = (3/2) * K0 * (eta**(7/2) - eta**(5/2)) * \
        (1 + (3/4) * (K0p - 4) * (eta - 1))
    return P

def fit_eos_2nd_order(pressures: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
    """
    Fit 2nd order Birch-Murnaghan EOS

    Returns:
        Dictionary with V0 and K0
    """
    def residual(params, P, V):
        V0, K0 = params
        P_calc = birch_murnaghan_2nd(V, V0, K0)
        return np.sum((P - P_calc)**2)

    # Initial guesses
    V0_guess = np.max(volumes)
    K0_guess = 100  # GPa

    from scipy.optimize import minimize
    result = minimize(residual, [V0_guess, K0_guess], args=(pressures, volumes),
                     method='Nelder-Mead')

    if result.success:
        V0, K0 = result.x
        return {'V0': V0, 'K0': K0}
    else:
        return {'V0': V0_guess, 'K0': K0_guess}

def fit_eos_3rd_order(pressures: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
    """
    Fit 3rd order Birch-Murnaghan EOS

    Returns:
        Dictionary with V0, K0, and K0'
    """
    def residual(params, P, V):
        V0, K0, K0p = params
        P_calc = birch_murnaghan_3rd(V, V0, K0, K0p)
        return np.sum((P - P_calc)**2)

    # Initial guesses
    V0_guess = np.max(volumes)
    K0_guess = 100  # GPa
    K0p_guess = 4.0  # Typical value

    from scipy.optimize import minimize
    result = minimize(residual, [V0_guess, K0_guess, K0p_guess],
                     args=(pressures, volumes),
                     method='Nelder-Mead',
                     bounds=[(V0_guess*0.8, V0_guess*1.2),
                            (10, 500),
                            (2, 10)])

    if result.success:
        V0, K0, K0p = result.x
        return {'V0': V0, 'K0': K0, "K0'": K0p}
    else:
        return {'V0': V0_guess, 'K0': K0_guess, "K0'": K0p_guess}

# ==================== MAIN ANALYSIS FUNCTION ====================
def analyze_phase_transition(csv_file: str, output_dir: str = "./output"):
    """
    Main function to analyze phase transitions from CSV file

    CSV format expected:
    - 'file' column: pressure in GPa
    - 'center' column: peak positions (2theta)
    - Different pressure points separated by empty rows
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("PHASE TRANSITION ANALYSIS")
    print("="*60)

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Parse data into pressure points
    pressure_data = []
    current_pressure = None
    current_peaks = []

    for idx, row in df.iterrows():
        # Check if row is empty (NaN in file column indicates separator)
        if pd.isna(row['file']):
            if current_pressure is not None and len(current_peaks) > 0:
                pressure_data.append(PressurePoint(
                    pressure=current_pressure,
                    peak_positions=np.array(current_peaks)
                ))
                current_peaks = []
            current_pressure = None
        else:
            # Extract pressure and peak position
            pressure = float(row['file'])
            peak_pos = float(row['center'])

            if current_pressure is None:
                current_pressure = pressure

            current_peaks.append(peak_pos)

    # Don't forget the last group
    if current_pressure is not None and len(current_peaks) > 0:
        pressure_data.append(PressurePoint(
            pressure=current_pressure,
            peak_positions=np.array(current_peaks)
        ))

    # Sort by pressure
    pressure_data.sort(key=lambda x: x.pressure)

    print(f"\n✓ Loaded {len(pressure_data)} pressure points")
    print(f"  Pressure range: {pressure_data[0].pressure:.2f} - {pressure_data[-1].pressure:.2f} GPa")

    # Identify phase transition point
    transition_idx = identify_phase_transition(pressure_data)

    if transition_idx is None:
        print("\n⚠ No phase transition detected. Analyzing as single phase.")
        transition_idx = len(pressure_data)  # Treat all as original phase

    # Analyze original phase (before transition)
    print(f"\n{'='*60}")
    print("ORIGINAL PHASE ANALYSIS")
    print(f"{'='*60}")

    original_peaks = pressure_data[0].peak_positions
    original_phase = identify_crystal_system(original_peaks)

    if original_phase:
        print(f"✓ Crystal system: {original_phase.crystal_system.upper()}")
        print(f"  Lattice parameters: {original_phase.lattice_params}")

        # Calculate volumes for all points in original phase
        original_pressures = []
        original_volumes = []

        for i in range(transition_idx):
            p_point = pressure_data[i]
            phase_info = identify_crystal_system(p_point.peak_positions)
            if phase_info:
                vol = calculate_unit_cell_volume(phase_info.lattice_params,
                                                 phase_info.crystal_system)
                vol_per_atom = calculate_volume_per_atom(vol, phase_info.crystal_system)

                original_pressures.append(p_point.pressure)
                original_volumes.append(vol_per_atom)

        # Fit EOS for original phase
        if len(original_pressures) >= 3:
            print(f"\n  Fitting EOS for original phase ({len(original_pressures)} points)...")

            P_orig = np.array(original_pressures)
            V_orig = np.array(original_volumes)

            # 2nd order fit
            eos_2nd = fit_eos_2nd_order(P_orig, V_orig)
            print(f"\n  2nd Order BM EOS:")
            print(f"    V0 = {eos_2nd['V0']:.4f} Å³/atom")
            print(f"    K0 = {eos_2nd['K0']:.2f} GPa")

            # 3rd order fit
            eos_3rd = fit_eos_3rd_order(P_orig, V_orig)
            print(f"\n  3rd Order BM EOS:")
            print(f"    V0 = {eos_3rd['V0']:.4f} Å³/atom")
            print(f"    K0 = {eos_3rd['K0']:.2f} GPa")
            print(f"    K0' = {eos_3rd[\"K0'\"]:.3f}")

            # Plot original phase P-V curve
            plt.figure(figsize=(10, 6))
            plt.scatter(V_orig, P_orig, s=100, c='blue', marker='o',
                       label='Original Phase Data', zorder=3)

            V_fit = np.linspace(V_orig.min()*0.9, V_orig.max()*1.1, 200)
            P_fit_2nd = birch_murnaghan_2nd(V_fit, eos_2nd['V0'], eos_2nd['K0'])
            P_fit_3rd = birch_murnaghan_3rd(V_fit, eos_3rd['V0'],
                                           eos_3rd['K0'], eos_3rd["K0'"])

            plt.plot(V_fit, P_fit_2nd, 'r--', linewidth=2, label='2nd Order BM')
            plt.plot(V_fit, P_fit_3rd, 'g-', linewidth=2, label='3rd Order BM')

            plt.xlabel('Volume (Å³/atom)', fontsize=12)
            plt.ylabel('Pressure (GPa)', fontsize=12)
            plt.title(f'Original Phase: {original_phase.crystal_system.upper()}', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'original_phase_PV.png'), dpi=300)
            plt.close()
            print(f"\n  ✓ P-V curve saved: original_phase_PV.png")

    # Analyze new phase (after transition)
    if transition_idx < len(pressure_data):
        print(f"\n{'='*60}")
        print("NEW PHASE ANALYSIS")
        print(f"{'='*60}")

        # Identify new peaks at transition point
        transition_peaks = pressure_data[transition_idx].peak_positions
        prev_peaks = pressure_data[transition_idx-1].peak_positions

        _, new_peak_indices = find_matching_peaks(prev_peaks, transition_peaks,
                                                   PEAK_TOLERANCE_2)

        num_new_peaks = len(new_peak_indices)
        new_peaks_only = transition_peaks[new_peak_indices]

        print(f"  Number of new peaks: {num_new_peaks}")
        print(f"  New peak positions: {new_peaks_only}")

        # Identify crystal system using only new peaks
        if num_new_peaks > 0:
            new_phase = identify_crystal_system(new_peaks_only, num_new_peaks)

            if new_phase:
                print(f"\n✓ New phase crystal system: {new_phase.crystal_system.upper()}")
                print(f"  Lattice parameters: {new_phase.lattice_params}")

                # Calculate volumes for new phase
                new_pressures = []
                new_volumes = []

                for i in range(transition_idx, len(pressure_data)):
                    p_point = pressure_data[i]

                    # Extract new peaks from this pressure point
                    _, curr_new_indices = find_matching_peaks(
                        prev_peaks, p_point.peak_positions, PEAK_TOLERANCE_3
                    )

                    if len(curr_new_indices) > 0:
                        curr_new_peaks = p_point.peak_positions[curr_new_indices]
                        phase_info = identify_crystal_system(curr_new_peaks, num_new_peaks)

                        if phase_info:
                            vol = calculate_unit_cell_volume(phase_info.lattice_params,
                                                             phase_info.crystal_system)
                            vol_per_atom = calculate_volume_per_atom(vol,
                                                                     phase_info.crystal_system)

                            new_pressures.append(p_point.pressure)
                            new_volumes.append(vol_per_atom)

                # Fit EOS for new phase
                if len(new_pressures) >= 3:
                    print(f"\n  Fitting EOS for new phase ({len(new_pressures)} points)...")

                    P_new = np.array(new_pressures)
                    V_new = np.array(new_volumes)

                    # 2nd order fit
                    eos_2nd_new = fit_eos_2nd_order(P_new, V_new)
                    print(f"\n  2nd Order BM EOS:")
                    print(f"    V0 = {eos_2nd_new['V0']:.4f} Å³/atom")
                    print(f"    K0 = {eos_2nd_new['K0']:.2f} GPa")

                    # 3rd order fit
                    eos_3rd_new = fit_eos_3rd_order(P_new, V_new)
                    print(f"\n  3rd Order BM EOS:")
                    print(f"    V0 = {eos_3rd_new['V0']:.4f} Å³/atom")
                    print(f"    K0 = {eos_3rd_new['K0']:.2f} GPa")
                    print(f"    K0' = {eos_3rd_new[\"K0'\"]:.3f}")

                    # Plot new phase P-V curve
                    plt.figure(figsize=(10, 6))
                    plt.scatter(V_new, P_new, s=100, c='red', marker='s',
                               label='New Phase Data', zorder=3)

                    V_fit_new = np.linspace(V_new.min()*0.9, V_new.max()*1.1, 200)
                    P_fit_2nd_new = birch_murnaghan_2nd(V_fit_new,
                                                        eos_2nd_new['V0'],
                                                        eos_2nd_new['K0'])
                    P_fit_3rd_new = birch_murnaghan_3rd(V_fit_new,
                                                        eos_3rd_new['V0'],
                                                        eos_3rd_new['K0'],
                                                        eos_3rd_new["K0'"])

                    plt.plot(V_fit_new, P_fit_2nd_new, 'orange', linestyle='--',
                            linewidth=2, label='2nd Order BM')
                    plt.plot(V_fit_new, P_fit_3rd_new, 'darkred', linestyle='-',
                            linewidth=2, label='3rd Order BM')

                    plt.xlabel('Volume (Å³/atom)', fontsize=12)
                    plt.ylabel('Pressure (GPa)', fontsize=12)
                    plt.title(f'New Phase: {new_phase.crystal_system.upper()}', fontsize=14)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'new_phase_PV.png'), dpi=300)
                    plt.close()
                    print(f"\n  ✓ P-V curve saved: new_phase_PV.png")

                    # Combined plot
                    if len(original_pressures) >= 3:
                        plt.figure(figsize=(12, 7))

                        # Original phase
                        plt.scatter(V_orig, P_orig, s=100, c='blue', marker='o',
                                   label=f'Original ({original_phase.crystal_system})', zorder=3)
                        V_fit = np.linspace(V_orig.min()*0.9, V_orig.max()*1.1, 200)
                        P_fit_3rd = birch_murnaghan_3rd(V_fit, eos_3rd['V0'],
                                                        eos_3rd['K0'], eos_3rd["K0'"])
                        plt.plot(V_fit, P_fit_3rd, 'b-', linewidth=2,
                                label='Original 3rd Order BM')

                        # New phase
                        plt.scatter(V_new, P_new, s=100, c='red', marker='s',
                                   label=f'New ({new_phase.crystal_system})', zorder=3)
                        plt.plot(V_fit_new, P_fit_3rd_new, 'r-', linewidth=2,
                                label='New 3rd Order BM')

                        # Transition line
                        if transition_idx < len(pressure_data):
                            trans_P = pressure_data[transition_idx].pressure
                            plt.axhline(y=trans_P, color='gray', linestyle='--',
                                       linewidth=1.5, label=f'Transition ({trans_P:.1f} GPa)')

                        plt.xlabel('Volume (Å³/atom)', fontsize=12)
                        plt.ylabel('Pressure (GPa)', fontsize=12)
                        plt.title('Phase Transition: P-V Curves', fontsize=14, fontweight='bold')
                        plt.legend(fontsize=10)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'combined_PV.png'), dpi=300)
                        plt.close()
                        print(f"\n  ✓ Combined P-V curve saved: combined_PV.png")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Example usage
    csv_file = "peak_data.csv"  # Replace with your CSV file path
    output_dir = "./phase_analysis_output"

    analyze_phase_transition(csv_file, output_dir)
