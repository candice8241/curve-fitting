import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration Parameters ====================
WAVELENGTH = 0.4133  # X-ray wavelength (Å), can be modified according to actual conditions
PEAK_TOLERANCE_1 = 0.3  # Peak position tolerance for identifying phase transition points (degrees)
PEAK_TOLERANCE_2 = 0.3  # Tolerance for determining new peak count (degrees)
PEAK_TOLERANCE_3 = 0.02  # Tolerance for subsequent pressure points around new peaks (degrees)
N_PRESSURE_POINTS = 4  # Number of pressure points used to determine stable new peak count

# ==================== HKL Order Definition for Each Crystal System ====================
# Define hkl order from small to large 2theta for each crystal system (first 20)

CRYSTAL_SYSTEMS = {
    'cubic_fcc': {
        'name': 'Face-Centered Cubic (FCC)',
        'min_peaks': 1,
        'hkl_list': [
            (1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2),
            (4,0,0), (3,3,1), (4,2,0), (4,2,2), (3,3,3),
            (5,1,1), (4,4,0), (5,3,1), (6,0,0), (6,2,0),
            (5,3,3), (6,2,2), (4,4,4), (5,5,1), (6,4,0)
        ]
    },
    'cubic_bcc': {
        'name': 'Body-Centered Cubic (BCC)',
        'min_peaks': 1,
        'hkl_list': [
            (1,1,0), (2,0,0), (2,1,1), (2,2,0), (3,1,0),
            (2,2,2), (3,2,1), (4,0,0), (3,3,0), (4,1,1),
            (3,3,2), (4,2,0), (4,2,2), (3,3,3), (5,1,0),
            (4,3,1), (5,2,1), (4,4,0), (5,3,0), (6,0,0)
        ]
    },
    'cubic_sc': {
        'name': 'Simple Cubic (SC)',
        'min_peaks': 1,
        'hkl_list': [
            (1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0),
            (2,1,1), (2,2,0), (2,2,1), (3,0,0), (3,1,0),
            (3,1,1), (2,2,2), (3,2,0), (3,2,1), (4,0,0),
            (4,1,0), (3,3,0), (4,1,1), (3,3,1), (4,2,0)
        ]
    },
    'hexagonal': {
        'name': 'Hexagonal (HCP)',
        'min_peaks': 2,
        'hkl_list': [
            (1,0,0), (0,0,2), (1,0,1), (1,0,2), (1,1,0),
            (1,0,3), (2,0,0), (1,1,2), (2,0,1), (0,0,4),
            (2,0,2), (1,0,4), (2,0,3), (2,1,0), (2,1,1),
            (2,0,4), (2,1,2), (3,0,0), (2,1,3), (2,2,0)
        ]
    },
    'tetragonal': {
        'name': 'Tetragonal',
        'min_peaks': 2,
        'hkl_list': [
            (1,0,0), (0,0,1), (1,1,0), (1,0,1), (1,1,1),
            (2,0,0), (2,1,0), (0,0,2), (2,1,1), (2,0,1),
            (2,2,0), (2,1,2), (3,0,0), (2,2,1), (3,1,0),
            (2,0,2), (3,1,1), (2,2,2), (3,2,0), (3,0,1)
        ]
    },
    'orthorhombic': {
        'name': 'Orthorhombic',
        'min_peaks': 3,
        'hkl_list': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (1,1,1), (2,0,0), (2,1,0), (2,0,1),
            (1,2,0), (0,2,0), (1,2,1), (0,2,1), (2,1,1),
            (2,2,0), (2,0,2), (0,0,2), (2,2,1), (3,0,0)
        ]
    },
    'monoclinic': {
        'name': 'Monoclinic',
        'min_peaks': 4,
        'hkl_list': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (1,-1,0), (1,0,-1), (1,1,1), (2,0,0),
            (1,-1,1), (2,1,0), (0,2,0), (2,0,1), (1,2,0),
            (0,0,2), (2,1,1), (1,1,-1), (2,-1,0), (2,0,-1)
        ]
    },
    'triclinic': {
        'name': 'Triclinic',
        'min_peaks': 6,
        'hkl_list': [
            (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
            (0,1,1), (1,-1,0), (1,0,-1), (0,1,-1), (1,1,1),
            (1,-1,1), (1,1,-1), (2,0,0), (0,2,0), (0,0,2),
            (2,1,0), (2,0,1), (1,2,0), (0,2,1), (1,0,2)
        ]
    }
}

# ==================== Helper Functions ====================

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

# ==================== CSV Reading and Data Preprocessing ====================

def read_pressure_peak_data(csv_path):
    """
    Read CSV file and extract pressure point and peak position data

    Parameters:
        csv_path: CSV file path

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
        # Check if it's an empty row (separator)
        if pd.isna(row['File']) or row['File'] == '':
            current_pressure = None
            continue

        # Extract pressure value
        try:
            # Assume File column contains pressure info, format may be "filename_XXGPa" or just a number
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

    # Sort peaks for each pressure point
    for pressure in pressure_data:
        pressure_data[pressure] = sorted(pressure_data[pressure])

    return pressure_data

# ==================== Phase Transition Point Identification ====================

def find_phase_transition_point(pressure_data, tolerance=PEAK_TOLERANCE_1):
    """
    Identify phase transition point (interval method): Starting from the minimum pressure point,
    compare adjacent pressure points one by one. If a peak does not fall within the tolerance
    range of any peak from the previous pressure point, it is a new peak.

    Parameters:
        pressure_data: Pressure-peak position data dictionary
        tolerance: Peak position tolerance (unit: degrees)

    Returns:
        transition_pressure: Phase transition pressure point (GPa)
        before_pressures: List of pressure points before phase transition
        after_pressures: List of pressure points after phase transition
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

        # Build tolerance window list for previous pressure point peaks
        tolerance_windows = [(p - tolerance, p + tolerance) for p in prev_peaks]

        # Check if current point has new peaks
        has_new_peak = False
        for peak in curr_peaks:
            # Check if this peak falls within any tolerance window
            in_any_window = any(lower <= peak <= upper for (lower, upper) in tolerance_windows)
            if not in_any_window:
                has_new_peak = True
                break

        if has_new_peak:
            print(f"\n>>> Phase transition point found: {curr_pressure:.2f} GPa")
            return curr_pressure, sorted_pressures[:i], sorted_pressures[i:]

    print("\n>>> No obvious phase transition point found")
    return None, sorted_pressures, []



# ==================== New Peak Statistics ====================

def build_original_peak_dataset(pressure_data, new_peak_dataset, tolerance=PEAK_TOLERANCE_3):
    """
    Build original peak dataset (all pressure points) based on new peak dataset

    Parameters:
        pressure_data: dict, {pressure: [all peak positions]}
        new_peak_dataset: dict, {pressure: [new peak positions]} from collect_tracked_new_peaks
        tolerance: Tolerance range to determine if a peak belongs to new peaks

    Returns:
        original_peak_dataset: dict, {pressure: {'original_peaks': [...], 'count': x}}
    """
    original_peak_dataset = {}

    for pressure, all_peaks in pressure_data.items():
        new_peaks = new_peak_dataset.get(pressure, [])
        original_peaks = []

        for peak in all_peaks:
            # Check if it matches any new peak (falls within ±tolerance range)
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
    Track the existence of specified new peaks starting from phase transition pressure point
    (based on ±tolerance range matching)

    Parameters:
        pressure_data: dict, pressure → peak position list
        transition_pressure: Phase transition point, used as reference for new peaks
        after_pressures: List of pressure points after phase transition
        new_peaks_ref: List of new peaks found at phase transition point
        tolerance: Tolerance range (for matching determination)

    Returns:
        stable_count: Number of stable new peaks that appear ≥ N times (determined by N_PRESSURE_POINTS)
        new_peak_dataset: dict with pressure points as keys and matched new peak lists as values
    """
    new_peak_dataset = {}
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
            new_peak_dataset[pressure] = matched_peaks

    # Count "stable" new peaks (occurrence count ≥ N_PRESSURE_POINTS)
    stable_count = sum(1 for count in peak_occurrences.values() if count >= N_PRESSURE_POINTS)

    return stable_count, new_peak_dataset


# ==================== Crystal System Determination ====================

def fit_cubic_lattice(peaks_2theta, hkl_list):
    """
    Fit cubic crystal system lattice parameters

    Parameters:
        peaks_2theta: Peak position list (2theta, degrees)
        hkl_list: hkl index list

    Returns:
        a: Lattice parameter (Å)
        residual: Fitting residual
    """
    d_values = [two_theta_to_d(tt) for tt in peaks_2theta]

    def objective(params):
        a = params[0]
        residuals = []
        for i, (h, k, l) in enumerate(hkl_list[:len(d_values)]):
            d_calc = calculate_d_cubic((h, k, l), a)
            residuals.append((d_values[i] - d_calc)**2)
        return np.sum(residuals)

    # Initial guess
    a_init = d_values[0] * np.sqrt(hkl_list[0][0]**2 + hkl_list[0][1]**2 + hkl_list[0][2]**2)

    result = minimize(objective, [a_init], bounds=[(1.0, 20.0)])

    if result.success:
        return result.x[0], result.fun
    else:
        return None, float('inf')

def fit_hexagonal_lattice(peaks_2theta, hkl_list):
    """
    Fit hexagonal crystal system lattice parameters

    Parameters:
        peaks_2theta: Peak position list (2theta, degrees)
        hkl_list: hkl index list

    Returns:
        (a, c): Lattice parameters (Å)
        residual: Fitting residual
    """
    d_values = [two_theta_to_d(tt) for tt in peaks_2theta]

    def objective(params):
        a, c = params
        residuals = []
        for i, (h, k, l) in enumerate(hkl_list[:len(d_values)]):
            d_calc = calculate_d_hexagonal((h, k, l), a, c)
            residuals.append((d_values[i] - d_calc)**2)
        return np.sum(residuals)

    # Initial guess
    a_init = d_values[0] * 2.0
    c_init = d_values[0] * 3.0

    result = minimize(objective, [a_init, c_init],
                     bounds=[(1.0, 20.0), (1.0, 20.0)])

    if result.success:
        return tuple(result.x), result.fun
    else:
        return None, float('inf')

def fit_tetragonal_lattice(peaks_2theta, hkl_list):
    """
    Fit tetragonal crystal system lattice parameters

    Parameters:
        peaks_2theta: Peak position list (2theta, degrees)
        hkl_list: hkl index list

    Returns:
        (a, c): Lattice parameters (Å)
        residual: Fitting residual
    """
    d_values = [two_theta_to_d(tt) for tt in peaks_2theta]

    def objective(params):
        a, c = params
        residuals = []
        for i, (h, k, l) in enumerate(hkl_list[:len(d_values)]):
            d_calc = calculate_d_tetragonal((h, k, l), a, c)
            residuals.append((d_values[i] - d_calc)**2)
        return np.sum(residuals)

    # Initial guess
    a_init = d_values[0] * 2.0
    c_init = d_values[0] * 2.0

    result = minimize(objective, [a_init, c_init],
                     bounds=[(1.0, 20.0), (1.0, 20.0)])

    if result.success:
        return tuple(result.x), result.fun
    else:
        return None, float('inf')

def fit_orthorhombic_lattice(peaks_2theta, hkl_list):
    """
    Fit orthorhombic crystal system lattice parameters

    Parameters:
        peaks_2theta: Peak position list (2theta, degrees)
        hkl_list: hkl index list

    Returns:
        (a, b, c): Lattice parameters (Å)
        residual: Fitting residual
    """
    d_values = [two_theta_to_d(tt) for tt in peaks_2theta]

    def objective(params):
        a, b, c = params
        residuals = []
        for i, (h, k, l) in enumerate(hkl_list[:len(d_values)]):
            d_calc = calculate_d_orthorhombic((h, k, l), a, b, c)
            residuals.append((d_values[i] - d_calc)**2)
        return np.sum(residuals)

    # Initial guess
    a_init = d_values[0] * 2.0
    b_init = d_values[0] * 2.0
    c_init = d_values[0] * 2.0

    result = minimize(objective, [a_init, b_init, c_init],
                     bounds=[(1.0, 20.0), (1.0, 20.0), (1.0, 20.0)])

    if result.success:
        return tuple(result.x), result.fun
    else:
        return None, float('inf')

def determine_crystal_system(peaks_2theta, available_peak_count):
    """
    Determine crystal system (extended version)

    Parameters:
        peaks_2theta: Peak position list (2θ, degrees)
        available_peak_count: Number of new peaks available for determination

    Returns:
        best_system: Best matching crystal system name
        lattice_params: Lattice parameters (including volume, angles)
        fit_quality: Fitting residual
        top3_results: Information of top 3 crystal systems with smallest residuals
    """
    results = []

    print(f"\nStarting crystal system determination, available peaks: {len(peaks_2theta)}")

    for system_key, system_info in CRYSTAL_SYSTEMS.items():
        min_peaks = system_info['min_peaks']
        if available_peak_count < min_peaks:
            continue

        hkl_list = system_info['hkl_list']

        try:
            if 'cubic' in system_key:
                params, residual = fit_cubic_lattice(peaks_2theta, hkl_list)
                if params is not None:
                    a = params
                    volume = a ** 3
                    results.append({
                        'system': system_info['name'],
                        'params': {'a': a, 'b': a, 'c': a, 'α': 90, 'β': 90, 'γ': 90, 'volume': volume},
                        'residual': residual
                    })
                    print(f"  {system_info['name']}: a={a:.4f} Å, V={volume:.4f} Å³, residual={residual:.6f}")

            elif system_key == 'hexagonal':
                params, residual = fit_hexagonal_lattice(peaks_2theta, hkl_list)
                if params is not None:
                    a, c = params
                    volume = (3 ** 0.5 / 2) * a**2 * c
                    results.append({
                        'system': system_info['name'],
                        'params': {'a': a, 'b': a, 'c': c, 'α': 90, 'β': 90, 'γ': 120, 'volume': volume, 'c/a': c/a},
                        'residual': residual
                    })
                    print(f"  {system_info['name']}: a={a:.4f} Å, c={c:.4f} Å, c/a={c/a:.4f}, V={volume:.4f} Å³, residual={residual:.6f}")

            elif system_key == 'tetragonal':
                params, residual = fit_tetragonal_lattice(peaks_2theta, hkl_list)
                if params is not None:
                    a, c = params
                    volume = a**2 * c
                    results.append({
                        'system': system_info['name'],
                        'params': {'a': a, 'b': a, 'c': c, 'α': 90, 'β': 90, 'γ': 90, 'volume': volume, 'c/a': c/a},
                        'residual': residual
                    })
                    print(f"  {system_info['name']}: a={a:.4f} Å, c={c:.4f} Å, c/a={c/a:.4f}, V={volume:.4f} Å³, residual={residual:.6f}")

            elif system_key == 'orthorhombic':
                params, residual = fit_orthorhombic_lattice(peaks_2theta, hkl_list)
                if params is not None:
                    a, b, c = params
                    volume = a * b * c
                    results.append({
                        'system': system_info['name'],
                        'params': {'a': a, 'b': b, 'c': c, 'α': 90, 'β': 90, 'γ': 90, 'volume': volume},
                        'residual': residual
                    })
                    print(f"  {system_info['name']}: a={a:.4f} Å, b={b:.4f} Å, c={c:.4f} Å, V={volume:.4f} Å³, residual={residual:.6f}")

        except Exception as e:
            print(f"  {system_info['name']}: Fitting failed ({str(e)})")
            continue

    if not results:
        print("\n>>> No suitable crystal system found")
        return None, None, None, []

    # Sort and select top 3 results with smallest residuals
    results_sorted = sorted(results, key=lambda x: x['residual'])
    top3_results = results_sorted[:3]

    best_result = top3_results[0]
    print(f"\n>>> Best matching crystal system: {best_result['system']}")
    print(f"    Lattice parameters: {best_result['params']}")
    print(f"    Fitting residual: {best_result['residual']:.6f}")

    # Output top 3 results
    if len(top3_results) > 1:
        print("\n>>> Other candidate crystal systems:")
        for i, res in enumerate(top3_results[1:], start=2):
            print(f"  [{i}] {res['system']}: residual={res['residual']:.6f}, params={res['params']}")

    return best_result['system'], best_result['params'], best_result['residual'], top3_results


def analyze_phase_transition(csv_path):
    """
    Main analysis function: Execute complete phase transition analysis workflow

    Parameters:
        csv_path: CSV file path

    Returns:
        analysis_results: Analysis results dictionary
    """
    print("="*70)
    print("Crystal System Determination and Phase Transition Identification Analysis")
    print("="*70)

    # 1. Read data
    print("\n[Step 1] Reading CSV data...")
    pressure_data = read_pressure_peak_data(csv_path)
    print(f"  Total {len(pressure_data)} pressure points read")
    for pressure in sorted(pressure_data.keys()):
        print(f"    {pressure:.2f} GPa: {len(pressure_data[pressure])} peaks")

    # 2. Identify phase transition point
    print("\n[Step 2] Identifying phase transition point...")
    transition_pressure, before_pressures, after_pressures = \
        find_phase_transition_point(pressure_data, PEAK_TOLERANCE_1)

    # 3. Build new peak dataset
    print("\n[Step 3] Building new peak and original peak datasets...")
    if not transition_pressure:
        print("No phase transition detected, analysis terminated.")
        return None

    # Get peaks at transition point and previous point
    idx = sorted(pressure_data.keys()).index(transition_pressure)
    prev_pressure = sorted(pressure_data.keys())[idx - 1]
    peaks_before = pressure_data[prev_pressure]
    peaks_after = pressure_data[transition_pressure]

    # Find new peaks (newly appearing peaks at transition point)
    new_peaks_ref = []
    for peak in peaks_after:
        if all(abs(peak - p) > PEAK_TOLERANCE_1 for p in peaks_before):
            new_peaks_ref.append(peak)

    # Build new peak dataset
    stable_count, new_peak_dataset = collect_tracked_new_peaks(
        pressure_data, transition_pressure, after_pressures, new_peaks_ref, tolerance=PEAK_TOLERANCE_2
    )

    # Build original peak dataset
    original_peak_dataset = build_original_peak_dataset(pressure_data, new_peak_dataset)

    # 4. Determine crystal system for each pressure point before phase transition (using original peaks)
    print("\n[Step 4] Determining crystal system for each pressure point before phase transition...")
    before_analysis = []
    for pressure in before_pressures:
        peak_info = original_peak_dataset.get(pressure, {})
        peaks = peak_info.get('original_peaks', [])
        if peaks:
            system, params, quality, _ = determine_crystal_system(peaks, len(peaks))
            before_analysis.append({
                'pressure': pressure,
                'original_peaks': peaks,
                'crystal_system': system,
                'params': params,
                'fit_quality': quality
            })
            print(f"  Pressure {pressure:.2f} GPa: {len(peaks)} original peaks → Crystal system: {system}")

    # 5. Determine crystal system for each pressure point after phase transition (using new peaks)
    print("\n[Step 5] Determining crystal system for each pressure point after phase transition...")
    after_analysis = []
    for pressure in after_pressures:
        peaks = new_peak_dataset.get(pressure, [])
        if peaks:
            system, params, quality, _ = determine_crystal_system(peaks, len(peaks))
            after_analysis.append({
                'pressure': pressure,
                'new_peaks': peaks,
                'crystal_system': system,
                'params': params,
                'fit_quality': quality
            })
            print(f"  Pressure {pressure:.2f} GPa: {len(peaks)} new peaks → Crystal system: {system}")

    # 6. Output summary
    print("\n" + "="*70)
    print("Analysis Summary")
    print("="*70)

    print(f"\nPhase transition pressure: {transition_pressure:.2f} GPa")

    if before_analysis:
        print("\nBefore phase transition analysis:")
        for result in before_analysis:
            print(f"  Pressure {result['pressure']:.2f} GPa → Crystal system: {result['crystal_system']}, Lattice parameters: {result['params']}")

    if after_analysis:
        print("\nAfter phase transition analysis:")
        for result in after_analysis:
            print(f"  Pressure {result['pressure']:.2f} GPa → Crystal system: {result['crystal_system']}, Lattice parameters: {result['params']}")

    print("\n" + "="*70)

    return {
        'transition_pressure': transition_pressure,
        'before_analysis': before_analysis,
        'after_analysis': after_analysis
    }

# ==================== Command Line Interface ====================

def main():
    """Main function"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python phase_transition_analysis.py <csv_file_path>")
        print("\nExample:")
        print("  python phase_transition_analysis.py data.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        results = analyze_phase_transition(csv_path)

        # Optional: Save results to JSON file
        import json
        output_path = csv_path.replace('.csv', '_phase_analysis.json')

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        results_serializable = convert_to_serializable(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        print(f"\nAnalysis results saved to: {output_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
