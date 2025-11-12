#!/usr/bin/env python3
"""
High-Pressure XRD Phase Transition and EOS Analysis - Enhanced Version
Author: For Felicity's Research
Features: Smart phase detection, lattice parameter calculation, EOS fitting
Enhanced: Separate old/new phase tracking after transition, constrained EOS fitting
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Crystal System Definitions with Extinction Rules
# ============================================================================

@dataclass
class CrystalSystem:
    """Crystal system with proper extinction rules applied"""
    name: str
    min_peaks_required: int
    atoms_per_cell: int
    hkl_list: List[Tuple[int, int, int]]  # Only observable reflections

    def d_spacing(self, hkl: Tuple[int, int, int], a: float, b: float = None,
                  c: float = None) -> float:
        """Calculate d-spacing from hkl and lattice parameters"""
        h, k, l = hkl

        if self.name == 'cubic':
            return a / np.sqrt(h**2 + k**2 + l**2)

        elif self.name == 'hexagonal':
            c = c if c else a * 1.633
            return 1 / np.sqrt(4/3 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)

        return None

    def unit_cell_volume(self, a: float, b: float = None, c: float = None) -> float:
        """Calculate unit cell volume"""
        if self.name == 'cubic':
            return a**3
        elif self.name == 'hexagonal':
            c = c if c else a * 1.633
            return np.sqrt(3) / 2 * a**2 * c
        return None


# Define crystal systems with correct extinction rules
CRYSTAL_SYSTEMS = {
    'fcc': CrystalSystem(
        name='cubic',
        min_peaks_required=3,
        atoms_per_cell=4,
        # FCC: h,k,l all odd or all even
        hkl_list=[
            (1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2), (4,0,0),
            (3,3,1), (4,2,0), (4,2,2), (5,1,1), (3,3,3), (4,4,0),
            (5,3,1), (6,0,0), (4,4,2)
        ]
    ),
    'bcc': CrystalSystem(
        name='cubic',
        min_peaks_required=3,
        atoms_per_cell=2,
        # BCC: h+k+l = even
        hkl_list=[
            (1,1,0), (2,0,0), (2,1,1), (2,2,0), (3,1,0), (2,2,2),
            (3,2,1), (4,0,0), (3,3,0), (4,1,1), (3,3,2), (4,2,0),
            (4,2,2), (5,1,0), (3,3,3)
        ]
    ),
    'hcp': CrystalSystem(
        name='hexagonal',
        min_peaks_required=3,
        atoms_per_cell=2,
        # HCP: l=2n when h-k=3n; no (0,0,l) with l=odd
        hkl_list=[
            (1,0,0), (0,0,2), (1,0,1), (1,0,2), (1,1,0), (1,0,3),
            (2,0,0), (1,1,2), (2,0,1), (0,0,4), (2,0,2), (1,0,4),
            (2,0,3), (2,1,0), (1,1,3), (2,1,1), (2,0,4), (3,0,0)
        ]
    ),
}

# ============================================================================
# EOS Functions
# ============================================================================

def birch_murnaghan_3rd(V, E0, V0, B0, Bp):
    """3rd order Birch-Murnaghan EOS"""
    eta = (V0 / V)**(1.0/3.0)
    E = E0 + 9.0*B0*V0/16.0 * ((eta**2 - 1.0)**2 *
                                (6.0 + Bp*(eta**2 - 1.0) - 4.0*eta**2))
    return E

def birch_murnaghan_2nd(V, E0, V0, B0):
    """2nd order Birch-Murnaghan EOS (Bp=4 fixed)"""
    eta = (V0 / V)**(1.0/3.0)
    E = E0 + 9.0*B0*V0/16.0 * ((eta**2 - 1.0)**2 * (6.0 - 4.0*eta**2))
    return E

def pressure_bm3(V, V0, B0, Bp):
    """Calculate pressure from BM3"""
    eta = (V0 / V)**(1.0/3.0)
    P = 3.0*B0/2.0 * (eta**7 - eta**5) * (1.0 + 3.0/4.0*(Bp - 4.0)*(eta**2 - 1.0))
    return P

# ============================================================================
# Phase Identifier with Smart Crystal System Recognition
# ============================================================================

class PhaseIdentifier:
    """Phase identification with intelligent crystal system validation"""

    def __init__(self, wavelength: float = 0.4133):
        self.wavelength = wavelength
        self.crystal_systems = CRYSTAL_SYSTEMS

    def two_theta_to_d(self, two_theta: float) -> float:
        """Convert 2theta (degrees) to d-spacing (Angstroms)"""
        theta = np.radians(two_theta / 2.0)
        return self.wavelength / (2.0 * np.sin(theta))

    def d_to_two_theta(self, d: float) -> float:
        """Convert d-spacing (Angstroms) to 2theta (degrees)"""
        return 2.0 * np.degrees(np.arcsin(self.wavelength / (2.0 * d)))

    def find_phase_transition(self, pressure_groups: List[pd.DataFrame],
                            tolerance: float = 0.15) -> Tuple[Optional[int], List[float]]:
        """
        Find phase transition point by detecting new peaks
        Returns: (transition_index, list of new phase peak positions at transition)
        """
        print(f"\n{'='*80}")
        print("STEP 1: Phase Transition Detection")
        print(f"{'='*80}")

        for i in range(1, len(pressure_groups)):
            current_peaks = pressure_groups[i]['Center'].values
            previous_peaks = pressure_groups[i-1]['Center'].values

            new_peaks = []
            for peak in current_peaks:
                if all(abs(peak - prev) > tolerance for prev in previous_peaks):
                    new_peaks.append(peak)

            if len(new_peaks) > 0:
                pressure = pressure_groups[i]['File'].iloc[0]
                prev_pressure = pressure_groups[i-1]['File'].iloc[0]
                print(f"\n✓ Phase transition found!")
                print(f"  {prev_pressure:.2f} GPa → {pressure:.2f} GPa")
                print(f"  {len(new_peaks)} new peaks appeared at: {new_peaks}")
                return i, new_peaks

        print("\n✗ No phase transition detected")
        return None, []

    def separate_old_new_peaks(self, all_peaks: np.ndarray, new_phase_peaks: List[float],
                               tolerance: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate peaks into old phase (residual) and new phase

        Parameters:
        -----------
        all_peaks : array of all peak positions at current pressure
        new_phase_peaks : list of peaks identified as new phase at transition
        tolerance : matching tolerance in degrees 2theta

        Returns:
        --------
        old_phase_peaks, new_phase_peaks (as arrays)
        """
        old_peaks = []
        new_peaks = []

        for peak in all_peaks:
            # Check if this peak matches any of the new phase peaks
            is_new_phase = any(abs(peak - new_peak) < tolerance for new_peak in new_phase_peaks)

            if is_new_phase:
                new_peaks.append(peak)
            else:
                old_peaks.append(peak)

        return np.array(old_peaks), np.array(new_peaks)

    def calculate_cubic_lattice(self, d_spacings: np.ndarray,
                               hkl_list: List[Tuple]) -> Dict:
        """Calculate cubic lattice parameter from d-spacings"""
        a_values = []
        for d, hkl in zip(d_spacings, hkl_list):
            h, k, l = hkl
            a = d * np.sqrt(h**2 + k**2 + l**2)
            a_values.append(a)

        return {
            'a': np.mean(a_values),
            'a_std': np.std(a_values)
        }

    def calculate_hexagonal_lattice(self, d_spacings: np.ndarray,
                                   hkl_list: List[Tuple]) -> Dict:
        """Calculate hexagonal lattice parameters"""
        def fit_func(params):
            a, c = params
            error = 0
            for d, hkl in zip(d_spacings, hkl_list):
                h, k, l = hkl
                d_calc = 1.0 / np.sqrt(4.0/3.0 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)
                error += (d - d_calc)**2
            return error

        # Initial guess
        a_init = d_spacings[0] * 2.0
        c_init = a_init * 1.633

        result = minimize(fit_func, [a_init, c_init], method='Nelder-Mead')
        if result.success:
            a, c = result.x
            return {'a': a, 'c': c, 'c/a': c/a}
        return None

    def validate_crystal_match(self, system_name: str, params: Dict,
                              rmse: float, n_peaks: int) -> Tuple[float, str]:
        """
        Smart validation combining multiple criteria
        Not just c/a ratio - consider peak count, RMSE, and structural logic
        """
        confidence = 0.0
        reasons = []

        # Base confidence from RMSE
        rmse_score = 1.0 / (1.0 + rmse * 100)

        if system_name == 'fcc':
            # FCC should have good peak count and consistent lattice
            if 'a_std' in params:
                consistency = 1.0 - params['a_std'] / params['a']
                confidence = rmse_score * consistency

                if n_peaks >= 4:
                    confidence *= 1.1  # Bonus for sufficient peaks

                reasons.append(f"Lattice consistency: {consistency:.3f}")
                reasons.append(f"RMSE: {rmse:.4f}")

                # FCC is common at low pressure - give it priority
                if n_peaks >= 4 and rmse < 0.01:
                    confidence = min(0.95, confidence * 1.2)
                    reasons.append("Strong FCC indicators")

        elif system_name == 'bcc':
            # BCC less common than FCC, needs stronger evidence
            if 'a_std' in params:
                consistency = 1.0 - params['a_std'] / params['a']
                confidence = rmse_score * consistency * 0.85  # Slight penalty

                reasons.append(f"Lattice consistency: {consistency:.3f}")
                reasons.append(f"RMSE: {rmse:.4f}")

        elif system_name == 'hcp':
            # HCP needs c/a check BUT not absolute
            if 'c/a' in params:
                ca_ratio = params['c/a']
                ca_dev = abs(ca_ratio - 1.633)

                # c/a criterion important but not sole determinant
                if ca_dev < 0.1:
                    ca_score = 0.9
                elif ca_dev < 0.2:
                    ca_score = 0.7
                else:
                    ca_score = 0.4

                # Combine with RMSE and peak count
                confidence = rmse_score * ca_score

                if n_peaks >= 4:
                    confidence *= 1.1

                reasons.append(f"c/a = {ca_ratio:.3f} (ideal: 1.633)")
                reasons.append(f"c/a deviation: {ca_dev:.3f}")
                reasons.append(f"RMSE: {rmse:.4f}")

                # Strong HCP only if multiple criteria met
                if ca_dev < 0.1 and rmse < 0.01 and n_peaks >= 4:
                    confidence = min(0.95, confidence * 1.3)
                    reasons.append("Strong HCP indicators")

        reason_str = "; ".join(reasons)
        return min(confidence, 1.0), reason_str

    def match_crystal_system(self, peak_positions: np.ndarray) -> List[Dict]:
        """Match peaks to crystal systems"""
        matches = []
        n_peaks = len(peak_positions)

        if n_peaks == 0:
            return matches

        d_spacings = np.array([self.two_theta_to_d(tt) for tt in peak_positions])

        for sys_name, crystal_sys in self.crystal_systems.items():
            if n_peaks < crystal_sys.min_peaks_required:
                continue

            # Use first n peaks for matching
            n_use = min(n_peaks, len(crystal_sys.hkl_list))
            hkl_indices = crystal_sys.hkl_list[:n_use]
            d_use = d_spacings[:n_use]

            # Calculate lattice parameters
            if crystal_sys.name == 'cubic':
                params = self.calculate_cubic_lattice(d_use, hkl_indices)
            elif crystal_sys.name == 'hexagonal':
                params = self.calculate_hexagonal_lattice(d_use, hkl_indices)
            else:
                continue

            if params is None:
                continue

            # Calculate predicted d-spacings
            d_pred = []
            for hkl in hkl_indices:
                if crystal_sys.name == 'cubic':
                    d = crystal_sys.d_spacing(hkl, params['a'])
                elif crystal_sys.name == 'hexagonal':
                    d = crystal_sys.d_spacing(hkl, params['a'], c=params['c'])
                else:
                    d = None
                if d:
                    d_pred.append(d)

            if len(d_pred) == len(d_use):
                rmse = np.sqrt(np.mean((np.array(d_pred) - d_use)**2))

                # Smart validation
                confidence, reason = self.validate_crystal_match(
                    sys_name, params, rmse, n_peaks
                )

                # Calculate volume per atom
                V_cell = crystal_sys.unit_cell_volume(
                    params['a'],
                    c=params.get('c')
                )
                V_atom = V_cell / crystal_sys.atoms_per_cell if V_cell else None

                matches.append({
                    'system': sys_name,
                    'confidence': confidence,
                    'lattice_params': params,
                    'hkl_assignment': hkl_indices,
                    'rmse': rmse,
                    'validation': reason,
                    'V_cell': V_cell,
                    'V_atom': V_atom,
                    'atoms_per_cell': crystal_sys.atoms_per_cell
                })

        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches

    def analyze_all_pressures(self, pressure_groups: List[pd.DataFrame],
                             transition_idx: Optional[int] = None,
                             new_phase_reference_peaks: List[float] = None,
                             tolerance: float = 0.15) -> List[Dict]:
        """
        Analyze crystal structure at all pressure points
        After phase transition, separate old and new phase peaks

        Parameters:
        -----------
        pressure_groups : list of dataframes for each pressure
        transition_idx : index where phase transition occurs
        new_phase_reference_peaks : peak positions of new phase at transition point
        tolerance : tolerance for peak matching
        """
        results = []

        for idx, group in enumerate(pressure_groups):
            pressure = group['File'].iloc[0]
            peaks = group['Center'].values

            result = {
                'pressure': pressure,
                'n_peaks': len(peaks),
                'peak_positions': peaks,
            }

            # Before transition or no transition: analyze all peaks together
            if transition_idx is None or idx < transition_idx:
                matches = self.match_crystal_system(peaks)
                result['crystal_matches'] = matches

                if matches:
                    best = matches[0]
                    result['best_system'] = best['system']
                    result['confidence'] = best['confidence']
                    result['lattice_params'] = best['lattice_params']
                    result['V_atom'] = best['V_atom']

            # After transition: separate old and new phase peaks
            else:
                old_peaks, new_peaks = self.separate_old_new_peaks(
                    peaks, new_phase_reference_peaks, tolerance
                )

                result['old_phase_peaks'] = old_peaks
                result['new_phase_peaks'] = new_peaks
                result['n_old_peaks'] = len(old_peaks)
                result['n_new_peaks'] = len(new_peaks)

                # Analyze old phase (residual)
                if len(old_peaks) >= 3:
                    old_matches = self.match_crystal_system(old_peaks)
                    if old_matches:
                        best_old = old_matches[0]
                        result['old_phase'] = {
                            'system': best_old['system'],
                            'confidence': best_old['confidence'],
                            'lattice_params': best_old['lattice_params'],
                            'V_atom': best_old['V_atom'],
                            'matches': old_matches
                        }

                # Analyze new phase
                if len(new_peaks) >= 3:
                    new_matches = self.match_crystal_system(new_peaks)
                    if new_matches:
                        best_new = new_matches[0]
                        result['new_phase'] = {
                            'system': best_new['system'],
                            'confidence': best_new['confidence'],
                            'lattice_params': best_new['lattice_params'],
                            'V_atom': best_new['V_atom'],
                            'matches': new_matches
                        }

                # For backward compatibility, set best_system to new phase if available
                if 'new_phase' in result:
                    result['best_system'] = result['new_phase']['system']
                    result['confidence'] = result['new_phase']['confidence']
                    result['lattice_params'] = result['new_phase']['lattice_params']
                    result['V_atom'] = result['new_phase']['V_atom']

            results.append(result)

        return results

# ============================================================================
# EOS Fitter with Physically Reasonable Constraints
# ============================================================================

class EOSFitter:
    """
    Fit equation of state to P-V data with physically reasonable constraints

    Typical ranges from literature:
    - B0 (bulk modulus): 50-400 GPa for most materials
    - B' (pressure derivative): 3-7 for most materials
    """

    def __init__(self, B0_range: Tuple[float, float] = (30, 500),
                 Bp_range: Tuple[float, float] = (2.5, 8.0)):
        """
        Initialize with reasonable parameter ranges

        Parameters:
        -----------
        B0_range : (min, max) for bulk modulus in GPa
        Bp_range : (min, max) for B' (dimensionless)
        """
        self.B0_range = B0_range
        self.Bp_range = Bp_range
        self.results_bm2 = None
        self.results_bm3 = None

    def fit_bm2(self, pressures: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Fit BM2 to P-V data with constraints on B0
        """
        # Convert to energy-volume for fitting (approximate)
        V_sorted = np.sort(volumes)[::-1]
        P_sorted = pressures[np.argsort(volumes)[::-1]]

        # Numerical integration: E = -integral(P dV)
        energies = np.zeros_like(V_sorted)
        for i in range(1, len(V_sorted)):
            dV = V_sorted[i-1] - V_sorted[i]
            energies[i] = energies[i-1] - 0.5 * (P_sorted[i-1] + P_sorted[i]) * dV / 160.21766208

        V0_guess = V_sorted[np.argmin(energies)]
        E0_guess = np.min(energies)
        B0_guess = 100.0  # in eV/Ang^3

        # Set bounds: B0 in eV/Ang^3 (convert from GPa)
        B0_min_eV = self.B0_range[0] / 160.21766208
        B0_max_eV = self.B0_range[1] / 160.21766208

        bounds = (
            [E0_guess - 1.0, V_sorted.min() * 0.8, B0_min_eV],
            [E0_guess + 1.0, V_sorted.max() * 1.2, B0_max_eV]
        )

        try:
            popt, pcov = curve_fit(
                birch_murnaghan_2nd,
                V_sorted, energies,
                p0=[E0_guess, V0_guess, B0_guess],
                bounds=bounds,
                maxfev=10000
            )

            E0, V0, B0 = popt
            perr = np.sqrt(np.diag(pcov))

            residuals = energies - birch_murnaghan_2nd(V_sorted, *popt)
            r_squared = 1 - np.sum(residuals**2) / np.sum((energies - np.mean(energies))**2)

            self.results_bm2 = {
                'E0': E0,
                'V0': V0,
                'B0': B0,
                'B0_GPa': B0 * 160.21766208,
                'Bp': 4.0,
                'errors': {'E0': perr[0], 'V0': perr[1], 'B0': perr[2]},
                'r_squared': r_squared
            }
            return self.results_bm2
        except Exception as e:
            print(f"  Warning: BM2 fitting failed: {e}")
            return None

    def fit_bm3(self, pressures: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Fit BM3 to P-V data with constraints on B0 and B'
        """
        V_sorted = np.sort(volumes)[::-1]
        P_sorted = pressures[np.argsort(volumes)[::-1]]

        # Numerical integration
        energies = np.zeros_like(V_sorted)
        for i in range(1, len(V_sorted)):
            dV = V_sorted[i-1] - V_sorted[i]
            energies[i] = energies[i-1] - 0.5 * (P_sorted[i-1] + P_sorted[i]) * dV / 160.21766208

        V0_guess = V_sorted[np.argmin(energies)]
        E0_guess = np.min(energies)
        B0_guess = 100.0
        Bp_guess = 4.0

        # Set bounds with physical constraints
        B0_min_eV = self.B0_range[0] / 160.21766208
        B0_max_eV = self.B0_range[1] / 160.21766208

        bounds = (
            [E0_guess - 1.0, V_sorted.min() * 0.8, B0_min_eV, self.Bp_range[0]],
            [E0_guess + 1.0, V_sorted.max() * 1.2, B0_max_eV, self.Bp_range[1]]
        )

        try:
            popt, pcov = curve_fit(
                birch_murnaghan_3rd,
                V_sorted, energies,
                p0=[E0_guess, V0_guess, B0_guess, Bp_guess],
                bounds=bounds,
                maxfev=10000
            )

            E0, V0, B0, Bp = popt
            perr = np.sqrt(np.diag(pcov))

            residuals = energies - birch_murnaghan_3rd(V_sorted, *popt)
            r_squared = 1 - np.sum(residuals**2) / np.sum((energies - np.mean(energies))**2)

            self.results_bm3 = {
                'E0': E0,
                'V0': V0,
                'B0': B0,
                'B0_GPa': B0 * 160.21766208,
                'Bp': Bp,
                'errors': {'E0': perr[0], 'V0': perr[1], 'B0': perr[2], 'Bp': perr[3]},
                'r_squared': r_squared
            }
            return self.results_bm3
        except Exception as e:
            print(f"  Warning: BM3 fitting failed: {e}")
            return None

    def plot_eos(self, pressures: np.ndarray, volumes: np.ndarray,
                output_file: str, phase_name: str = 'Phase'):
        """Plot P-V curve with BM fits"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # P-V plot
        ax1.scatter(volumes, pressures, s=100, c='navy', marker='o',
                   edgecolors='black', linewidths=1.5, label='Data', zorder=3)

        if self.results_bm3:
            V_fine = np.linspace(volumes.min()*0.95, volumes.max()*1.05, 200)
            P_fine = pressure_bm3(V_fine, self.results_bm3['V0'],
                                 self.results_bm3['B0'], self.results_bm3['Bp']) * 160.21766208
            ax1.plot(V_fine, P_fine, 'r-', linewidth=2.5, label='BM3', alpha=0.7)

        ax1.set_xlabel('Volume per atom (Ų)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Pressure (GPa)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'P-V: {phase_name}', fontsize=14, fontweight='bold')

        # Parameters table
        ax2.axis('off')
        text_content = f"EOS Fitting Results\n{'='*40}\n\n"
        text_content += f"Phase: {phase_name}\n\n"

        if self.results_bm2:
            text_content += "BM2 (2nd Order):\n"
            text_content += f"  V₀ = {self.results_bm2['V0']:.4f} ± {self.results_bm2['errors']['V0']:.4f} Ų\n"
            text_content += f"  B₀ = {self.results_bm2['B0_GPa']:.1f} ± {self.results_bm2['errors']['B0']*160.22:.1f} GPa\n"
            text_content += f"  B' = 4.0 (fixed)\n"
            text_content += f"  R² = {self.results_bm2['r_squared']:.4f}\n\n"

        if self.results_bm3:
            text_content += "BM3 (3rd Order):\n"
            text_content += f"  V₀ = {self.results_bm3['V0']:.4f} ± {self.results_bm3['errors']['V0']:.4f} Ų\n"
            text_content += f"  B₀ = {self.results_bm3['B0_GPa']:.1f} ± {self.results_bm3['errors']['B0']*160.22:.1f} GPa\n"
            text_content += f"  B' = {self.results_bm3['Bp']:.2f} ± {self.results_bm3['errors']['Bp']:.2f}\n"
            text_content += f"  R² = {self.results_bm3['r_squared']:.4f}\n"

        ax2.text(0.1, 0.9, text_content, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Plot saved: {output_file}")
        plt.close()

# ============================================================================
# Enhanced Report Writer
# ============================================================================

def write_enhanced_report(all_results: List[Dict], transition_idx: Optional[int],
                         eos_old_all: Dict, eos_new: Dict, output_file: str):
    """
    Write enhanced analysis report with separate old/new phase tracking

    Parameters:
    -----------
    all_results : analysis results at each pressure
    transition_idx : index of phase transition
    eos_old_all : EOS fit for old phase using ALL pressure points
    eos_new : EOS fit for new phase (only post-transition)
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("XRD PHASE TRANSITION AND EOS ANALYSIS - ENHANCED\n")
        f.write("="*80 + "\n\n")

        # Phase transition info
        if transition_idx:
            f.write(f"Phase Transition Detected at: {all_results[transition_idx]['pressure']:.2f} GPa\n\n")

        # Results for each pressure
        f.write("CRYSTAL STRUCTURE AT EACH PRESSURE\n")
        f.write("-"*80 + "\n\n")

        for i, result in enumerate(all_results):
            f.write(f"Pressure: {result['pressure']:.2f} GPa\n")
            f.write(f"  Total Peaks: {result['n_peaks']}\n")

            # Before transition or no transition
            if transition_idx is None or i < transition_idx:
                if 'best_system' in result:
                    f.write(f"  Crystal System: {result['best_system'].upper()}\n")
                    f.write(f"  Confidence: {result['confidence']:.3f}\n")

                    params = result['lattice_params']
                    if 'a' in params:
                        f.write(f"  a = {params['a']:.4f} Å")
                        if 'a_std' in params:
                            f.write(f" (±{params['a_std']:.4f})")
                        f.write("\n")

                    if 'c' in params:
                        f.write(f"  c = {params['c']:.4f} Å\n")
                        f.write(f"  c/a = {params['c/a']:.4f}\n")

                    if result['V_atom']:
                        f.write(f"  V/atom = {result['V_atom']:.4f} Ų\n")

            # After transition - show both phases
            else:
                f.write(f"  Old Phase Peaks: {result.get('n_old_peaks', 0)}\n")
                f.write(f"  New Phase Peaks: {result.get('n_new_peaks', 0)}\n")

                # Old phase (residual)
                if 'old_phase' in result:
                    f.write(f"\n  OLD PHASE (Residual):\n")
                    old = result['old_phase']
                    f.write(f"    System: {old['system'].upper()}\n")
                    f.write(f"    Confidence: {old['confidence']:.3f}\n")

                    params = old['lattice_params']
                    if 'a' in params:
                        f.write(f"    a = {params['a']:.4f} Å")
                        if 'a_std' in params:
                            f.write(f" (±{params['a_std']:.4f})")
                        f.write("\n")

                    if 'c' in params:
                        f.write(f"    c = {params['c']:.4f} Å\n")
                        f.write(f"    c/a = {params['c/a']:.4f}\n")

                    if old['V_atom']:
                        f.write(f"    V/atom = {old['V_atom']:.4f} Ų\n")

                # New phase
                if 'new_phase' in result:
                    f.write(f"\n  NEW PHASE:\n")
                    new = result['new_phase']
                    f.write(f"    System: {new['system'].upper()}\n")
                    f.write(f"    Confidence: {new['confidence']:.3f}\n")

                    params = new['lattice_params']
                    if 'a' in params:
                        f.write(f"    a = {params['a']:.4f} Å")
                        if 'a_std' in params:
                            f.write(f" (±{params['a_std']:.4f})")
                        f.write("\n")

                    if 'c' in params:
                        f.write(f"    c = {params['c']:.4f} Å\n")
                        f.write(f"    c/a = {params['c/a']:.4f}\n")

                    if new['V_atom']:
                        f.write(f"    V/atom = {new['V_atom']:.4f} Ų\n")

            f.write("\n")

        # EOS results
        if eos_old_all or eos_new:
            f.write("="*80 + "\n")
            f.write("EQUATION OF STATE RESULTS\n")
            f.write("="*80 + "\n\n")

        if eos_old_all:
            f.write("OLD PHASE (All Pressure Points):\n")
            f.write("-"*80 + "\n")

            if 'bm2' in eos_old_all and eos_old_all['bm2']:
                bm2 = eos_old_all['bm2']
                f.write(f"\nBM2:\n")
                f.write(f"  V₀ = {bm2['V0']:.4f} ± {bm2['errors']['V0']:.4f} Ų\n")
                f.write(f"  B₀ = {bm2['B0_GPa']:.1f} ± {bm2['errors']['B0']*160.22:.1f} GPa\n")
                f.write(f"  R² = {bm2['r_squared']:.4f}\n")

            if 'bm3' in eos_old_all and eos_old_all['bm3']:
                bm3 = eos_old_all['bm3']
                f.write(f"\nBM3:\n")
                f.write(f"  V₀ = {bm3['V0']:.4f} ± {bm3['errors']['V0']:.4f} Ų\n")
                f.write(f"  B₀ = {bm3['B0_GPa']:.1f} ± {bm3['errors']['B0']*160.22:.1f} GPa\n")
                f.write(f"  B' = {bm3['Bp']:.2f} ± {bm3['errors']['Bp']:.2f}\n")
                f.write(f"  R² = {bm3['r_squared']:.4f}\n")
            f.write("\n")

        if eos_new:
            f.write("NEW PHASE (Post-Transition Only):\n")
            f.write("-"*80 + "\n")

            if 'bm2' in eos_new and eos_new['bm2']:
                bm2 = eos_new['bm2']
                f.write(f"\nBM2:\n")
                f.write(f"  V₀ = {bm2['V0']:.4f} ± {bm2['errors']['V0']:.4f} Ų\n")
                f.write(f"  B₀ = {bm2['B0_GPa']:.1f} ± {bm2['errors']['B0']*160.22:.1f} GPa\n"
                f.write(f"  R² = {bm2['r_squared']:.4f}\n")

            if 'bm3' in eos_new and eos_new['bm3']:
                bm3 = eos_new['bm3']
                f.write(f"\nBM3:\n")
                f.write(f"  V₀ = {bm3['V0']:.4f} ± {bm3['errors']['V0']:.4f} Ų\n")
                f.write(f"  B₀ = {bm3['B0_GPa']:.1f} ± {bm3['errors']['B0']*160.22:.1f} GPa\n")
                f.write(f"  B' = {bm3['Bp']:.2f} ± {bm3['errors']['Bp']:.2f}\n")
                f.write(f"  R² = {bm3['r_squared']:.4f}\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

# ============================================================================
# Main Analysis Function - Enhanced
# ============================================================================

def run_analysis(csv_file: str, wavelength: float = 0.4133,
                tolerance: float = 0.3, output_prefix: str = 'analysis',
                B0_range: Tuple[float, float] = (30, 500),
                Bp_range: Tuple[float, float] = (2.5, 8.0)):
    """
    Main analysis function with enhanced phase separation

    Parameters:
    -----------
    csv_file : str
        Path to CSV with peak data
    wavelength : float
        X-ray wavelength in Angstroms
    tolerance : float
        Tolerance for phase transition detection (degrees 2theta)
    output_prefix : str
        Prefix for output files
    B0_range : tuple
        (min, max) for bulk modulus in GPa
    Bp_range : tuple
        (min, max) for B' (pressure derivative)
    """
    print("="*80)
    print("XRD PHASE TRANSITION AND EOS ANALYSIS - ENHANCED")
    print("="*80)
    print(f"\nWavelength: {wavelength} Å")
    print(f"Tolerance: {tolerance}°")
    print(f"B₀ constraints: {B0_range[0]}-{B0_range[1]} GPa")
    print(f"B' constraints: {Bp_range[0]}-{Bp_range[1]}\n")

    # Read data
    df = pd.read_csv(csv_file)

    # Group by pressure
    pressure_groups = []
    current_group = []

    for idx, row in df.iterrows():
        if pd.isna(row['Center']):
            if current_group:
                pressure_groups.append(pd.DataFrame(current_group))
                current_group = []
        else:
            current_group.append(row)

    if current_group:
        pressure_groups.append(pd.DataFrame(current_group))

    pressure_groups.sort(key=lambda g: g['File'].iloc[0])

    print(f"Total pressure points: {len(pressure_groups)}")
    print(f"Pressure range: {pressure_groups[0]['File'].iloc[0]:.2f} - {pressure_groups[-1]['File'].iloc[0]:.2f} GPa\n")

    # Initialize identifier
    identifier = PhaseIdentifier(wavelength=wavelength)

    # Find phase transition
    transition_idx, new_phase_peaks = identifier.find_phase_transition(pressure_groups, tolerance)

    # Analyze all pressures with phase separation
    print(f"\n{'='*80}")
    print("STEP 2: Crystal Structure Identification with Phase Separation")
    print(f"{'='*80}\n")

    all_results = identifier.analyze_all_pressures(
        pressure_groups,
        transition_idx,
        new_phase_peaks,
        tolerance
    )

    # Print results
    for result in all_results:
        print(f"P = {result['pressure']:.2f} GPa: ", end='')

        if transition_idx is None or result['pressure'] < all_results[transition_idx]['pressure']:
            if 'best_system' in result:
                print(f"{result['best_system'].upper()} (conf={result['confidence']:.3f})")
                if result['V_atom']:
                    print(f"  V/atom = {result['V_atom']:.4f} Ų")
            else:
                print("No match")
        else:
            print("")
            if 'old_phase' in result:
                old = result['old_phase']
                print(f"  Old phase: {old['system'].upper()} (conf={old['confidence']:.3f}), V={old['V_atom']:.4f} Ų")
            if 'new_phase' in result:
                new = result['new_phase']
                print(f"  New phase: {new['system'].upper()} (conf={new['confidence']:.3f}), V={new['V_atom']:.4f} Ų")

    # EOS fitting with constraints
    print(f"\n{'='*80}")
    print("STEP 3: EOS Fitting with Physical Constraints")
    print(f"{'='*80}\n")

    eos_results_old_all = {}
    eos_results_new = {}

    if transition_idx:
        # OLD PHASE - Use ALL pressure points (before and after transition)
        print("Collecting old phase data from ALL pressure points...")
        old_phase_data = []

        # Before transition
        for r in all_results[:transition_idx]:
            if 'V_atom' in r and r['V_atom']:
                old_phase_data.append({
                    'pressure': r['pressure'],
                    'V_atom': r['V_atom']
                })

        # After transition (residual old phase)
        for r in all_results[transition_idx:]:
            if 'old_phase' in r and r['old_phase']['V_atom']:
                old_phase_data.append({
                    'pressure': r['pressure'],
                    'V_atom': r['old_phase']['V_atom']
                })

        if len(old_phase_data) >= 3:
            print(f"Fitting old phase EOS with {len(old_phase_data)} data points...")
            pressures_old = np.array([d['pressure'] for d in old_phase_data])
            volumes_old = np.array([d['V_atom'] for d in old_phase_data])

            fitter_old = EOSFitter(B0_range=B0_range, Bp_range=Bp_range)
            eos_results_old_all['bm2'] = fitter_old.fit_bm2(pressures_old, volumes_old)
            eos_results_old_all['bm3'] = fitter_old.fit_bm3(pressures_old, volumes_old)

            fitter_old.plot_eos(pressures_old, volumes_old,
                               output_file=f'{output_prefix}_eos_old_all.png',
                               phase_name='Old Phase (All Data)')

        # NEW PHASE - Only post-transition data
        print("\nCollecting new phase data from post-transition points...")
        new_phase_data = []

        for r in all_results[transition_idx:]:
            if 'new_phase' in r and r['new_phase']['V_atom']:
                new_phase_data.append({
                    'pressure': r['pressure'],
                    'V_atom': r['new_phase']['V_atom']
                })

        if len(new_phase_data) >= 3:
            print(f"Fitting new phase EOS with {len(new_phase_data)} data points...")
            pressures_new = np.array([d['pressure'] for d in new_phase_data])
            volumes_new = np.array([d['V_atom'] for d in new_phase_data])

            fitter_new = EOSFitter(B0_range=B0_range, Bp_range=Bp_range)
            eos_results_new['bm2'] = fitter_new.fit_bm2(pressures_new, volumes_new)
            eos_results_new['bm3'] = fitter_new.fit_bm3(pressures_new, volumes_new)

            fitter_new.plot_eos(pressures_new, volumes_new,
                               output_file=f'{output_prefix}_eos_new.png',
                               phase_name='New Phase')
    else:
        # No transition - fit all data
        valid_results = [r for r in all_results if 'V_atom' in r and r['V_atom']]
        if len(valid_results) >= 3:
            print("No transition detected. Fitting EOS for all data...")
            pressures_all = np.array([r['pressure'] for r in valid_results])
            volumes_all = np.array([r['V_atom'] for r in valid_results])

            fitter = EOSFitter(B0_range=B0_range, Bp_range=Bp_range)
            eos_results_old_all['bm2'] = fitter.fit_bm2(pressures_all, volumes_all)
            eos_results_old_all['bm3'] = fitter.fit_bm3(pressures_all, volumes_all)

            fitter.plot_eos(pressures_all, volumes_all,
                           output_file=f'{output_prefix}_eos.png',
                           phase_name='Phase')

    # Write enhanced report
    print(f"\n{'='*80}")
    print("Writing Enhanced Report")
    print(f"{'='*80}\n")

    report_file = f"{output_prefix}_report_enhanced.txt"
    write_enhanced_report(all_results, transition_idx,
                         eos_results_old_all, eos_results_new, report_file)

    print(f"✓ Report saved: {report_file}\n")
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return all_results, transition_idx

# ============================================================================
# Main Entry
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CSV_FILE = r'D:\HEPS\ID31\dioptas_data\Al0\fit_output\all_results.csv'
    WAVELENGTH = 0.4133  # Angstroms
    TOLERANCE = 0.3      # degrees 2theta
    OUTPUT_PREFIX = r'D:\HEPS\ID31\dioptas_data\Al0\fit_output\analysis'

    # Physical constraints for EOS fitting (based on literature)
    # For typical metals/materials:
    # - Bulk modulus B0: 30-500 GPa
    # - Pressure derivative B': 2.5-8.0
    B0_RANGE = (30, 500)   # GPa
    BP_RANGE = (2.5, 8.0)  # dimensionless

    run_analysis(
        CSV_FILE,
        WAVELENGTH,
        TOLERANCE,
        OUTPUT_PREFIX,
        B0_RANGE,
        BP_RANGE
    )

    print("\n✓ All done!")
