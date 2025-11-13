# -*- coding: utf-8 -*-
"""
Birch-Murnaghan Equation Fitting Class (Encapsulated Version)
Object-oriented implementation for P-V curve fitting with BM equations
@author: candicewang928@gmail.com
Created: 2025-11-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from typing import Tuple, Dict, Optional, Union


class BMFitter:
    """
    Birch-Murnaghan Equation of State Fitter

    A comprehensive class for fitting pressure-volume data using
    2nd and 3rd order Birch-Murnaghan equations.

    Attributes:
    -----------
    V_data : np.ndarray
        Volume data (Å³/atom)
    P_data : np.ndarray
        Pressure data (GPa)
    phase_name : str
        Name of the phase being fitted
    results_2nd : dict or None
        Fitting results for 2nd order BM equation
    results_3rd : dict or None
        Fitting results for 3rd order BM equation

    Example:
    --------
    >>> fitter = BMFitter(V_data, P_data, phase_name="Original Phase")
    >>> fitter.fit_both_orders()
    >>> fitter.plot_pv_curve(save_path="output.png")
    >>> params = fitter.get_parameters()
    """

    def __init__(self, V_data: np.ndarray, P_data: np.ndarray,
                 phase_name: str = "Unknown Phase"):
        """
        Initialize the BM fitter with volume and pressure data

        Parameters:
        -----------
        V_data : array-like
            Volume data (Å³/atom)
        P_data : array-like
            Pressure data (GPa)
        phase_name : str, optional
            Name of the phase for identification
        """
        self.V_data = np.asarray(V_data)
        self.P_data = np.asarray(P_data)
        self.phase_name = phase_name
        self.results_2nd = None
        self.results_3rd = None

        # Configure matplotlib for proper symbol rendering
        self._configure_matplotlib()

        # Validate input data
        self._validate_data()

    @staticmethod
    def _configure_matplotlib():
        """Configure matplotlib to properly display special characters"""
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.default'] = 'regular'

    def _validate_data(self):
        """Validate input data arrays"""
        if len(self.V_data) != len(self.P_data):
            raise ValueError(f"Volume and pressure data must have the same length. "
                           f"Got V: {len(self.V_data)}, P: {len(self.P_data)}")

        if len(self.V_data) < 3:
            raise ValueError(f"At least 3 data points required for fitting. "
                           f"Got {len(self.V_data)} points")

        if np.any(self.V_data <= 0):
            raise ValueError("Volume data must be positive")

    @staticmethod
    def birch_murnaghan_2nd(V: Union[float, np.ndarray],
                           V0: float, B0: float) -> Union[float, np.ndarray]:
        """
        2nd order Birch-Murnaghan Equation of State

        Parameters:
        -----------
        V : float or array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)

        Returns:
        --------
        P : float or array
            Pressure (GPa)
        """
        eta = (V0 / V) ** (1/3)
        P = 3 * B0 / 2 * (eta**7 - eta**5)
        return P

    @staticmethod
    def birch_murnaghan_3rd(V: Union[float, np.ndarray],
                           V0: float, B0: float,
                           B0_prime: float) -> Union[float, np.ndarray]:
        """
        3rd order Birch-Murnaghan Equation of State

        Parameters:
        -----------
        V : float or array
            Volume (Å³/atom)
        V0 : float
            Zero-pressure volume (Å³/atom)
        B0 : float
            Zero-pressure bulk modulus (GPa)
        B0_prime : float
            First pressure derivative of bulk modulus (dimensionless)

        Returns:
        --------
        P : float or array
            Pressure (GPa)
        """
        eta = (V0 / V) ** (1/3)
        P = 3 * B0 / 2 * (eta**7 - eta**5) * (1 + 0.75 * (B0_prime - 4) * (eta**2 - 1))
        return P

    def fit_2nd_order(self, V0_bounds: Optional[Tuple[float, float]] = None,
                     B0_bounds: Tuple[float, float] = (50, 500),
                     initial_guess: Optional[Tuple[float, float]] = None,
                     verbose: bool = True) -> Dict:
        """
        Fit 2nd order Birch-Murnaghan equation

        Parameters:
        -----------
        V0_bounds : tuple of (min, max), optional
            Bounds for V0 parameter. If None, uses 0.8-1.3 times max volume
        B0_bounds : tuple of (min, max), default (50, 500)
            Bounds for B0 parameter in GPa
        initial_guess : tuple of (V0, B0), optional
            Initial guess for parameters. If None, uses automatic guess
        verbose : bool, default True
            Whether to print fitting results

        Returns:
        --------
        results : dict
            Dictionary containing fitting parameters and statistics
        """
        # Set bounds
        if V0_bounds is None:
            V0_bounds = (np.max(self.V_data) * 0.8, np.max(self.V_data) * 1.3)

        bounds = ([V0_bounds[0], B0_bounds[0]],
                 [V0_bounds[1], B0_bounds[1]])

        # Set initial guess
        if initial_guess is None:
            V0_guess = np.max(self.V_data) * 1.02
            B0_guess = 150
            initial_guess = [V0_guess, B0_guess]

        try:
            popt, pcov = curve_fit(
                self.birch_murnaghan_2nd,
                self.V_data,
                self.P_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )

            V0, B0 = popt
            perr = np.sqrt(np.diag(pcov))

            # Calculate statistics
            P_fit = self.birch_murnaghan_2nd(self.V_data, *popt)
            residuals = self.P_data - P_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.P_data - np.mean(self.P_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals**2))

            self.results_2nd = {
                'V0': V0,
                'V0_err': perr[0],
                'B0': B0,
                'B0_err': perr[1],
                'B0_prime': 4.0,  # Fixed for 2nd order
                'B0_prime_err': 0,
                'R_squared': r_squared,
                'RMSE': rmse,
                'fitted_P': P_fit,
                'residuals': residuals,
                'success': True
            }

            if verbose:
                self._print_results(self.results_2nd, order=2)

            return self.results_2nd

        except Exception as e:
            if verbose:
                print(f"⚠️ {self.phase_name} - 2nd order BM fitting failed: {e}")
            self.results_2nd = {'success': False, 'error': str(e)}
            return self.results_2nd

    def fit_3rd_order(self, V0_bounds: Optional[Tuple[float, float]] = None,
                     B0_bounds: Tuple[float, float] = (50, 500),
                     B0_prime_bounds: Tuple[float, float] = (2.5, 6.5),
                     initial_guess: Optional[Tuple[float, float, float]] = None,
                     verbose: bool = True) -> Dict:
        """
        Fit 3rd order Birch-Murnaghan equation

        Parameters:
        -----------
        V0_bounds : tuple of (min, max), optional
            Bounds for V0 parameter. If None, uses 0.8-1.3 times max volume
        B0_bounds : tuple of (min, max), default (50, 500)
            Bounds for B0 parameter in GPa
        B0_prime_bounds : tuple of (min, max), default (2.5, 6.5)
            Bounds for B0' parameter (dimensionless)
        initial_guess : tuple of (V0, B0, B0'), optional
            Initial guess for parameters. If None, uses automatic guess
        verbose : bool, default True
            Whether to print fitting results

        Returns:
        --------
        results : dict
            Dictionary containing fitting parameters and statistics
        """
        # Set bounds
        if V0_bounds is None:
            V0_bounds = (np.max(self.V_data) * 0.8, np.max(self.V_data) * 1.3)

        bounds = ([V0_bounds[0], B0_bounds[0], B0_prime_bounds[0]],
                 [V0_bounds[1], B0_bounds[1], B0_prime_bounds[1]])

        # Set initial guess
        if initial_guess is None:
            V0_guess = np.max(self.V_data) * 1.02
            B0_guess = 150
            B0_prime_guess = 4.0
            initial_guess = [V0_guess, B0_guess, B0_prime_guess]

        try:
            popt, pcov = curve_fit(
                self.birch_murnaghan_3rd,
                self.V_data,
                self.P_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )

            V0, B0, B0_prime = popt
            perr = np.sqrt(np.diag(pcov))

            # Calculate statistics
            P_fit = self.birch_murnaghan_3rd(self.V_data, *popt)
            residuals = self.P_data - P_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.P_data - np.mean(self.P_data))**2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals**2))

            self.results_3rd = {
                'V0': V0,
                'V0_err': perr[0],
                'B0': B0,
                'B0_err': perr[1],
                'B0_prime': B0_prime,
                'B0_prime_err': perr[2],
                'R_squared': r_squared,
                'RMSE': rmse,
                'fitted_P': P_fit,
                'residuals': residuals,
                'success': True
            }

            if verbose:
                self._print_results(self.results_3rd, order=3)

            return self.results_3rd

        except Exception as e:
            if verbose:
                print(f"⚠️ {self.phase_name} - 3rd order BM fitting failed: {e}")
            self.results_3rd = {'success': False, 'error': str(e)}
            return self.results_3rd

    def fit_both_orders(self, verbose: bool = True) -> Tuple[Dict, Dict]:
        """
        Fit both 2nd and 3rd order BM equations

        Parameters:
        -----------
        verbose : bool, default True
            Whether to print fitting results

        Returns:
        --------
        results_2nd, results_3rd : tuple of dicts
            Fitting results for both orders
        """
        self.fit_2nd_order(verbose=verbose)
        self.fit_3rd_order(verbose=verbose)
        return self.results_2nd, self.results_3rd

    def _print_results(self, results: Dict, order: int):
        """Print fitting results in formatted output"""
        if not results.get('success', False):
            return

        print(f"\n{'='*60}")
        print(f"{self.phase_name} - {order}{'nd' if order == 2 else 'rd'} Order BM Fitting Results:")
        print(f"{'='*60}")
        print(f"V₀ = {results['V0']:.4f} ± {results['V0_err']:.4f} Å³/atom")
        print(f"B₀ = {results['B0']:.2f} ± {results['B0_err']:.2f} GPa")

        if order == 2:
            print(f"B₀' = 4.0 (fixed)")
        else:
            print(f"B₀' = {results['B0_prime']:.3f} ± {results['B0_prime_err']:.3f}")

        print(f"R² = {results['R_squared']:.6f}")
        print(f"RMSE = {results['RMSE']:.4f} GPa")

    def plot_pv_curve(self, save_path: Optional[str] = None,
                     show_2nd: bool = True, show_3rd: bool = True,
                     figsize: Tuple[int, int] = (12, 8),
                     dpi: int = 300) -> plt.Figure:
        """
        Plot P-V curve with fitting results

        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure. If None, figure is not saved
        show_2nd : bool, default True
            Whether to show 2nd order fit
        show_3rd : bool, default True
            Whether to show 3rd order fit
        figsize : tuple, default (12, 8)
            Figure size in inches
        dpi : int, default 300
            Resolution for saved figure

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot experimental data
        ax.scatter(self.V_data, self.P_data, s=80, c='blue', marker='o',
                  label='Experimental Data', alpha=0.7, edgecolors='black', zorder=3)

        # Generate smooth curve for fitting
        V_fit = np.linspace(self.V_data.min()*0.95, self.V_data.max()*1.05, 200)

        # Plot 2nd order fit
        if show_2nd and self.results_2nd and self.results_2nd.get('success', False):
            P_fit = self.birch_murnaghan_2nd(V_fit,
                                            self.results_2nd['V0'],
                                            self.results_2nd['B0'])
            ax.plot(V_fit, P_fit, 'r-', linewidth=2.5,
                   label='2nd Order BM Fit', alpha=0.8, zorder=2)

            # Add text box
            textstr = f"2nd Order:\n"
            textstr += f"$V_0$ = {self.results_2nd['V0']:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {self.results_2nd['B0']:.2f} GPa\n"
            textstr += f"$B_0'$ = 4.0 (fixed)\n"
            textstr += f"$R^2$ = {self.results_2nd['R_squared']:.6f}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 3rd order fit
        if show_3rd and self.results_3rd and self.results_3rd.get('success', False):
            P_fit = self.birch_murnaghan_3rd(V_fit,
                                            self.results_3rd['V0'],
                                            self.results_3rd['B0'],
                                            self.results_3rd['B0_prime'])
            ax.plot(V_fit, P_fit, 'g-', linewidth=2.5,
                   label='3rd Order BM Fit', alpha=0.8, zorder=2)

            # Add text box
            textstr = f"3rd Order:\n"
            textstr += f"$V_0$ = {self.results_3rd['V0']:.4f} Å³/atom\n"
            textstr += f"$B_0$ = {self.results_3rd['B0']:.2f} GPa\n"
            textstr += f"$B_0'$ = {self.results_3rd['B0_prime']:.3f}\n"
            textstr += f"$R^2$ = {self.results_3rd['R_squared']:.6f}"
            ax.text(0.05, 0.50, textstr, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
        ax.set_ylabel('Pressure P (GPa)', fontsize=12)
        ax.set_title(f'{self.phase_name} - Birch-Murnaghan Equation Fitting',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✅ Figure saved to: {save_path}")

        return fig

    def plot_residuals(self, save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (14, 6),
                      dpi: int = 300) -> plt.Figure:
        """
        Plot residuals for both 2nd and 3rd order fits

        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure. If None, figure is not saved
        figsize : tuple, default (14, 6)
            Figure size in inches
        dpi : int, default 300
            Resolution for saved figure

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'{self.phase_name} - Fitting Residuals Analysis',
                    fontsize=14, fontweight='bold')

        # 2nd order residuals
        if self.results_2nd and self.results_2nd.get('success', False):
            ax = axes[0]
            residuals = self.results_2nd['residuals']
            ax.scatter(self.V_data, residuals, s=60, c='blue', marker='o', alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
            ax.set_ylabel('Residuals (GPa)', fontsize=11)
            ax.set_title('2nd Order BM Residuals', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            textstr = f"RMSE = {self.results_2nd['RMSE']:.4f} GPa"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5))

        # 3rd order residuals
        if self.results_3rd and self.results_3rd.get('success', False):
            ax = axes[1]
            residuals = self.results_3rd['residuals']
            ax.scatter(self.V_data, residuals, s=60, c='green', marker='o', alpha=0.7)
            ax.axhline(y=0, color='g', linestyle='--', linewidth=2)
            ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
            ax.set_ylabel('Residuals (GPa)', fontsize=11)
            ax.set_title('3rd Order BM Residuals', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            textstr = f"RMSE = {self.results_3rd['RMSE']:.4f} GPa"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✅ Residuals figure saved to: {save_path}")

        return fig

    def get_parameters(self) -> pd.DataFrame:
        """
        Get fitting parameters as a pandas DataFrame

        Returns:
        --------
        df : pd.DataFrame
            DataFrame containing all fitting parameters
        """
        data = []

        if self.results_2nd and self.results_2nd.get('success', False):
            data.append({
                'Phase': self.phase_name,
                'Order': '2nd',
                'V₀ (Å³/atom)': self.results_2nd['V0'],
                'V₀ Error': self.results_2nd['V0_err'],
                'B₀ (GPa)': self.results_2nd['B0'],
                'B₀ Error': self.results_2nd['B0_err'],
                "B₀'": self.results_2nd['B0_prime'],
                "B₀' Error": self.results_2nd['B0_prime_err'],
                'R²': self.results_2nd['R_squared'],
                'RMSE (GPa)': self.results_2nd['RMSE']
            })

        if self.results_3rd and self.results_3rd.get('success', False):
            data.append({
                'Phase': self.phase_name,
                'Order': '3rd',
                'V₀ (Å³/atom)': self.results_3rd['V0'],
                'V₀ Error': self.results_3rd['V0_err'],
                'B₀ (GPa)': self.results_3rd['B0'],
                'B₀ Error': self.results_3rd['B0_err'],
                "B₀'": self.results_3rd['B0_prime'],
                "B₀' Error": self.results_3rd['B0_prime_err'],
                'R²': self.results_3rd['R_squared'],
                'RMSE (GPa)': self.results_3rd['RMSE']
            })

        return pd.DataFrame(data)

    def save_parameters(self, filepath: str):
        """
        Save fitting parameters to CSV file

        Parameters:
        -----------
        filepath : str
            Path to save the CSV file
        """
        df = self.get_parameters()
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ Parameters saved to: {filepath}")

    def compare_fits(self) -> Optional[str]:
        """
        Compare 2nd and 3rd order fits and recommend which to use

        Returns:
        --------
        recommendation : str or None
            Recommendation on which order to use
        """
        if not (self.results_2nd and self.results_2nd.get('success', False) and
                self.results_3rd and self.results_3rd.get('success', False)):
            print("⚠️ Both fits must be successful to compare")
            return None

        r2_2nd = self.results_2nd['R_squared']
        r2_3rd = self.results_3rd['R_squared']
        rmse_2nd = self.results_2nd['RMSE']
        rmse_3rd = self.results_3rd['RMSE']
        b0_prime = self.results_3rd['B0_prime']
        b0_prime_err = self.results_3rd['B0_prime_err']

        print(f"\n{'='*60}")
        print(f"Comparison of Fitting Results")
        print(f"{'='*60}")
        print(f"2nd Order: R² = {r2_2nd:.6f}, RMSE = {rmse_2nd:.4f} GPa")
        print(f"3rd Order: R² = {r2_3rd:.6f}, RMSE = {rmse_3rd:.4f} GPa")
        print(f"3rd Order B₀' = {b0_prime:.3f} ± {b0_prime_err:.3f}")

        # Recommendation logic
        recommendation = ""

        if len(self.V_data) < 8:
            recommendation = "2nd order (insufficient data points for 3rd order)"
        elif abs(b0_prime - 4.0) < 0.5:
            recommendation = "2nd order (B₀' close to 4, 2nd order sufficient)"
        elif b0_prime_err / b0_prime > 0.2:
            recommendation = "2nd order (large uncertainty in B₀', possible overfitting)"
        elif rmse_3rd < rmse_2nd * 0.8:
            recommendation = "3rd order (significantly better fit)"
        elif r2_3rd > 0.995 and r2_3rd > r2_2nd:
            recommendation = "3rd order (excellent fit with improvement over 2nd)"
        else:
            recommendation = "2nd order (simpler model adequate)"

        print(f"\n💡 Recommendation: Use {recommendation}")
        print(f"{'='*60}\n")

        return recommendation


class BMMultiPhaseFitter:
    """
    Multi-phase Birch-Murnaghan Fitter

    Convenience class for fitting multiple phases simultaneously
    and generating comparison plots
    """

    def __init__(self):
        """Initialize multi-phase fitter"""
        self.fitters = {}

    def add_phase(self, phase_name: str, V_data: np.ndarray,
                 P_data: np.ndarray) -> BMFitter:
        """
        Add a phase to fit

        Parameters:
        -----------
        phase_name : str
            Name of the phase
        V_data : array-like
            Volume data
        P_data : array-like
            Pressure data

        Returns:
        --------
        fitter : BMFitter
            The created BMFitter object
        """
        fitter = BMFitter(V_data, P_data, phase_name)
        self.fitters[phase_name] = fitter
        return fitter

    def fit_all(self, verbose: bool = True):
        """Fit all phases with both orders"""
        for name, fitter in self.fitters.items():
            if verbose:
                print(f"\n{'#'*70}")
                print(f"Fitting {name}")
                print(f"{'#'*70}")
            fitter.fit_both_orders(verbose=verbose)

    def plot_comparison(self, save_path: Optional[str] = None,
                       order: int = 3, figsize: Tuple[int, int] = (14, 8),
                       dpi: int = 300) -> plt.Figure:
        """
        Plot comparison of all phases

        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        order : int, default 3
            Which order to plot (2 or 3)
        figsize : tuple, default (14, 8)
            Figure size in inches
        dpi : int, default 300
            Resolution for saved figure

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'D', 'v', 'p']

        for idx, (name, fitter) in enumerate(self.fitters.items()):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # Plot experimental data
            ax.scatter(fitter.V_data, fitter.P_data, s=80, c=color,
                      marker=marker, label=f'{name} (Data)',
                      alpha=0.7, edgecolors='black', zorder=3)

            # Plot fit
            results = fitter.results_3rd if order == 3 else fitter.results_2nd
            if results and results.get('success', False):
                V_fit = np.linspace(fitter.V_data.min()*0.95,
                                   fitter.V_data.max()*1.05, 200)
                if order == 3:
                    P_fit = BMFitter.birch_murnaghan_3rd(
                        V_fit, results['V0'], results['B0'], results['B0_prime'])
                else:
                    P_fit = BMFitter.birch_murnaghan_2nd(
                        V_fit, results['V0'], results['B0'])

                ax.plot(V_fit, P_fit, '-', color=color, linewidth=2.5,
                       label=f'{name} ({order}{"nd" if order==2 else "rd"} Order Fit)',
                       alpha=0.8, zorder=2)

        ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
        ax.set_ylabel('Pressure P (GPa)', fontsize=12)
        ax.set_title(f'Multi-Phase Comparison - {order}{"nd" if order==2 else "rd"} Order BM Fits',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✅ Comparison figure saved to: {save_path}")

        return fig

    def get_all_parameters(self) -> pd.DataFrame:
        """
        Get parameters for all phases

        Returns:
        --------
        df : pd.DataFrame
            Combined DataFrame with all parameters
        """
        dfs = []
        for fitter in self.fitters.values():
            dfs.append(fitter.get_parameters())

        return pd.concat(dfs, ignore_index=True)

    def save_all_parameters(self, filepath: str):
        """
        Save all parameters to CSV

        Parameters:
        -----------
        filepath : str
            Path to save the CSV file
        """
        df = self.get_all_parameters()
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ All parameters saved to: {filepath}")


# Convenience function for quick fitting
def quick_fit(V_data: np.ndarray, P_data: np.ndarray,
             phase_name: str = "Phase",
             save_dir: Optional[str] = None,
             show_plots: bool = True) -> BMFitter:
    """
    Quick fitting function with automatic plotting and saving

    Parameters:
    -----------
    V_data : array-like
        Volume data (Å³/atom)
    P_data : array-like
        Pressure data (GPa)
    phase_name : str, default "Phase"
        Name of the phase
    save_dir : str, optional
        Directory to save results. If None, results are not saved
    show_plots : bool, default True
        Whether to display plots

    Returns:
    --------
    fitter : BMFitter
        The fitted BMFitter object

    Example:
    --------
    >>> V = np.array([16.5, 16.2, 15.9, 15.6, 15.3])
    >>> P = np.array([0.5, 2.3, 5.1, 8.7, 12.4])
    >>> fitter = quick_fit(V, P, phase_name="Original Phase", save_dir="./output")
    """
    print(f"\n{'='*70}")
    print(f"Quick Fitting: {phase_name}")
    print(f"{'='*70}")

    # Create fitter and fit
    fitter = BMFitter(V_data, P_data, phase_name)
    fitter.fit_both_orders(verbose=True)

    # Compare fits
    fitter.compare_fits()

    # Plot
    if show_plots or save_dir:
        pv_path = os.path.join(save_dir, f"{phase_name}_PV_curve.png") if save_dir else None
        res_path = os.path.join(save_dir, f"{phase_name}_residuals.png") if save_dir else None

        fig1 = fitter.plot_pv_curve(save_path=pv_path)
        fig2 = fitter.plot_residuals(save_path=res_path)

        if show_plots:
            plt.show()

    # Save parameters
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        param_path = os.path.join(save_dir, f"{phase_name}_parameters.csv")
        fitter.save_parameters(param_path)

    return fitter


if __name__ == "__main__":
    # Example usage
    print("BMFitter Class - Example Usage\n")

    # Generate example data
    V_example = np.array([16.5432, 16.2341, 15.9876, 15.6543, 15.3421,
                         15.0567, 14.7892, 14.5234, 14.2876, 14.0543])
    P_example = np.array([0.5, 2.3, 5.1, 8.7, 12.4, 16.8, 21.5, 26.9, 32.7, 39.2])

    # Method 1: Using quick_fit function
    print("="*70)
    print("Method 1: Quick Fit")
    print("="*70)
    fitter = quick_fit(V_example, P_example, phase_name="Example Phase",
                      save_dir="./BM_output_class", show_plots=False)

    print("\n" + "="*70)
    print("Method 2: Manual Usage")
    print("="*70)

    # Method 2: Manual usage with more control
    fitter2 = BMFitter(V_example, P_example, phase_name="Manual Fit Example")

    # Fit with custom bounds
    fitter2.fit_2nd_order(B0_bounds=(100, 300), verbose=True)
    fitter2.fit_3rd_order(B0_bounds=(100, 300), B0_prime_bounds=(3.0, 5.0), verbose=True)

    # Get parameters
    df = fitter2.get_parameters()
    print("\nFitting Parameters:")
    print(df)

    print("\n" + "="*70)
    print("✅ Example completed successfully!")
    print("="*70)
