# -*- coding: utf-8 -*-
"""
Birch-Murnaghan Equation of State Fitting
Fits pressure-volume data to 2nd or 3rd order BM EOS
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit


class BirchMurnaghanFitter:
    """Birch-Murnaghan equation of state fitter"""

    def __init__(self, output_dir, order=3):
        """
        Initialize BM fitter

        Parameters:
        -----------
        output_dir : str
            Output directory for results
        order : int
            Order of BM EOS (2 or 3)
        """
        self.output_dir = output_dir
        self.order = order
        self.P = None
        self.V = None
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def birch_murnaghan_3rd(V, V0, K0, K0_prime):
        """
        3rd order Birch-Murnaghan equation

        Parameters:
        -----------
        V : array-like
            Volume
        V0 : float
            Reference volume at zero pressure
        K0 : float
            Bulk modulus at zero pressure (GPa)
        K0_prime : float
            Pressure derivative of bulk modulus

        Returns:
        --------
        P : array-like
            Pressure (GPa)
        """
        x = (V0 / V) ** (2/3)
        P = 1.5 * K0 * (x**3.5 - x**2.5) * (1 + 0.75 * (K0_prime - 4) * (x - 1))
        return P

    @staticmethod
    def birch_murnaghan_2nd(V, V0, K0):
        """
        2nd order Birch-Murnaghan equation (K0' = 4)

        Parameters:
        -----------
        V : array-like
            Volume
        V0 : float
            Reference volume at zero pressure
        K0 : float
            Bulk modulus at zero pressure (GPa)

        Returns:
        --------
        P : array-like
            Pressure (GPa)
        """
        x = (V0 / V) ** (2/3)
        P = 1.5 * K0 * (x**3.5 - x**2.5)
        return P

    def generate_initial_guess(self):
        """Generate initial guess for fitting parameters"""
        # V0 is approximately the maximum volume
        V0_guess = np.max(self.V)

        # Estimate K0 from the slope near V0
        # For small compression: P ≈ K0 * (V0 - V) / V0
        idx_sort = np.argsort(self.V)
        V_sorted = self.V[idx_sort]
        P_sorted = self.P[idx_sort]

        # Use the first few points for linear fit
        n_points = min(5, len(V_sorted))
        if n_points >= 3:
            strain = 1 - V_sorted[-n_points:] / V0_guess
            pressure = P_sorted[-n_points:]
            # K0 ≈ dP/d(strain)
            K0_guess = np.polyfit(strain, pressure, 1)[0]
            K0_guess = max(K0_guess, 50)  # Reasonable minimum
        else:
            K0_guess = 100  # Default guess

        if self.order == 3:
            K0_prime_guess = 4.0  # Standard starting value
            return [V0_guess, K0_guess, K0_prime_guess]
        else:
            return [V0_guess, K0_guess]

    def fit(self, P=None, V=None, initial_guess=None):
        """
        Fit the Birch-Murnaghan equation to P-V data

        Parameters:
        -----------
        P : array-like, optional
            Pressure data (GPa)
        V : array-like, optional
            Volume data (Å³)
        initial_guess : list, optional
            Initial guess for parameters [V0, K0] or [V0, K0, K0']

        Returns:
        --------
        results : dict
            Fitting results including parameters and uncertainties
        """
        if P is not None:
            self.P = np.array(P)
        if V is not None:
            self.V = np.array(V)

        if self.P is None or self.V is None:
            raise ValueError("Pressure and volume data must be provided")

        # Generate initial guess if not provided
        if initial_guess is None:
            initial_guess = self.generate_initial_guess()

        # Define residual function
        def residuals(params):
            if self.order == 3:
                V0, K0, K0_prime = params
                P_fit = self.birch_murnaghan_3rd(self.V, V0, K0, K0_prime)
            else:
                V0, K0 = params
                P_fit = self.birch_murnaghan_2nd(self.V, V0, K0)
            return P_fit - self.P

        # Set bounds
        if self.order == 3:
            bounds = ([0, 0, 0], [np.inf, np.inf, 10])
        else:
            bounds = ([0, 0], [np.inf, np.inf])

        # Perform fitting
        result = least_squares(residuals, initial_guess, bounds=bounds)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        # Extract parameters
        params = result.x
        if self.order == 3:
            V0, K0, K0_prime = params
        else:
            V0, K0 = params
            K0_prime = 4.0  # Fixed for 2nd order

        # Calculate uncertainties
        try:
            # Jacobian matrix
            J = result.jac
            # Covariance matrix
            cov = np.linalg.inv(J.T @ J) * (result.fun @ result.fun) / (len(self.P) - len(params))
            perr = np.sqrt(np.diag(cov))

            if self.order == 3:
                V0_err, K0_err, K0_prime_err = perr
            else:
                V0_err, K0_err = perr
                K0_prime_err = 0.0
        except:
            V0_err = K0_err = K0_prime_err = 0.0

        # Calculate R-squared
        P_fit = self.birch_murnaghan_3rd(self.V, V0, K0, K0_prime) if self.order == 3 else self.birch_murnaghan_2nd(self.V, V0, K0)
        ss_res = np.sum((self.P - P_fit) ** 2)
        ss_tot = np.sum((self.P - np.mean(self.P)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Store results
        results = {
            'V0': V0,
            'V0_err': V0_err,
            'K0': K0,
            'K0_err': K0_err,
            'K0_prime': K0_prime,
            'K0_prime_err': K0_prime_err,
            'r_squared': r_squared,
            'residuals': result.fun
        }

        # Save results
        self._save_results(results)
        self._plot_results(results)

        return results

    def _save_results(self, results):
        """Save fitting results to CSV"""
        # Save parameters
        params_df = pd.DataFrame({
            'Parameter': ['V0', 'K0', "K0'", 'R²'],
            'Value': [results['V0'], results['K0'], results['K0_prime'], results['r_squared']],
            'Uncertainty': [results['V0_err'], results['K0_err'], results['K0_prime_err'], 0.0],
            'Unit': ['Å³', 'GPa', '-', '-']
        })
        params_file = os.path.join(self.output_dir, 'bm_parameters.csv')
        params_df.to_csv(params_file, index=False)
        print(f"✅ Parameters saved to {params_file}")

        # Save fitted curve
        V_fit = np.linspace(self.V.min(), self.V.max(), 200)
        if self.order == 3:
            P_fit = self.birch_murnaghan_3rd(V_fit, results['V0'], results['K0'], results['K0_prime'])
        else:
            P_fit = self.birch_murnaghan_2nd(V_fit, results['V0'], results['K0'])

        curve_df = pd.DataFrame({
            'Volume (Å³)': V_fit,
            'Pressure (GPa)': P_fit
        })
        curve_file = os.path.join(self.output_dir, 'bm_fitted_curve.csv')
        curve_df.to_csv(curve_file, index=False)

    def _plot_results(self, results):
        """Plot P-V data and fitted curve"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot data and fit
        V_fit = np.linspace(self.V.min(), self.V.max(), 200)
        if self.order == 3:
            P_fit = self.birch_murnaghan_3rd(V_fit, results['V0'], results['K0'], results['K0_prime'])
            title = f"3rd Order Birch-Murnaghan EOS\nV₀ = {results['V0']:.3f} ± {results['V0_err']:.3f} Å³\n"
            title += f"K₀ = {results['K0']:.2f} ± {results['K0_err']:.2f} GPa\n"
            title += f"K₀' = {results['K0_prime']:.3f} ± {results['K0_prime_err']:.3f}"
        else:
            P_fit = self.birch_murnaghan_2nd(V_fit, results['V0'], results['K0'])
            title = f"2nd Order Birch-Murnaghan EOS\nV₀ = {results['V0']:.3f} ± {results['V0_err']:.3f} Å³\n"
            title += f"K₀ = {results['K0']:.2f} ± {results['K0_err']:.2f} GPa\nK₀' = 4.0 (fixed)"

        ax1.plot(self.V, self.P, 'ko', markersize=8, label='Data')
        ax1.plot(V_fit, P_fit, 'r-', linewidth=2, label='BM Fit')
        ax1.set_xlabel('Volume (Å³)', fontsize=12)
        ax1.set_ylabel('Pressure (GPa)', fontsize=12)
        ax1.set_title(title, fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot residuals
        P_data_fit = self.birch_murnaghan_3rd(self.V, results['V0'], results['K0'], results['K0_prime']) \
                     if self.order == 3 else self.birch_murnaghan_2nd(self.V, results['V0'], results['K0'])
        residuals = self.P - P_data_fit

        ax2.plot(self.P, residuals, 'bo', markersize=8)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Pressure (GPa)', fontsize=12)
        ax2.set_ylabel('Residuals (GPa)', fontsize=12)
        ax2.set_title(f'Residuals (R² = {results["r_squared"]:.6f})', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'bm_fit_plot.png')
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"✅ Plot saved to {plot_file}")
