# -*- coding: utf-8 -*-
"""
Birch-Murnaghan Equation of State Fitting Module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


class BirchMurnaghanFitter:
    """Birch-Murnaghan equation of state fitter"""

    def __init__(self, data_file, output_dir, order=3):
        """
        Initialize BM fitter

        Parameters:
        -----------
        data_file : str
            Path to CSV file with Pressure and Volume columns
        output_dir : str
            Output directory for results
        order : int
            BM equation order (2 or 3)
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.order = order

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def bm2(V, V0, K0):
        """2nd order Birch-Murnaghan equation"""
        x = (V0 / V)**(2/3)
        return (3/2) * K0 * (x - 1)

    @staticmethod
    def bm3(V, V0, K0, K0_prime):
        """3rd order Birch-Murnaghan equation"""
        x = (V0 / V)**(2/3)
        return (3/2) * K0 * (x - 1) * (1 + (3/4) * (K0_prime - 4) * (x - 1))

    def fit(self, initial_guess=None):
        """
        Perform Birch-Murnaghan fitting

        Parameters:
        -----------
        initial_guess : tuple, optional
            Initial guess for fitting parameters
            For 2nd order: (V0, K0)
            For 3rd order: (V0, K0, K0_prime)

        Returns:
        --------
        dict: Fitting results
        """
        # Read data
        df = pd.read_csv(self.data_file)

        # Find pressure and volume columns (flexible naming)
        pressure_col = None
        volume_col = None

        for col in df.columns:
            col_lower = col.lower().replace(' ', '')
            if 'pressure' in col_lower or col_lower == 'p':
                pressure_col = col
            if 'volume' in col_lower or col_lower == 'v':
                volume_col = col

        if pressure_col is None or volume_col is None:
            raise ValueError(
                f"Could not find Pressure and Volume columns.\n"
                f"Found columns: {list(df.columns)}\n"
                f"Please ensure CSV has 'Pressure' and 'Volume' columns"
            )

        P = df[pressure_col].values
        V = df[volume_col].values

        # Remove any NaN values
        mask = ~(np.isnan(P) | np.isnan(V))
        P = P[mask]
        V = V[mask]

        if len(P) == 0:
            raise ValueError("No valid data points found")

        # Auto-generate initial guess if not provided
        if initial_guess is None:
            V0_guess = np.max(V)  # Assume max volume is near V0
            K0_guess = 100.0  # Typical bulk modulus in GPa

            if self.order == 2:
                initial_guess = (V0_guess, K0_guess)
            else:
                K0_prime_guess = 4.0
                initial_guess = (V0_guess, K0_guess, K0_prime_guess)

        # Perform fitting
        if self.order == 2:
            popt, pcov = curve_fit(self.bm2, V, P, p0=initial_guess, maxfev=10000)
            V0, K0 = popt
            V0_err, K0_err = np.sqrt(np.diag(pcov))
            K0_prime = 4.0  # Fixed for 2nd order
            K0_prime_err = 0.0

            P_fit = self.bm2(V, V0, K0)

        else:  # 3rd order
            popt, pcov = curve_fit(self.bm3, V, P, p0=initial_guess, maxfev=10000)
            V0, K0, K0_prime = popt
            V0_err, K0_err, K0_prime_err = np.sqrt(np.diag(pcov))

            P_fit = self.bm3(V, V0, K0, K0_prime)

        # Calculate R-squared
        ss_res = np.sum((P - P_fit)**2)
        ss_tot = np.sum((P - np.mean(P))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Save results
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

        # Save to CSV
        results_df = pd.DataFrame([results])
        results_csv = os.path.join(self.output_dir, 'bm_fit_results.csv')
        results_df.to_csv(results_csv, index=False)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(V, P, s=60, c='#B794F6', edgecolors='black', linewidth=1.5,
                    label='Experimental Data', zorder=3)

        V_fit = np.linspace(V.min(), V.max(), 200)
        if self.order == 2:
            P_fit_line = self.bm2(V_fit, V0, K0)
        else:
            P_fit_line = self.bm3(V_fit, V0, K0, K0_prime)

        plt.plot(V_fit, P_fit_line, 'r-', linewidth=2,
                 label=f'{self.order}rd Order BM Fit' if self.order == 3 else '2nd Order BM Fit')

        plt.xlabel('Volume (Ų)', fontsize=12, fontweight='bold')
        plt.ylabel('Pressure (GPa)', fontsize=12, fontweight='bold')
        plt.title(f'Birch-Murnaghan {self.order}{"rd" if self.order == 3 else "nd"} Order Fit',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add text box with results
        textstr = f'$V_0$ = {V0:.4f} ± {V0_err:.4f} ų\n'
        textstr += f'$K_0$ = {K0:.2f} ± {K0_err:.2f} GPa\n'
        textstr += f"$K_0'$ = {K0_prime:.3f} ± {K0_prime_err:.3f}\n"
        textstr += f'$R^2$ = {r_squared:.6f}'

        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plot_file = os.path.join(self.output_dir, 'bm_fit_plot.png')
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ BM fit results saved to: {results_csv}")
        print(f"✓ BM fit plot saved to: {plot_file}")

        return results
