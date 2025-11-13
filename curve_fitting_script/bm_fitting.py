# -*- coding: utf-8 -*-
"""
Birch-Murnaghan Equation Fitting for Pressure-Volume Curves
For fitting pressure-volume data and calculating bulk modulus parameters
@author: candicewang928@gmail.com
Created: 2025-11-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Configure matplotlib to properly display special characters and symbols
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
# Enable proper rendering of subscripts and superscripts
plt.rcParams['mathtext.default'] = 'regular'


def birch_murnaghan_2nd(V, V0, B0):
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


def birch_murnaghan_3rd(V, V0, B0, B0_prime):
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


def fit_bm_equations(V_data, P_data, phase_name=""):
    """
    Fit 2nd and 3rd order Birch-Murnaghan equations to P-V data

    Parameters:
    -----------
    V_data : array
        Volume data array
    P_data : array
        Pressure data array
    phase_name : str
        Phase name for output display

    Returns:
    --------
    results : dict
        Dictionary containing fitting parameters and statistics
    """
    results = {}

    # Initial guess values
    V0_guess = np.max(V_data) * 1.02  # Zero-pressure volume slightly larger than max volume
    B0_guess = 150  # Initial guess for bulk modulus (GPa)
    B0_prime_guess = 4.0  # Initial guess for first derivative of bulk modulus

    # ==================== 2nd order BM equation fitting ====================
    # Set reasonable parameter bounds to avoid overfitting
    # V0: 0.8-1.3 times the maximum experimental volume
    # B0: 50-500 GPa (covers most materials)
    bounds_2nd = ([np.max(V_data) * 0.8, 50],
                  [np.max(V_data) * 1.3, 500])

    try:
        popt_2nd, pcov_2nd = curve_fit(
            birch_murnaghan_2nd,
            V_data,
            P_data,
            p0=[V0_guess, B0_guess],
            bounds=bounds_2nd,
            maxfev=10000
        )

        V0_2nd, B0_2nd = popt_2nd
        perr_2nd = np.sqrt(np.diag(pcov_2nd))

        # Calculate fitting residuals and R²
        P_fit_2nd = birch_murnaghan_2nd(V_data, *popt_2nd)
        residuals_2nd = P_data - P_fit_2nd
        ss_res_2nd = np.sum(residuals_2nd**2)
        ss_tot_2nd = np.sum((P_data - np.mean(P_data))**2)
        r_squared_2nd = 1 - (ss_res_2nd / ss_tot_2nd)
        rmse_2nd = np.sqrt(np.mean(residuals_2nd**2))

        results['2nd_order'] = {
            'V0': V0_2nd,
            'V0_err': perr_2nd[0],
            'B0': B0_2nd,
            'B0_err': perr_2nd[1],
            'B0_prime': 4.0,  # Fixed to 4 for 2nd order equation
            'B0_prime_err': 0,
            'R_squared': r_squared_2nd,
            'RMSE': rmse_2nd,
            'fitted_P': P_fit_2nd
        }

        print(f"\n{'='*60}")
        print(f"{phase_name} - 2nd Order Birch-Murnaghan Fitting Results:")
        print(f"{'='*60}")
        print(f"V₀ = {V0_2nd:.4f} ± {perr_2nd[0]:.4f} Å³/atom")
        print(f"B₀ = {B0_2nd:.2f} ± {perr_2nd[1]:.2f} GPa")
        print(f"B₀' = 4.0 (fixed)")
        print(f"R² = {r_squared_2nd:.6f}")
        print(f"RMSE = {rmse_2nd:.4f} GPa")

    except Exception as e:
        print(f"⚠️ {phase_name} - 2nd order BM fitting failed: {e}")
        results['2nd_order'] = None

    # ==================== 3rd order BM equation fitting ====================
    # Set reasonable parameter bounds
    # V0: 0.8-1.3 times the maximum experimental volume
    # B0: 50-500 GPa
    # B0': 2.5-6.5 (based on literature values, most materials between 3-6)
    bounds_3rd = ([np.max(V_data) * 0.8, 50, 2.5],
                  [np.max(V_data) * 1.3, 500, 6.5])

    try:
        popt_3rd, pcov_3rd = curve_fit(
            birch_murnaghan_3rd,
            V_data,
            P_data,
            p0=[V0_guess, B0_guess, B0_prime_guess],
            bounds=bounds_3rd,
            maxfev=10000
        )

        V0_3rd, B0_3rd, B0_prime_3rd = popt_3rd
        perr_3rd = np.sqrt(np.diag(pcov_3rd))

        # Calculate fitting residuals and R²
        P_fit_3rd = birch_murnaghan_3rd(V_data, *popt_3rd)
        residuals_3rd = P_data - P_fit_3rd
        ss_res_3rd = np.sum(residuals_3rd**2)
        ss_tot_3rd = np.sum((P_data - np.mean(P_data))**2)
        r_squared_3rd = 1 - (ss_res_3rd / ss_tot_3rd)
        rmse_3rd = np.sqrt(np.mean(residuals_3rd**2))

        results['3rd_order'] = {
            'V0': V0_3rd,
            'V0_err': perr_3rd[0],
            'B0': B0_3rd,
            'B0_err': perr_3rd[1],
            'B0_prime': B0_prime_3rd,
            'B0_prime_err': perr_3rd[2],
            'R_squared': r_squared_3rd,
            'RMSE': rmse_3rd,
            'fitted_P': P_fit_3rd
        }

        print(f"\n{'='*60}")
        print(f"{phase_name} - 3rd Order Birch-Murnaghan Fitting Results:")
        print(f"{'='*60}")
        print(f"V₀ = {V0_3rd:.4f} ± {perr_3rd[0]:.4f} Å³/atom")
        print(f"B₀ = {B0_3rd:.2f} ± {perr_3rd[1]:.2f} GPa")
        print(f"B₀' = {B0_prime_3rd:.3f} ± {perr_3rd[2]:.3f}")
        print(f"R² = {r_squared_3rd:.6f}")
        print(f"RMSE = {rmse_3rd:.4f} GPa")

    except Exception as e:
        print(f"⚠️ {phase_name} - 3rd order BM fitting failed: {e}")
        results['3rd_order'] = None

    return results


def plot_pv_curves(V_orig, P_orig, V_new, P_new,
                   results_orig, results_new, save_dir):
    """
    Plot P-V curves and fitting results

    Parameters:
    -----------
    V_orig, P_orig : array
        Volume and pressure data for original phase
    V_new, P_new : array
        Volume and pressure data for new phase
    results_orig, results_new : dict
        Fitting results
    save_dir : str
        Directory to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Birch-Murnaghan Equation Fitting for P-V Curves',
                 fontsize=16, fontweight='bold')

    # Original phase - 2nd order BM
    ax = axes[0, 0]
    ax.scatter(V_orig, P_orig, s=80, c='blue', marker='o',
               label='Experimental Data (Original Phase)', alpha=0.7, edgecolors='black')
    if results_orig['2nd_order'] is not None:
        V_fit = np.linspace(V_orig.min()*0.95, V_orig.max()*1.05, 200)
        P_fit = birch_murnaghan_2nd(V_fit,
                                     results_orig['2nd_order']['V0'],
                                     results_orig['2nd_order']['B0'])
        ax.plot(V_fit, P_fit, 'r-', linewidth=2.5, label='2nd Order BM Fit', alpha=0.8)

        # Add fitting parameters text
        textstr = f"$V_0$ = {results_orig['2nd_order']['V0']:.4f} Å³/atom\n"
        textstr += f"$B_0$ = {results_orig['2nd_order']['B0']:.2f} GPa\n"
        textstr += f"$B_0'$ = 4.0 (fixed)\n"
        textstr += f"$R^2$ = {results_orig['2nd_order']['R_squared']:.6f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
    ax.set_ylabel('Pressure P (GPa)', fontsize=12)
    ax.set_title('Original Phase - 2nd Order BM Equation', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Original phase - 3rd order BM
    ax = axes[0, 1]
    ax.scatter(V_orig, P_orig, s=80, c='blue', marker='o',
               label='Experimental Data (Original Phase)', alpha=0.7, edgecolors='black')
    if results_orig['3rd_order'] is not None:
        V_fit = np.linspace(V_orig.min()*0.95, V_orig.max()*1.05, 200)
        P_fit = birch_murnaghan_3rd(V_fit,
                                     results_orig['3rd_order']['V0'],
                                     results_orig['3rd_order']['B0'],
                                     results_orig['3rd_order']['B0_prime'])
        ax.plot(V_fit, P_fit, 'g-', linewidth=2.5, label='3rd Order BM Fit', alpha=0.8)

        textstr = f"$V_0$ = {results_orig['3rd_order']['V0']:.4f} Å³/atom\n"
        textstr += f"$B_0$ = {results_orig['3rd_order']['B0']:.2f} GPa\n"
        textstr += f"$B_0'$ = {results_orig['3rd_order']['B0_prime']:.3f}\n"
        textstr += f"$R^2$ = {results_orig['3rd_order']['R_squared']:.6f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

    ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
    ax.set_ylabel('Pressure P (GPa)', fontsize=12)
    ax.set_title('Original Phase - 3rd Order BM Equation', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # New phase - 2nd order BM
    ax = axes[1, 0]
    ax.scatter(V_new, P_new, s=80, c='red', marker='s',
               label='Experimental Data (New Phase)', alpha=0.7, edgecolors='black')
    if results_new['2nd_order'] is not None:
        V_fit = np.linspace(V_new.min()*0.95, V_new.max()*1.05, 200)
        P_fit = birch_murnaghan_2nd(V_fit,
                                     results_new['2nd_order']['V0'],
                                     results_new['2nd_order']['B0'])
        ax.plot(V_fit, P_fit, 'r-', linewidth=2.5, label='2nd Order BM Fit', alpha=0.8)

        textstr = f"$V_0$ = {results_new['2nd_order']['V0']:.4f} Å³/atom\n"
        textstr += f"$B_0$ = {results_new['2nd_order']['B0']:.2f} GPa\n"
        textstr += f"$B_0'$ = 4.0 (fixed)\n"
        textstr += f"$R^2$ = {results_new['2nd_order']['R_squared']:.6f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
    ax.set_ylabel('Pressure P (GPa)', fontsize=12)
    ax.set_title('New Phase - 2nd Order BM Equation', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # New phase - 3rd order BM
    ax = axes[1, 1]
    ax.scatter(V_new, P_new, s=80, c='red', marker='s',
               label='Experimental Data (New Phase)', alpha=0.7, edgecolors='black')
    if results_new['3rd_order'] is not None:
        V_fit = np.linspace(V_new.min()*0.95, V_new.max()*1.05, 200)
        P_fit = birch_murnaghan_3rd(V_fit,
                                     results_new['3rd_order']['V0'],
                                     results_new['3rd_order']['B0'],
                                     results_new['3rd_order']['B0_prime'])
        ax.plot(V_fit, P_fit, 'g-', linewidth=2.5, label='3rd Order BM Fit', alpha=0.8)

        textstr = f"$V_0$ = {results_new['3rd_order']['V0']:.4f} Å³/atom\n"
        textstr += f"$B_0$ = {results_new['3rd_order']['B0']:.2f} GPa\n"
        textstr += f"$B_0'$ = {results_new['3rd_order']['B0_prime']:.3f}\n"
        textstr += f"$R^2$ = {results_new['3rd_order']['R_squared']:.6f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

    ax.set_xlabel('Volume V (Å³/atom)', fontsize=12)
    ax.set_ylabel('Pressure P (GPa)', fontsize=12)
    ax.set_title('New Phase - 3rd Order BM Equation', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(save_dir, 'BM_fitting_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ P-V curve figure saved to: {output_path}")

    plt.show()


def plot_residuals(V_orig, P_orig, V_new, P_new,
                   results_orig, results_new, save_dir):
    """
    Plot fitting residuals analysis

    Parameters:
    -----------
    V_orig, P_orig : array
        Volume and pressure data for original phase
    V_new, P_new : array
        Volume and pressure data for new phase
    results_orig, results_new : dict
        Fitting results
    save_dir : str
        Directory to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Fitting Residuals Analysis', fontsize=16, fontweight='bold')

    # Original phase - 2nd order BM residuals
    ax = axes[0, 0]
    if results_orig['2nd_order'] is not None:
        residuals = P_orig - results_orig['2nd_order']['fitted_P']
        ax.scatter(V_orig, residuals, s=60, c='blue', marker='o', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
        ax.set_ylabel('Residuals (GPa)', fontsize=11)
        ax.set_title('Original Phase - 2nd Order BM Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        textstr = f"RMSE = {results_orig['2nd_order']['RMSE']:.4f} GPa"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    # Original phase - 3rd order BM residuals
    ax = axes[0, 1]
    if results_orig['3rd_order'] is not None:
        residuals = P_orig - results_orig['3rd_order']['fitted_P']
        ax.scatter(V_orig, residuals, s=60, c='blue', marker='o', alpha=0.7)
        ax.axhline(y=0, color='g', linestyle='--', linewidth=2)
        ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
        ax.set_ylabel('Residuals (GPa)', fontsize=11)
        ax.set_title('Original Phase - 3rd Order BM Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        textstr = f"RMSE = {results_orig['3rd_order']['RMSE']:.4f} GPa"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

    # New phase - 2nd order BM residuals
    ax = axes[1, 0]
    if results_new['2nd_order'] is not None:
        residuals = P_new - results_new['2nd_order']['fitted_P']
        ax.scatter(V_new, residuals, s=60, c='red', marker='s', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
        ax.set_ylabel('Residuals (GPa)', fontsize=11)
        ax.set_title('New Phase - 2nd Order BM Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        textstr = f"RMSE = {results_new['2nd_order']['RMSE']:.4f} GPa"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    # New phase - 3rd order BM residuals
    ax = axes[1, 1]
    if results_new['3rd_order'] is not None:
        residuals = P_new - results_new['3rd_order']['fitted_P']
        ax.scatter(V_new, residuals, s=60, c='red', marker='s', alpha=0.7)
        ax.axhline(y=0, color='g', linestyle='--', linewidth=2)
        ax.set_xlabel('Volume V (Å³/atom)', fontsize=11)
        ax.set_ylabel('Residuals (GPa)', fontsize=11)
        ax.set_title('New Phase - 3rd Order BM Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        textstr = f"RMSE = {results_new['3rd_order']['RMSE']:.4f} GPa"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    output_path = os.path.join(save_dir, 'BM_fitting_residuals.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Residuals figure saved to: {output_path}")

    plt.show()


def save_results_to_csv(results_orig, results_new, save_dir):
    """
    Save fitting results to CSV file

    Parameters:
    -----------
    results_orig : dict
        Fitting results for original phase
    results_new : dict
        Fitting results for new phase
    save_dir : str
        Directory to save the CSV file

    Returns:
    --------
    df_summary : DataFrame
        Summary dataframe of fitting parameters
    """
    # Create summary table
    summary_data = []

    for phase_name, results in [('Original Phase', results_orig), ('New Phase', results_new)]:
        for order in ['2nd_order', '3rd_order']:
            if results[order] is not None:
                row = {
                    'Phase': phase_name,
                    'Fitting Order': '2nd Order' if order == '2nd_order' else '3rd Order',
                    'V₀ (Å³/atom)': f"{results[order]['V0']:.6f}",
                    'V₀ Error': f"{results[order]['V0_err']:.6f}",
                    'B₀ (GPa)': f"{results[order]['B0']:.4f}",
                    'B₀ Error': f"{results[order]['B0_err']:.4f}",
                    "B₀'": f"{results[order]['B0_prime']:.6f}",
                    "B₀' Error": f"{results[order]['B0_prime_err']:.6f}",
                    'R²': f"{results[order]['R_squared']:.8f}",
                    'RMSE (GPa)': f"{results[order]['RMSE']:.6f}"
                }
                summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)

    output_path = os.path.join(save_dir, 'BM_fitting_parameters.csv')
    df_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Fitting parameters saved to: {output_path}")

    return df_summary


def main():
    """
    Main function: Read data, fit, plot, and save results
    """
    print("\n" + "="*80)
    print("Birch-Murnaghan Equation Fitting for P-V Curves")
    print("="*80)

    # Set data paths (modify according to your actual data directory)
    data_dir = r"D:\HEPS\ID31\dioptas_data\Al0"  # Modify to your data directory
    orig_file = os.path.join(data_dir, "all_results_original_peaks_lattice.csv")
    new_file = os.path.join(data_dir, "all_results_new_peaks_lattice.csv")

    # Create output directory
    save_dir = os.path.join(data_dir, "BM_fitting_output")
    os.makedirs(save_dir, exist_ok=True)

    # Read data
    print(f"\n📂 Reading data files...")
    print(f"   Original phase data: {orig_file}")
    print(f"   New phase data: {new_file}")

    try:
        df_orig = pd.read_csv(orig_file)
        df_new = pd.read_csv(new_file)
        print("✅ Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: Data files not found")
        print(f"   Please ensure the following files exist:")
        print(f"   - {orig_file}")
        print(f"   - {new_file}")
        print(f"\n💡 Tip: Please modify the data_dir variable in main() function to your actual data directory")
        return

    # Check required columns
    required_columns = ['V_atomic', 'Pressure (GPa)']
    for col in required_columns:
        if col not in df_orig.columns or col not in df_new.columns:
            print(f"❌ Error: Required column '{col}' missing in data files")
            print(f"   Original phase columns: {df_orig.columns.tolist()}")
            print(f"   New phase columns: {df_new.columns.tolist()}")
            return

    # Extract data and remove null values
    V_orig = df_orig['V_atomic'].dropna().values
    P_orig = df_orig['Pressure (GPa)'].dropna().values
    V_new = df_new['V_atomic'].dropna().values
    P_new = df_new['Pressure (GPa)'].dropna().values

    # Ensure data pairing
    min_len_orig = min(len(V_orig), len(P_orig))
    V_orig = V_orig[:min_len_orig]
    P_orig = P_orig[:min_len_orig]

    min_len_new = min(len(V_new), len(P_new))
    V_new = V_new[:min_len_new]
    P_new = P_new[:min_len_new]

    print(f"\n📊 Data Overview:")
    print(f"   Original phase data points: {len(V_orig)}")
    print(f"   New phase data points: {len(V_new)}")
    print(f"   Original phase volume range: {V_orig.min():.4f} - {V_orig.max():.4f} Å³/atom")
    print(f"   Original phase pressure range: {P_orig.min():.2f} - {P_orig.max():.2f} GPa")
    print(f"   New phase volume range: {V_new.min():.4f} - {V_new.max():.4f} Å³/atom")
    print(f"   New phase pressure range: {P_new.min():.2f} - {P_new.max():.2f} GPa")

    # Perform fitting
    print(f"\n🔧 Starting Birch-Murnaghan equation fitting...")
    results_orig = fit_bm_equations(V_orig, P_orig, "Original Phase")
    results_new = fit_bm_equations(V_new, P_new, "New Phase")

    # Plot P-V curves
    print(f"\n📈 Plotting P-V curves...")
    plot_pv_curves(V_orig, P_orig, V_new, P_new,
                   results_orig, results_new, save_dir)

    # Plot residuals
    print(f"\n📉 Plotting residuals...")
    plot_residuals(V_orig, P_orig, V_new, P_new,
                   results_orig, results_new, save_dir)

    # Save results
    print(f"\n💾 Saving fitting parameters...")
    df_summary = save_results_to_csv(results_orig, results_new, save_dir)

    print(f"\n{'='*80}")
    print("✨ All tasks completed!")
    print(f"{'='*80}")
    print(f"📁 Output directory: {save_dir}")
    print(f"   - BM_fitting_results.png : P-V curve fitting plots")
    print(f"   - BM_fitting_residuals.png : Residual analysis plots")
    print(f"   - BM_fitting_parameters.csv : Fitting parameters summary table")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
