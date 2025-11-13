# -*- coding: utf-8 -*-
"""
Example Usage of BMFitter Class
Demonstrates various ways to use the encapsulated BM fitting module
@author: candicewang928@gmail.com
Created: 2025-11-13
"""

import numpy as np
import pandas as pd
import os
from bm_fitting_class import BMFitter, BMMultiPhaseFitter, quick_fit


def example_1_basic_usage():
    """
    Example 1: Basic usage with single phase fitting
    """
    print("\n" + "="*70)
    print("Example 1: Basic Single Phase Fitting")
    print("="*70 + "\n")

    # Sample data (V in Å³/atom, P in GPa)
    V_data = np.array([16.5432, 16.2341, 15.9876, 15.6543, 15.3421,
                      15.0567, 14.7892, 14.5234, 14.2876, 14.0543])
    P_data = np.array([0.5, 2.3, 5.1, 8.7, 12.4, 16.8, 21.5, 26.9, 32.7, 39.2])

    # Create fitter object
    fitter = BMFitter(V_data, P_data, phase_name="Sample Phase")

    # Fit both orders
    fitter.fit_both_orders()

    # Compare results
    fitter.compare_fits()

    # Plot results
    fitter.plot_pv_curve(save_path="example1_pv_curve.png")
    fitter.plot_residuals(save_path="example1_residuals.png")

    # Get parameters as DataFrame
    df = fitter.get_parameters()
    print("\nFitting Parameters:")
    print(df)

    # Save parameters
    fitter.save_parameters("example1_parameters.csv")

    return fitter


def example_2_custom_bounds():
    """
    Example 2: Fitting with custom parameter bounds
    """
    print("\n" + "="*70)
    print("Example 2: Fitting with Custom Bounds")
    print("="*70 + "\n")

    # Sample data
    V_data = np.array([16.5, 16.0, 15.5, 15.0, 14.5, 14.0])
    P_data = np.array([1.0, 5.0, 10.0, 16.0, 23.0, 31.0])

    fitter = BMFitter(V_data, P_data, phase_name="Custom Bounds Phase")

    # Fit 2nd order with custom bounds
    fitter.fit_2nd_order(
        V0_bounds=(14.0, 17.0),      # Custom V0 range
        B0_bounds=(100, 300),         # Custom B0 range
        initial_guess=(16.5, 150)     # Custom initial guess
    )

    # Fit 3rd order with custom bounds
    fitter.fit_3rd_order(
        V0_bounds=(14.0, 17.0),
        B0_bounds=(100, 300),
        B0_prime_bounds=(3.5, 5.0),   # Narrower B0' range
        initial_guess=(16.5, 150, 4.0)
    )

    # Plot
    fitter.plot_pv_curve(save_path="example2_pv_curve.png")

    return fitter


def example_3_from_csv():
    """
    Example 3: Load data from CSV file and fit
    """
    print("\n" + "="*70)
    print("Example 3: Loading Data from CSV")
    print("="*70 + "\n")

    # Read CSV file (assuming it has 'V_atomic' and 'Pressure (GPa)' columns)
    # For this example, we'll create a sample CSV first
    sample_data = pd.DataFrame({
        'V_atomic': [16.5, 16.2, 15.9, 15.6, 15.3, 15.0, 14.7],
        'Pressure (GPa)': [0.5, 3.0, 6.5, 11.0, 16.5, 23.0, 30.5]
    })
    sample_data.to_csv('sample_data.csv', index=False)

    # Load data from CSV
    df = pd.read_csv('sample_data.csv')
    V_data = df['V_atomic'].values
    P_data = df['Pressure (GPa)'].values

    print(f"Loaded {len(V_data)} data points from CSV")

    # Fit
    fitter = BMFitter(V_data, P_data, phase_name="CSV Data Phase")
    fitter.fit_both_orders()

    # Plot
    fitter.plot_pv_curve(save_path="example3_pv_curve.png")

    return fitter


def example_4_multi_phase():
    """
    Example 4: Fitting multiple phases and comparing them
    """
    print("\n" + "="*70)
    print("Example 4: Multi-Phase Fitting and Comparison")
    print("="*70 + "\n")

    # Original phase data
    V_orig = np.array([16.8, 16.5, 16.2, 15.9, 15.6, 15.3, 15.0])
    P_orig = np.array([0.5, 2.5, 5.5, 9.5, 14.5, 20.5, 27.5])

    # New phase data (more compressible)
    V_new = np.array([15.5, 15.0, 14.5, 14.0, 13.5, 13.0])
    P_new = np.array([5.0, 10.0, 16.0, 23.0, 31.0, 40.0])

    # Create multi-phase fitter
    multi_fitter = BMMultiPhaseFitter()

    # Add phases
    multi_fitter.add_phase("Original Phase", V_orig, P_orig)
    multi_fitter.add_phase("New Phase", V_new, P_new)

    # Fit all phases
    multi_fitter.fit_all(verbose=True)

    # Plot comparison
    multi_fitter.plot_comparison(save_path="example4_comparison_2nd.png", order=2)
    multi_fitter.plot_comparison(save_path="example4_comparison_3rd.png", order=3)

    # Get all parameters
    df_all = multi_fitter.get_all_parameters()
    print("\nAll Phases Parameters:")
    print(df_all)

    # Save all parameters
    multi_fitter.save_all_parameters("example4_all_parameters.csv")

    return multi_fitter


def example_5_quick_fit():
    """
    Example 5: Using the quick_fit convenience function
    """
    print("\n" + "="*70)
    print("Example 5: Quick Fit Function")
    print("="*70 + "\n")

    # Sample data
    V_data = np.array([17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5])
    P_data = np.array([0.5, 3.0, 6.5, 11.0, 16.5, 23.0, 30.5, 39.0])

    # Quick fit - does everything in one function call
    fitter = quick_fit(
        V_data,
        P_data,
        phase_name="Quick Fit Phase",
        save_dir="./quick_fit_output",
        show_plots=False
    )

    return fitter


def example_6_programmatic_comparison():
    """
    Example 6: Programmatic comparison and selection
    """
    print("\n" + "="*70)
    print("Example 6: Programmatic Fit Comparison")
    print("="*70 + "\n")

    # Sample data
    V_data = np.array([16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5])
    P_data = np.array([1.0, 5.0, 10.0, 16.0, 23.0, 31.0, 40.0])

    fitter = BMFitter(V_data, P_data, phase_name="Comparison Phase")
    fitter.fit_both_orders()

    # Compare and get recommendation
    recommendation = fitter.compare_fits()

    # Access specific results
    if fitter.results_2nd and fitter.results_2nd.get('success'):
        print(f"\n2nd Order Results:")
        print(f"  V₀ = {fitter.results_2nd['V0']:.4f} ± {fitter.results_2nd['V0_err']:.4f} Å³/atom")
        print(f"  B₀ = {fitter.results_2nd['B0']:.2f} ± {fitter.results_2nd['B0_err']:.2f} GPa")
        print(f"  R² = {fitter.results_2nd['R_squared']:.6f}")

    if fitter.results_3rd and fitter.results_3rd.get('success'):
        print(f"\n3rd Order Results:")
        print(f"  V₀ = {fitter.results_3rd['V0']:.4f} ± {fitter.results_3rd['V0_err']:.4f} Å³/atom")
        print(f"  B₀ = {fitter.results_3rd['B0']:.2f} ± {fitter.results_3rd['B0_err']:.2f} GPa")
        print(f"  B₀' = {fitter.results_3rd['B0_prime']:.3f} ± {fitter.results_3rd['B0_prime_err']:.3f}")
        print(f"  R² = {fitter.results_3rd['R_squared']:.6f}")

    return fitter


def example_7_real_world_workflow():
    """
    Example 7: Real-world workflow with original and new phase data
    """
    print("\n" + "="*70)
    print("Example 7: Real-World Workflow")
    print("="*70 + "\n")

    # Simulate loading data from your actual CSV files
    # In real usage, you would load from actual files:
    # df_orig = pd.read_csv("all_results_original_peaks_lattice.csv")
    # df_new = pd.read_csv("all_results_new_peaks_lattice.csv")

    # For this example, create simulated data
    df_orig = pd.DataFrame({
        'V_atomic': [16.8, 16.5, 16.2, 15.9, 15.6, 15.3, 15.0, 14.7],
        'Pressure (GPa)': [0.5, 2.5, 5.5, 9.5, 14.5, 20.5, 27.5, 35.5]
    })

    df_new = pd.DataFrame({
        'V_atomic': [15.5, 15.0, 14.5, 14.0, 13.5, 13.0, 12.5],
        'Pressure (GPa)': [5.0, 10.0, 16.0, 23.0, 31.0, 40.0, 50.0]
    })

    # Extract data
    V_orig = df_orig['V_atomic'].dropna().values
    P_orig = df_orig['Pressure (GPa)'].dropna().values
    V_new = df_new['V_atomic'].dropna().values
    P_new = df_new['Pressure (GPa)'].dropna().values

    # Create output directory
    output_dir = "./real_world_output"
    os.makedirs(output_dir, exist_ok=True)

    # Fit original phase
    print("\n" + "-"*60)
    print("Fitting Original Phase")
    print("-"*60)
    fitter_orig = BMFitter(V_orig, P_orig, phase_name="Original Phase")
    fitter_orig.fit_both_orders()
    fitter_orig.compare_fits()
    fitter_orig.plot_pv_curve(save_path=os.path.join(output_dir, "original_pv_curve.png"))
    fitter_orig.plot_residuals(save_path=os.path.join(output_dir, "original_residuals.png"))
    fitter_orig.save_parameters(os.path.join(output_dir, "original_parameters.csv"))

    # Fit new phase
    print("\n" + "-"*60)
    print("Fitting New Phase")
    print("-"*60)
    fitter_new = BMFitter(V_new, P_new, phase_name="New Phase")
    fitter_new.fit_both_orders()
    fitter_new.compare_fits()
    fitter_new.plot_pv_curve(save_path=os.path.join(output_dir, "new_pv_curve.png"))
    fitter_new.plot_residuals(save_path=os.path.join(output_dir, "new_residuals.png"))
    fitter_new.save_parameters(os.path.join(output_dir, "new_parameters.csv"))

    # Combined comparison
    print("\n" + "-"*60)
    print("Creating Combined Comparison")
    print("-"*60)
    multi_fitter = BMMultiPhaseFitter()
    multi_fitter.add_phase("Original Phase", V_orig, P_orig)
    multi_fitter.add_phase("New Phase", V_new, P_new)
    multi_fitter.fit_all(verbose=False)
    multi_fitter.plot_comparison(save_path=os.path.join(output_dir, "phases_comparison.png"), order=3)
    multi_fitter.save_all_parameters(os.path.join(output_dir, "all_phases_parameters.csv"))

    print(f"\n✅ All results saved to: {output_dir}")

    return fitter_orig, fitter_new, multi_fitter


def main():
    """
    Main function to run all examples
    """
    print("\n" + "#"*70)
    print("# BMFitter Class - Comprehensive Examples")
    print("#"*70)

    # Run examples
    try:
        # Uncomment the examples you want to run:

        # example_1_basic_usage()
        # example_2_custom_bounds()
        # example_3_from_csv()
        # example_4_multi_phase()
        # example_5_quick_fit()
        # example_6_programmatic_comparison()
        example_7_real_world_workflow()

        print("\n" + "#"*70)
        print("# All examples completed successfully!")
        print("#"*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
