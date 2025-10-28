import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from basic import read_file,plasma_potential, floating_potential, guess_Te, guess_ne, interp_laframboise


def find_ne_te_iterative(settings, max_iterations=1000, tolerance=1e-4):
    """
    Iteratively find the final electron density (n_e) and electron temperature (T_e).
    
    Args:
        V: Voltage array (V).
        I : Current array (A).
        settings: Settings object containing constants and parameters.
        max_iterations: Maximum number of iterations
        tolerance: Convergence (relative) tolerance for T_e and n_e .
    
    Returns:
        Final electron temperature in eV.
        Final electron density in m^-3.
    """
    # Initial guesses for T_e and n_e

    V,I,coeffs = read_file(settings)


    Vp, Ip = plasma_potential(V, coeffs)  # Plasma potential and saturation current
    Vf = floating_potential(V, coeffs)   # Floating potential
    T_e = guess_Te(Vp, Vf, settings)                 # Initial guess for T_e (eV)
    n_e = guess_ne(Ip, T_e, settings)                # Initial guess for n_e (m^-3)

    for iteration in range(max_iterations):
        T_e_prev = T_e
        n_e_prev = n_e


        # Interpolate Laframboise parameter and refine T_e and n_e. Ie is also given as to calculate eedf.
        T_e, n_e,Ie = interp_laframboise(V, I, Vp, Vf, T_e, n_e, settings)



        # Check for convergence
        if (abs(T_e - T_e_prev)/T_e < tolerance) and (abs(n_e - n_e_prev)/n_e < tolerance):
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print("Maximum iterations reached without convergence.")
    return T_e, n_e, Ie





