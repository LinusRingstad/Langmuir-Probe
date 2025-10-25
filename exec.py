import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import os
import analysis_marisol as am
from analysis_marisol import fit_data_to_polynomial, find_V_floating, Vplasma_from_model, get_j_i_star, find_Te_fit, find_Te_n_e_iteration, get_Te_ne_matrix
from constants import e, k, m_i, mu, A_probe, r_p, l_p, k_eV, K_to_eV



if False:
    print_eedf_whole()



if False:
    pressures = [10, 40, 70, 100]
    powers = [400, 600, 800, 1000]
    Te_matrix, ne_matrix, Te_std_matrix, ne_std_matrix = am.get_Te_ne_matrix()
        
    am.plot_Te_ne_vs_pressure("Te", "Pressure", pressures, powers, Te_matrix, Te_std_matrix)
    am.plot_Te_ne_vs_pressure("ne", "Pressure", pressures, powers, ne_matrix, ne_std_matrix)

    am.plot_Te_ne_vs_pressure("Te", "Power", pressures, powers, Te_matrix, Te_std_matrix)
    am.plot_Te_ne_vs_pressure("ne", "Power", pressures, powers, ne_matrix, ne_std_matrix)   


if False:
    positions = [0,2,4,6]
    pressures = [40,100]
    powers = [400, 600, 800, 1000]

    Te_matrix_list, ne_matrix_list, Te_std_matrix_list, ne_std_matrix_list, positions = am.get_Te_ne_matrix_vs_position()

    for i, x in enumerate(positions):
        Te_matrix = np.array(Te_matrix_list[i])
        print(Te_matrix.shape)
        Te_std_matrix = Te_std_matrix_list[i]
        ne_matrix = ne_matrix_list[i]
        ne_std_matrix = ne_std_matrix_list[i]

        am.plot_Te_ne_vs_pressure("Te", "Pressure", pressures, powers, Te_matrix, Te_std_matrix, x)
        am.plot_Te_ne_vs_pressure("ne", "Pressure", pressures, powers, ne_matrix, ne_std_matrix, x)

        am.plot_Te_ne_vs_pressure("Te", "Power", pressures, powers, Te_matrix, Te_std_matrix, x)
        am.plot_Te_ne_vs_pressure("ne", "Power", pressures, powers, ne_matrix, ne_std_matrix, x)
