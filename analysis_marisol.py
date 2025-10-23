import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from constants import e, k, m_i, mu, A_probe, r_p, l_p, k_eV, K_to_eV

def read_keithley_csv(path):
    """
        Reads the Keithley csv file and returns a pandas dataframe
        Args:
            path: path to the Keithley csv file
        Returns:
            df: pandas dataframe with the data
    """

    # find the real header row
    header_row = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.strip().startswith('"Index"'):
                header_row = i
                break

    df = pd.read_csv(path, skiprows=header_row, header=0, quotechar='"', skipfooter=5,engine='python')
    df.columns = [c.strip().strip('"') for c in df.columns]

    # keep only the three columns and coerce to numeric
    keep = ["SMU-1 Voltage (V)", "SMU-1 Current (A)"]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=keep).sort_values("SMU-1 Voltage (V)").reset_index(drop=True)
    return df

def fit_data_to_polynomial(path):
    """
        Fits the data to a polynomial and returns the coefficients and the model
        Args:
            path: path to the Keithley csv file
        Returns:
            I: current data
            V: voltage data
            coeffs: coefficients of the polynomial
            model: polynomial model
    """
    df = read_keithley_csv(path)

    V = df['SMU-1 Voltage (V)'].round(6) 
    I = df['SMU-1 Current (A)']

    coeffs = np.polyfit(V, I, deg=11)
    model = np.poly1d(coeffs)

    return I, V, coeffs, model

def find_V_floating(coeffs, domain=None, imag_tol=1e-9):
    """
    coeffs: np.polyfit/np.poly1d coefficients (highest power first)
    domain: (vmin, vmax) to keep only roots within your sweep range
    """
    roots = np.roots(coeffs)
    real_roots = roots[np.isclose(roots.imag, 0.0, atol=imag_tol)].real
    if domain is not None:
        vmin, vmax = domain
        real_roots = real_roots[(real_roots >= vmin) & (real_roots <= vmax)]
    return real_roots[0]

def Vplasma_from_model(model: np.poly1d, V, n=4000, trim=0.05):
    """
    Robust knee finder for I(V) given an np.poly1d 'model' and the measured V range.
    - Trims edges
    - Ignores very flat-slope regions
    - Falls back to 'kneedle' distance-to-chord if curvature mask is empty
    """
    V = np.asarray(V, float)
    vmin, vmax = float(np.min(V)), float(np.max(V))
    x = np.linspace(vmin, vmax, n)
    # Derivatives
    #dI = np.polyder(model, 1)
    d2I = np.polyder(model, 2)
    d3I = np.polyder(model, 3)   # third derivative

    # critical points of p'' are the real roots of p'''
    roots = np.roots(d3I)
    real_roots = roots[np.isclose(roots.imag, 0.0)].real
    # keep only inside trimmed domain
    cand = real_roots[(real_roots >= vmin) & (real_roots <= vmax)]

    # always consider boundaries of trimmed domain as candidates, too
    cand = np.unique(np.concatenate([cand, [vmin, vmax]]))

    # evaluate p'' on the candidates and pick the minimum
    d2I_vals = d2I(cand)
    idx = int(np.argmin(d2I_vals))
    v_star = float(cand[idx])
    
    return v_star, float(model(v_star)), float(d2I_vals[idx]), {
        "candidates": cand.tolist(),
        "trim": float(trim)
    }

def plot_dIdV2(model,V):
    """
    Plots the second derivative of the IV trace to check for the knee
    """
    dI = np.polyder(model,2)
    y2 = dI(V)
 
    plt.figure(figsize=(6,4))
    plt.plot(V, y2, '-', lw=1 , label='IV trace second derivative')
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_j_i_star(ratio, Xp_star):
    """
        Gets the j_i_star value from the Laframboise table
        Args:
            ratio: ratio of the plasma radius to the Debye length
            Xp_star: Xp_star value calculated as e*(V_plasma - V[200])/(k*T_e)
        Returns:
            j_i_star: j_i_star value
    """
    data_j_i_star = pd.read_csv('/Users/marisolvelapatino/Desktop/Langmuir-Probe/lafram.csv')
    
    # Get the X values from the second row (index 1)
    x_values = data_j_i_star.columns[1:].astype(float)  # Skip first column (X header)
    
    Xstarcol = data_j_i_star["Xp"]
    if Xp_star in Xstarcol:
        idx_Xstar = Xstarcol.index(Xp_star)
    else:
        closest_idx = np.argmin(np.abs(Xstarcol - Xp_star))
        idx_Xstar = closest_idx
    
    # Get the j_i_star values from the idx_Xstar row 
    j_i_star_values = data_j_i_star.iloc[idx_Xstar, 1:].values.astype(float)  # Skip first column
    
    
    # Check if ratio exactly matches any X value
    if ratio in x_values:
        idx = np.where(x_values == ratio)[0][0]
        j_i_star = j_i_star_values[idx]
        #print(f"Exact match found! j_i_star = {j_i_star}")
        return j_i_star
    else:
        # If no exact match, find the closest value
        closest_idx = np.argmin(np.abs(x_values - ratio))
        j_i_star = j_i_star_values[closest_idx]
        #print(f"No exact match. Closest X value: {x_values[closest_idx]}, j_i_star = {j_i_star}")
        return j_i_star

def find_Te_fit(I_e, V, V_plasma):
    """
        Finds the electron temperature from the I_e and V data using a linear fit
        Args:
            I_e: current data
            V: voltage data
            V_plasma: plasma potential
        Returns:
            T_e: electron temperature in K
    """
    x = (V - V_plasma)
    
    y = np.log(I_e)

    p = np.polyfit(x,y,1)

    T_e = e/(k*p[0]) #in K
    return T_e


def find_Te_n_e_iteration(path):
    """
        Finds the electron temperature and electron density from the I_e and V data using an iterative method (Laframboise method).
        Args:
            path: path to the Keithley csv file
        Returns:
            T_e: electron temperature in K
            n_e: electron density in m^-3
    """
    I, V, coeffs, model = fit_data_to_polynomial(path)

    ## Find the floating potential
    V_floating = find_V_floating(coeffs, domain=(min(V), max(V)))

    ## Find the plasma potential
    V_plasma, Iplasma, min_d2I, info = Vplasma_from_model(model, V)

    #print("Found V_plasma: ", V_plasma, " V , with V floating : ", V_floating)

    # Variables
    n_e = 0
    T_e = 0
    T_e_prev = np.inf
    I_ion = 0
    j_i_star = 0
    
    for i in range(1000):
        
        #get initial ion current:
        if i == 0:
            I_ion = I[20]
        else: 
            I_ion = 0.25*e*n_e*(np.sqrt(8*k*T_e/(np.pi*m_i)))*A_probe*j_i_star
        
        
        I_e = I - I_ion 
        ## Prepare I_e to not have negatives or zeros:
        I_e = np.where(I_e <= 0, 1e-7, I_e)
        T_e = find_Te_fit(I_e, V, V_plasma)
        T_e_eV = T_e*K_to_eV #in eV
        
        if i == 0:
            n_e = 1.05e15*np.sqrt(mu/T_e_eV)*I_e[200]/(A_probe)
        else:
            #print("n_e at iteration ", i, " is ", n_e)
            #print("I_e at iteration ", i, " is ", I_e[200])
            n_e = 1.05e15*np.sqrt(mu/T_e_eV)*I_e[200]/(A_probe*j_i_star) # TODO: Check if this is correct
     
        #print("Te at iteration ", i, " is ", T_e)
        if np.abs(T_e_prev - T_e) < 1e-6:
            break
        
        T_e_prev = T_e
        
        # Calculate ratio:
        lambda_D = 7430*np.sqrt(T_e_eV/n_e)
        ratio = (r_p/lambda_D) #TODO Temporal fix is to take the mean of the ratios
        # Calculate j_i_star:
        Xp_star = e*(V_plasma - V[200])/(k*T_e)
        j_i_star = get_j_i_star(ratio, Xp_star)

    return T_e, n_e, I_e

def get_Te_ne_matrix():
    """
        Gets the electron temperature and electron density matrix for different pressures and powers
        Args:
            None
        Returns:
            Te_matrix: electron temperature matrix
            ne_matrix: electron density matrix
            pressures: list of pressures
            powers: list of powers
    """
    parent_dir= "/Users/marisolvelapatino/Desktop/Langmuir-Probe/LP"
    pressures = [10, 40, 70, 100]
    powers = [400, 600, 800, 1000]
    #Te_matrix = np.empty((len(pressures), len(powers)), dtype=object)
    #ne_matrix = np.empty((len(pressures), len(powers)), dtype=object)

    #for i in range(len(pressures)):
    #    for j in range(len(powers)):
    #        Te_matrix[i, j] = []
    #        ne_matrix[i, j] = []

    Te_matrix = np.zeros((len(pressures), len(powers)))
    ne_matrix = np.zeros((len(pressures), len(powers)))

    Te_std_matrix = np.zeros((len(pressures), len(powers)))
    ne_std_matrix = np.zeros((len(pressures), len(powers)))


    for i,p in enumerate(pressures):
        path = parent_dir + f"/{p} mTorr"
        for j,pw in enumerate(powers):
            new_path = path + f"/{pw}W"
            Te_vals_list = []
            ne_vals_list = []
            for file in os.listdir(new_path):
                file = new_path + "/" + file
                Te, ne, I_e = find_Te_n_e_iteration(file)
                
                Te_vals_list.append(Te*K_to_eV)
                ne_vals_list.append(ne)
            
            Te_matrix[i,j] = np.mean(Te_vals_list)
            Te_std_matrix[i,j] = np.std(Te_vals_list)
            
            ne_matrix[i,j] = np.mean(ne_vals_list)
            ne_std_matrix[i,j] = np.std(ne_vals_list)

    return Te_matrix, ne_matrix, Te_std_matrix, ne_std_matrix

def get_Te_ne_matrix_vs_position():
    """
        Gets the electron temperature and electron density matrix for different positions
        Args:
            None
        Returns:
            Te_matrix_list: list of electron temperature matrices
            ne_matrix_list: list of electron density matrices
            positions: list of positions
    """
    parent_dir= "/Users/marisolvelapatino/Desktop/Langmuir-Probe/LP"
    positions = [2,4,6]
    pressures = [40,100]
    powers = [400, 600, 800, 1000]
    
    Te_matrix_list = []
    ne_matrix_list = []
    Te_std_matrix_list = []
    ne_std_matrix_list = []

    for k, x in enumerate(positions):
        base_path = parent_dir + f"/{x}in"
        
        # Prepare the matrices:
        Te_matrix = np.zeros((len(pressures), len(powers)))
        Te_std_matrix = np.zeros((len(pressures), len(powers)))
        ne_matrix = np.zeros((len(pressures), len(powers)))
        ne_std_matrix = np.zeros((len(pressures), len(powers)))
        
        for i,p in enumerate(pressures):
            new_path = base_path + f"/{p} mTorr"
            for j,pw in enumerate(powers):
                final_path = new_path + f"/{pw}W"
                Te_vals_list = []
                ne_vals_list = []
                for file in os.listdir(final_path):
                    final_file = final_path + "/" + file
                    Te, ne, I_e = find_Te_n_e_iteration(final_file)
                    Te_vals_list.append(Te*K_to_eV)
                    ne_vals_list.append(ne)
                
                Te_matrix[i, j] = np.mean(Te_vals_list)
                Te_std_matrix[i, j] = np.std(Te_vals_list)
                ne_matrix[i, j] = np.mean(ne_vals_list)
                ne_std_matrix[i, j] = np.std(ne_vals_list)

        Te_matrix_list.append(Te_matrix)
        ne_matrix_list.append(ne_matrix)
        Te_std_matrix_list.append(Te_std_matrix)
        ne_std_matrix_list.append(ne_std_matrix)

    return Te_matrix_list, ne_matrix_list, Te_std_matrix_list, ne_std_matrix_list, positions


def plot_Te_ne_vs_pressure(plot_name, p_name, pressures, powers, matrix, std_matrix, location=None):
    """
    p_name: the FIXED parameter ("Pressure" or "Power")
      - "Pressure" -> plot vs Power; one curve per fixed pressure
      - "Power"    -> plot vs Pressure; one curve per fixed power
    matrix, std_matrix shape: (len(pressures), len(powers))  # rows=pressures, cols=powers
    """
    pressures = np.asarray(pressures, dtype=float)
    powers    = np.asarray(powers, dtype=float)
    M = np.asarray(matrix, dtype=float)
    S = np.asarray(std_matrix, dtype=float)

    assert M.shape == (len(pressures), len(powers)), \
        f"matrix shape {M.shape} must be ({len(pressures)}, {len(powers)})"
    assert S.shape == M.shape, "std_matrix must match matrix shape"

    plt.figure(figsize=(10, 6))

    if p_name == "Pressure":
        # fixed pressure -> iterate pressures; x = powers; y = row
        x = powers
        xlabel = "Power (W)"
        for i, p in enumerate(pressures):
            y    = M[i, :]
            yerr = S[i, :]
            mask = np.isfinite(y) & np.isfinite(yerr)
            if not mask.any(): 
                continue
            (line,) = plt.plot(x[mask], y[mask], '-', alpha=0.85, lw=1.5,
                               label=f"Pressure = {p} mTorr")
            c = line.get_color()
            plt.errorbar(x[mask], y[mask], yerr=yerr[mask],
                         fmt='o', capsize=4, elinewidth=1,
                         color=c, ecolor=c, markeredgecolor=c, markerfacecolor=c)

    elif p_name == "Power":
        # fixed power -> iterate powers; x = pressures; y = column
        x = pressures
        xlabel = "Pressure (mTorr)"
        for j, pw in enumerate(powers):
            y    = M[:, j]
            yerr = S[:, j]
            mask = np.isfinite(y) & np.isfinite(yerr)
            if not mask.any(): 
                continue
            (line,) = plt.plot(x[mask], y[mask], '-', alpha=0.85, lw=1.5,
                               label=f"Power = {pw} W")
            c = line.get_color()
            plt.errorbar(x[mask], y[mask], yerr=yerr[mask],
                         fmt='o', capsize=4, elinewidth=1,
                         color=c, ecolor=c, markeredgecolor=c, markerfacecolor=c)
    else:
        raise ValueError("p_name must be 'Pressure' or 'Power' (meaning the FIXED parameter)")

    plt.xlabel(xlabel)
    if plot_name == "Te":
        plt.ylabel("Electron Temperature (eV)")
        title_core = "Electron Temperature"
    else:
        plt.ylabel("Electron Density (m$^{-3}$)")
        title_core = "Electron Density"

    title = f"{title_core} vs {'Power' if p_name=='Pressure' else 'Pressure'}"
    if location is not None:
        title += f" at {location} inches"
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


