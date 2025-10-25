import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import savgol_filter
from constants import e, k, m_i, mu, A_probe, r_p, l_p, k_eV, K_to_eV, m_e

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

def fit_data_to_polynomial(path, degree=11):
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

    coeffs = np.polyfit(V, I, deg=degree)
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
    cand = cand[cand > 0]
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
        
        
        if i == 0:
            #print("V_plasma is ", V_plasma)
            #print("V_floating is ", V_floating)
            T_e = e*(abs(V_plasma) - V_floating)/(k*(3.34 + 0.5*np.log(mu)))
            #print("T_E at iteration 0 is in K ", T_e)
            T_e_eV = T_e*K_to_eV #in eV
            #print("T_E at iteration 0 is in eV ", T_e_eV)
            n_e = 1.05e15*np.sqrt(mu/T_e_eV)*I_e[200]/(A_probe)
        else:
            #print("n_e at iteration ", i, " is ", n_e)
            #print("I_e at iteration ", i, " is ", I_e[200])
            T_e = find_Te_fit(I_e, V, V_plasma)
            T_e_eV = T_e*K_to_eV #in eV
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
    positions = [0,2,4,6]
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


# -------------------------------- EEDF CODE --------------------------------
# ---------- distributions ----------
def maxwellian_pdf_per_particle(E_eV: np.ndarray, Te_eV: float) -> np.ndarray:
    """
    Per-particle Maxwellian energy PDF (integrates to 1) in units 1/eV.
    """
    E = np.clip(np.asarray(E_eV, float), 0, None)
    Te = float(Te_eV)
    return (2/np.sqrt(np.pi)) * (np.sqrt(E) / (Te**1.5)) * np.exp(-E/Te)


def eedf_data_per_particle(
    VminusVp: np.ndarray,
    I_e: np.ndarray,
    ne_m3: float,
    A_probe_m2: float,
    smooth_span_V: float = 5.0,
    polyorder: int = 3
):
    """
    EEDF with the exact SI prefactor, then convert to per-particle 1/eV.

    Inputs
    ------
    VminusVp : array of (V - Vp) [V]
    I_e      : electron current [A]
    ne_m3    : electron density [m^-3]
    A_probe_m2 : probe area [m^2]
    smooth_span_V : Savitzky-Golay span in Volts for derivatives
    polyorder : Savitzky-Golay polynomial order

    Returns
    -------
    E_eV_sorted : energy axis (eV), ascending
    p_eV_sorted : per-particle EEDF (1/eV)
    f_eV_sorted : absolute EEDF (m^-3 / eV)
    d2I_sorted  : second derivative (A/V^2) on retarding region
    """
    VminusVp = np.asarray(VminusVp, float)
    I_e = np.asarray(I_e, float)

    # Retarding region (V <= Vp) -> V - Vp <= 0  and E = Vp - V >= 0
    m = VminusVp <= 0
    x = VminusVp[m]
    I = I_e[m]

    # Step size and SG window (odd length)
    dV = float(np.median(np.diff(x)))
    
    win = max(7, int(round(smooth_span_V / max(abs(dV), 1e-12))) | 1)

    # Stable second derivative (A / V^2)
    print(win)
    print(len(I))
    d2I = savgol_filter(I, window_length=win, polyorder=polyorder,
                        deriv=2, delta=dV, mode='interp')

    # Energy in eV and J
    E_eV = -(x)                              # E = Vp - V  (eV numerically)
    E_J  = e * np.clip(E_eV, 0, None)        # Joules

    # Druyvesteyn (SI): f_J [m^-3 J^-1]
    pref_SI = -4.0 / (A_probe_m2 * e**2) * np.sqrt(m_e / (2.0 * e))
    f_J = pref_SI * np.sqrt(E_J) * d2I

    # Convert to m^-3/eV and then per-particle 1/eV
    f_eV = f_J / e           # m^-3 / eV
    p_eV = f_eV / ne_m3      # 1 / eV

    # Sort for plotting/integration
    order = np.argsort(E_eV)
    return E_eV[order], p_eV[order], f_eV[order], d2I[order]

def calibrate_eedf_to_density(E_eV: np.ndarray, f_eV: np.ndarray, ne_m3: float):
    """
    Enforce integral of f_eV dE = n_e by a single scale factor (useful if A_probe/baseline
    makes the raw integral drift). Returns scaled f_eV and factor c.
    """
    area = float(np.trapezoid(np.clip(f_eV, 0, None), E_eV))
    c = ne_m3 / area if area != 0 else 1.0
    return f_eV * c, c


# ---------- one-file plotting ----------
def plot_eedf_vs_maxwellian(
    I_e, V, V_plasma,
    Te_K: float,
    ne_m3: float,
    A_probe_m2: float,
    smooth_span_V: float = 5.0,
    polyorder: int = 3,
    calibrate_to_ne: bool = True,
    figsize=(7,4.5)
):
    """
    Make the plot shown: per-particle EEDF (1/eV) via Druyvesteyn (SI prefactor),
    optionally calibrated to satisfy integral of f_eV dE = n_e, plus Maxwellian per-particle PDF.
    """
    VminusVp = (V-V_plasma).to_numpy()
    Ie = I_e

    # eedf (SI) -> per-particle EEDF
    E_eV, p_eV, f_eV, _ = eedf_data_per_particle(
        VminusVp, Ie, ne_m3, A_probe_m2, smooth_span_V, polyorder
    )

    # Optional calibration to enforce integral of f_eV dE = n_e (absolute scaling)
    if calibrate_to_ne:
        f_eV_cal, c = calibrate_eedf_to_density(E_eV, f_eV, ne_m3)
        p_eV = f_eV_cal / ne_m3

    # Maxwellian (per particle)
    Te_eV = Te_K * k_eV
    pM = maxwellian_pdf_per_particle(E_eV, Te_eV)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(E_eV, np.clip(p_eV, 0, None), label='EEDF per particle (1/eV)', lw=2)
    ax.plot(E_eV, pM, '--', label=f'Maxwellian (Te={Te_eV:.2f} eV)', lw=2)
    ax.set_xlabel('Energy  E = Vp - V  (eV)')
    ax.set_ylabel('Per-particle density (1/eV)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig, ax


# --- Multi-trace aggregator ---
def aggregate_three_traces(traces, files, A_probe_m2,
                           smooth_span_V=5.0, polyorder=3, Ndatapoints=923, calibrate_to_ne=True):
    """
    traces: e.g., [1,2,3]
    file : list of 3 files
    A_probe_m2: probe area in m^2
    Returns:
      dict with means/stds for Te_K, ne, I_e, V, Vp,
      and (E_common, p_mean, p_std, f_abs_mean) for the EEDF.
    """
    VminusVp_matrix= np.zeros((1001, len(traces)))
    E_matrix= []
    p_matrix= []
    f_matrix= []
    TeK_list, ne_list = [], []
    

    for i,trace in enumerate(traces):
        file = files[i]

        # --- use your EXISTING functions (unchanged) ---
        Te_K, ne, I_e = find_Te_n_e_iteration(file)             # Te in K, ne in m^-3, I_e array
        I, V, coeffs, model = fit_data_to_polynomial(file)
        Vp, Iplasma, min_d2I, info = Vplasma_from_model(model, V)

    
        TeK_list.append(float(Te_K))
        ne_list.append(float(ne))
        VminusVp = (V-Vp).to_numpy()
        VminusVp_matrix[:,i] = VminusVp
        
        # per-trace EEDF with SI prefactor
        E_eV, p_eV, f_eV, _ = eedf_data_per_particle(
            VminusVp, I_e, ne, A_probe_m2,
            smooth_span_V=smooth_span_V, polyorder=polyorder
        )
        
        #calibrate to ne
        if calibrate_to_ne:
            print("calibrating to ne")
            f_eV_cal, c = calibrate_eedf_to_density(E_eV, f_eV, ne)
            p_eV = f_eV_cal / ne

        print(f"p_eV, {i+1} trace:\n {p_eV}")
        E_matrix.append(E_eV)
        p_matrix.append(p_eV)
        f_matrix.append(f_eV)

    
    TeK_mean = float(np.mean(TeK_list))
    TeK_std  = float(np.std (TeK_list, ddof=1)) if len(TeK_list)>1 else 0.0
    ne_mean  = float(np.mean(ne_list))
    ne_std   = float(np.std (ne_list, ddof=1)) if len(ne_list)>1 else 0.0

    # --- common energy grid (use overlap of the three traces) ---
    
    Emin = max(E.min() for E in E_matrix)
    Emax = min(E.max() for E in E_matrix)
    E_common = np.linspace(Emin, Emax, Ndatapoints)

    # interpolate per-trace EEDF onto common grid and aggregate
    p_interp = np.vstack([np.interp(E_common, E, p, left=0, right=0) for E,p in zip(E_matrix, p_matrix)])
    print("Shape of p_interp: ", p_interp.shape)
    f_interp = np.vstack([np.interp(E_common, E, f, left=0.0, right=0.0) for E,f in zip(E_matrix, f_matrix)])

    p_mean = np.mean(p_interp, axis=0)
    p_mean = np.clip(p_mean, 0, None)
    
    # E_common is uniform (linspace), so this is perfect
    dE = E_common[1] - E_common[0]

    def smooth_savgol(y, span_eV=5.0, polyorder=3):
        # convert a desired energy span into an odd window length in points
        w = max(3, int(round(span_eV / dE)) | 1)           # odd int >= 3
        w = max(w, polyorder + 2 | 1)                      # ensure > polyorder
        ys = savgol_filter(y, window_length=w, polyorder=polyorder, mode="interp")
        return np.clip(ys, 0, None)                        # keep nonnegative

    p_mean_smooth = smooth_savgol(p_mean, span_eV=5.0, polyorder=3)

    p_std  = np.std(p_mean_smooth, axis=0, ddof=1) / np.sqrt(len(traces)) 


    bundle_data = {
        "E_common": E_common,
        "p_mean": p_mean_smooth,          # 1/eV
        "p_std": p_std,            # 1/eV
        "TeK_mean": TeK_mean, "TeK_std": TeK_std,
        "ne_mean": ne_mean, "ne_std": ne_std,
    }
    
    return bundle_data
    
# --- plotting helper (mean ±1σ + Maxwellian from mean Te) ---
def plot_eedf_bundle(bundle, title=None):
    E = bundle["E_common"]
    p = bundle["p_mean"]
    s = bundle["p_std"]

    Te_eV = bundle["TeK_mean"] * k_eV
    pM = maxwellian_pdf_per_particle(E, Te_eV)

    
    (line,) = plt.plot(E, p, label=f"EEDF mean (1/eV) for {title}", lw=2)
    plt.fill_between(E, np.clip(p-s,0,None), p+s, alpha=0.25, label=f"±1 sigma for {title}")
    c = line.get_color()
    plt.plot(E, pM, "--", lw=2, color= c, label=f"Maxwellian (Te={Te_eV:.2f} eV) for {title}")
    
    plt.xlabel("Energy  E = Vp - V  (eV)")
    plt.ylabel("Per-particle density (1/eV)")
    plt.grid(True); plt.legend(); plt.tight_layout()


def print_eedf_whole():
    for pos in [0,2,4,6]:
        for press in [40,100]:
            plt.figure(figsize=(7,4.5))
            
            for power in [400, 600, 800, 1000]:
                files = []
                template = f'/Users/marisolvelapatino/Desktop/Langmuir-Probe/LP/{pos}in/{press} mTorr/{power}W'
                for file in os.listdir(template):
                    final_file = template + "/" + file
                    files.append(final_file)

                if pos == 0:
                    traces = [1,2,3,4,5]
                else:
                    traces = [1,2,3]
                    
                bundle = am.aggregate_three_traces(
                    traces,
                    files=files,
                    A_probe_m2=A_probe,
                    smooth_span_V=5.0,     # ~5 V SG window (tune 2–8 V)
                    polyorder=3,
                    Ndatapoints=923
                )

                am.plot_eedf_bundle(bundle, title=f"{power} W")
        
            plt.title(f"{pos} in, {press} mTorr")
            plt.show()
