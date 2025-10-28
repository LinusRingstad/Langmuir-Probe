#### ---------- EEDF calculation ----------

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from basic import read_file, plasma_potential


def maxwellian(E_eV, Te_eV) -> np.ndarray:
    """
    Per-particle Maxwellian energy PDF
    """
    E = np.clip(np.asarray(E_eV, float), 0, None)
    Te = float(Te_eV)
    return (2/np.sqrt(np.pi)) * (np.sqrt(E) / (Te**1.5)) * np.exp(-E/Te)

def eedf_data(
    settings,
    I_e,
    ne_m3,
    smooth_span_V: float = 5.0,
    polyorder: int = 3,
    calibrate_to_density: bool = True
):
    """
    EEDF with the exact SI prefactor, then convert to per-particle 1/eV.
    Includes optional calibration to enforce integral of f_eV dE = n_e.

    Inputs
    ------
    VminusVp : array of (V - Vp) [V]
    I_e      : electron current [A]
    ne_m3    : electron density [m^-3]
    A_probe_m2 : probe area [m^2]
    smooth_span_V : Savitzky-Golay span in Volts for derivatives
    polyorder : Savitzky-Golay polynomial order
    calibrate_to_density : Whether to scale EEDF to match given density

    Returns
    -------
    E_eV_sorted : energy axis (eV), ascending
    p_eV_sorted : per-particle EEDF (1/eV)
    f_eV_sorted : absolute EEDF (m^-3 / eV), calibrated if requested
    d2I_sorted  : second derivative (A/V^2) on retarding region
    scale_factor : the calibration factor applied (1.0 if no calibration)
    """



    # Constants
    e = settings.e  # C
    m_e = settings.m_e  # kg
    
    A_probe_m2 = settings.Aprob


    V_0,I_0,coeffs = read_file(settings)


    Vp, Ip = plasma_potential(V_0, coeffs)

    VminusVp = V_0 - Vp
    VminusVp = np.asarray(VminusVp, float)
    I_e = np.asarray(I_e, float)

    # Retarding region (V <= Vp) -> V - Vp <= 0  and E = Vp - V >= 0
    mask = VminusVp <= 0
    V_diff = VminusVp[mask]
    I = I_e[mask]
    
    
    # Sort by voltage to ensure monotonic
    sort_idx = np.argsort(V_diff)
    V_diff= V_diff[sort_idx]
    I = I[sort_idx]
    
    # Remove duplicates
    V_diff, unique_idx = np.unique(V_diff, return_index=True)
    I = I[unique_idx]
    
    dV = float(np.median(np.diff(V_diff)))
    
    # Calculate window length for Savitzky-Golay
    win_points = max(7, int(round(smooth_span_V / max(abs(dV), 1e-12))))
    win_points = min(win_points, len(V_diff) - 1)  # Cannot exceed data length
    if win_points % 2 == 0:  # Must be odd
        win_points -= 1
    win_points = max(win_points, polyorder + 2)  # Must be > polyorder
    
    # Stable second derivative (A / V^2)


    d2I = savgol_filter(I, window_length=win_points, polyorder=polyorder,
                            deriv=2, delta=dV, mode='interp')

    # Energy in eV and J
    E_eV = -V_diff  # E = Vp - V  (eV numerically)

    E_J = e * np.clip(E_eV, 0, None)  # Joules

    # Remove points with negative energy or problematic derivatives
    valid_mask = (E_eV > 0.1) & (np.isfinite(d2I)) & (I > 0)
    if np.sum(valid_mask) < 5:
        raise ValueError("Not enough valid points for EEDF calculation")
    
    E_eV = E_eV[valid_mask]
    d2I = d2I[valid_mask]
    I = I[valid_mask]
    V_diff = V_diff[valid_mask]
    E_J = E_J[valid_mask]

    # Druyvesteyn (SI): f_J [m^-3 J^-1]

    f_J = -4.0 / (A_probe_m2 * e**2) * np.sqrt(m_e / (2.0 * e)) * np.sqrt(E_J) * d2I

    # Convert to m^-3/eV and then per-particle 1/eV
    f_eV = f_J / e           # m^-3 / eV
    
    # Remove negative values (unphysical)
    f_eV = np.clip(f_eV, 0, None)
    
    # Calibrate EEDF to match the given electron density
    scale_factor = 1.0
    if calibrate_to_density and ne_m3 > 0:
        area = float(np.trapezoid(f_eV, E_eV))
        if area > 0:
            scale_factor = ne_m3 / area
            f_eV = f_eV * scale_factor
    
    # Per-particle distribution
    if ne_m3 > 0:
        p_eV = f_eV / ne_m3      # 1 / eV
    else:
        # If no density given, normalize per-particle distribution to area = 1
        area = np.trapezoid(f_eV, E_eV)
        if area > 0:
            p_eV = f_eV / area
        else:
            p_eV = f_eV
    
    # Sort for plotting/integration
    order = np.argsort(E_eV)
    return E_eV[order], p_eV[order], f_eV[order], d2I[order], scale_factor

