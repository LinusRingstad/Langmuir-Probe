### Basic Langmuir probe data analysis functions

import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class lp_Settings:
    pi = np.pi
    k = 1.380649e-23  # J/K
    k_eV = 8.617333262145e-5  # eV/K
    m_e = 9.1093837015e-31  # kg
    e = 1.602176634e-19  # C
    u = 1.66e-27  # kg, one atomic unit
    e0 = 8.854187817e-12  # F/m
    

    
    
    def __init__(self):
        self.rounding = 1
        self.T_eV = 300 *self.k  # room temperature eV
        self.A = 40
        self.m_i = self.A * self.u  # mass argon in kg
        self.skiplines_csv = 61 #junk lines in output file
        self.skipfooter_csv = 5 

        self.press_int = int(75.1)
        self.press_dec = int(round(75.1 % 1 *10,self.rounding))
        self.pow_default = 1000 #W
        self.seqno = 1
        self.length_str = '2in_'
        self.prober = (.254)/2*0.001 #probe radius m
        self.probel = 4e-3 #probe length in m
        self.Aprob = 2*np.pi*self.prober*self.probel #probe area

        self.fname_fmt = f'Trace_{self.press_int}p{self.press_dec}mTorr_{self.pow_default}W_{self.length_str}{self.seqno}.csv'

    
    def set_T_eV(self,newT):
        """Convert temperature from K to eV"""
        self.T = newT * self.k_eV

    def set_probe_r(self,radius_m):
        """Set probe radius in meters"""
        self.prober = radius_m

    def set_probe_l(self,length_m):
        """Set probe length in meters"""
        self.probel = length_m

    
    def set_m_i(self,mu):
        """Convert atomic mass unit to kg"""
        self.A = mu
        self.m_i =  mu * self.u

    def set_skipheader(self,lines_header):
        """Set number of junk lines in output csv file"""
        self.skiplines_csv = lines_header

    def set_skipfooter(self,lines_footer):
        """Set number of junk lines in bottom of output csv file"""
        self.skipfooter_csv = lines_footer


    def set_params(self,pressure,power,seqno,length=0,set_round=1):
        """Set all parameters at once"""
        self.rounding = set_round
        self.press_int = int(pressure)
        self.press_dec = int(round(pressure % 1 *10,self.rounding))
        self.pow_default = power
        if length != 0:
            self.length_str = f'{length}'+'in_'
        else:
            self.length_str = ''
        self.seqno = seqno
  
        self.fname_fmt = f'Trace_{self.press_int}p{self.press_dec}mTorr_{self.pow_default}W_{self.length_str}{self.seqno}.csv'


def read_file(settings,degree = 9):
    """Read Langmuir probe data from CSV file"""
    data = pd.read_csv(settings.fname_fmt, skiprows=settings.skiplines_csv, skipfooter=settings.skipfooter_csv, engine='python').to_numpy()
    T = data[:, 1] 
    V = data[:, 2]  
    I = data[:, 3]
    coeffs = np.polyfit(V, I, degree)
    return V, I,coeffs


def atanh_fit(V,I):
    """
    Fit I-V data to a hyperbolic tangent function.
    """

    def tanh_func(V, a, b, c, d):
        return a * np.tanh(b * (V - c)) + d

    initial_guess = [max(I), 1, np.median(V), min(I)]
    params, _ = curve_fit(tanh_func, V, I, p0=initial_guess)
    return params



def floating_potential(V,coeffs):
    """
    Find floating potential (V_f) where I = 0
    """
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    real_roots = [r for r in real_roots if min(V) <= r <= max(V)]
    return min(real_roots,key=abs)



def plasma_potential(V, coeffs):
    """
    Estimate the plasma potential (Vp) from a fitted I(V) polynomial model.
    """
    V = np.asarray(V, dtype=float)
    v_min, v_max = float(np.min(V)), float(np.max(V))


    # Construct model and its derivatives
    p = np.poly1d(coeffs)
    d1 = np.polyder(p, 1)
    d2 = np.polyder(p, 2)

    # Find real roots of p''
    roots = np.roots(d2)
    real_roots = roots.real

    real_roots = [r for r in real_roots if v_min <= r <= v_max]

    real_roots = np.unique(np.concatenate([real_roots, [v_min, v_max]]))

    # Evaluate curvature (second derivative)
    d1_vals = d1(real_roots)



    idx = np.argmax(d1_vals)

    V_p = float(real_roots[idx])
    I_p = float(p(V_p))

    return V_p, I_p #plasma potential and saturation current

def plot_IV(V,I,coeffs,vp,vf):
    """Plot I-V data and  polynomial"""

    plt.figure(figsize=(10, 6))
    plt.plot(V, I, 'b.', label='Measured Data')
    V_fit = np.linspace(min(V), max(V), 500)
    I_fit = np.polyval(coeffs, V_fit)
    plt.plot(V_fit, I_fit, 'r-', label='Fitted Polynomial')
    plt.axvline(x=vp, color='g', linestyle='--', label='Plasma Potential (V_p)')
    plt.axvline(x=vf, color='m', linestyle='--', label='Floating Potential (V_f)')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (I)')
    plt.title('Langmuir Probe I-V Characteristic')
    plt.legend()
    plt.grid()
    plt.show()

def guess_Te(Vp,Vf,settings):
    """Estimate electron temperature from plasma and floating potentials"""
    Te = (Vp - Vf) / (3.34+np.log(settings.A))
    return Te



def guess_ne(Ip,Te,settings):
    """Estimate electron density from ion saturation current (A), Te (eV)"""
    e = settings.e
    Aprob = settings.Aprob

    ne = 3.73e15*Ip/Aprob/np.sqrt(Te)
    return ne


def guess_ratio(Te,ne,settings):
    """Estimate Debye length from Te (eV) and ne (m^-3)"""
    r = settings.prober
    debye = 7430*np.sqrt(Te/ne)
    rat = r/debye
    return rat

def interp_laframboise(V,I,Vp,Vf,Te,ne,settings):
    """Calculate laframboise paramteter ji* and find ne and Te given:
        ratio of plasma radius to debye length
        voltage array
        plasma potential
        electron temperature in eV
        electron density
        settings class
        """
    e = settings.e
    k = settings.k
    k_eV = settings.k_eV
    e0 = settings.e0

    Te_K = Te / k_eV #convert to kelvin
    

    rat = guess_ratio(Te,ne,settings)

    xp = e*(Vp-V[200]) / (k*Te_K)


    data_j_i_star = pd.read_csv('lafram.csv')
    
    Xp_values = data_j_i_star["Xp"].values.astype(float)
    ratio_values = data_j_i_star.columns[1:].astype(float)

    # Extract the j_i_star table
    j_i_star_table = data_j_i_star.iloc[:, 1:].values.astype(float)

    # Interpolate along the Xp dimension (rows)
    if xp < Xp_values.min():
        xp = Xp_values.min()
    elif xp > Xp_values.max():
        xp = Xp_values.max()
    j_i_star_row = np.array([
        np.interp(xp, Xp_values, j_i_star_table[:, col_idx])
        for col_idx in range(j_i_star_table.shape[1])
    ])

    # Interpolate along the ratio dimension (columns)
    if rat < ratio_values.min():
        rat = ratio_values.min()
    elif rat > ratio_values.max():
        rat = ratio_values.max()
    j_i_star = np.interp(rat, ratio_values, j_i_star_row)


    
    Ii = 0.25*e*ne*(np.sqrt(8*k*Te_K/(np.pi*settings.m_i)))*settings.Aprob*j_i_star #ion current
    
    Ie = I - Ii #electron current

    mask = np.where((Ie>0) & (V<Vp) & (V>Vf))[0]

    V_e = V[mask]-Vp
    I_e = Ie[mask]


    lin_fit = np.polyfit(V_e, np.log(I_e), 1)
    m = lin_fit[0]
    Te_new_K = e/m/k #new electron temperature in Kelvin


    ne_new = np.sqrt(settings.pi*settings.m_i/Te_new_K/settings.k/8)*4*Ii/settings.Aprob/j_i_star/settings.e #new electron density in m^-3
   
    
    return Te_new_K*settings.k_eV,ne_new,Ie