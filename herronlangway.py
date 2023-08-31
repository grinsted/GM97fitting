import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import density_core

rhow = 1000
rhoi = 917
R = 8.31446261815324  # J/mol/K
g = 9.82


def ArthernHL(T, rho, bdot):
    # herron langway as given in Arthern et al. 2010
    # units: celcius, kg/m3, kg/m2
    T = T + 273.15
    c0 = 11 * (bdot / rhow) * np.exp(-10160 / (R * T))
    c1 = 575 * np.sqrt(bdot / rhow) * np.exp(-21400 / (R * T))
    c = np.where(rho < 550, c0, c1)
    drho_dt = c * (rhoi - rho)
    return drho_dt


def HL_density_profile(T=-31.7, rhos=300, bdot=0.13 * rhow):
    # This makes a herron langway profile and returns it as a DensityCore

    drho = 10
    rho = np.arange(rhos, 900, drho)
    w = bdot / rho
    drho_dt = ArthernHL(T, rho, bdot)
    #    ezz = -drho_dt/rho

    drho_dz = drho_dt / w

    dz = drho / drho_dz
    z = cumtrapz(dz, initial=0)

    # return z, rho
    core = density_core.DensityCore(site_name="HL", e1=0, e2=0, T=T, bdot=bdot)
    core.set_density_profile(z, rho, is_smooth=True)
    return core
