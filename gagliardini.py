import numpy as np

# import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.optimize import root_scalar, least_squares

# import density_core


sec_per_year = 365.25 * 24 * 60 * 60
rhow = 1000
rhoi = 917
R = 8.31446261815324  # J/mol/K
g = 9.82


def A(T):
    T = T + 273.15
    A1 = 3.985e-13 * np.exp(-60e3 / (R * T))
    A2 = 1.916e3 * np.exp(-139e3 / (R * T))
    return np.maximum(A1, A2)


def r_fun(a, b):
    return 2 / (3 * a) + 3 / (2 * b)


def gagliardini_ezz(sigma_zz, a, b, T, e1=0, e2=0):
    # assuming e1=0 and e2=0
    r = r_fun(a, b)
    p = (e1 + e2) / (3 * a) - 3 * (e1 + e2) / (2 * b)
    Asig3 = A(T) * sigma_zz**3
    k0 = Asig3 * ((e1**2 + e2**2 - e1 * e2) / (3 * a) + 3 * (e1**2 + e2**2 + 2 * e1 * e2) / (4 * b)) + p**3
    k1 = -p * Asig3 - 3 * p**2 * r
    k2 = 0.5 * r * Asig3 + 3 * p * r**2
    k3 = -(r**3)
    rts = np.roots([k3, k2, k1, k0])
    rts = rts[np.isreal(rts)]  # & (np.sign(rts) == np.sign(sigma_zz))]
    if len(rts) == 0:
        return np.inf
    return np.real(rts[0])  # TODO: figure out a better way to pick the correct root.


def a_fun(rho):
    # zwinger 2007
    r = rho / rhoi
    a1 = np.exp(13.22240 - 15.78652 * r)
    a2 = (1 + (2 / 3) * (1 - r)) * (r ** (-1.5))  # assuming n=3
    return np.where(r < 0.81, a1, a2)


def b_fun(rho):
    r = rho / rhoi
    b1 = np.exp(15.09371 - 20.46489 * r)
    b2 = 0.75 * (((1 - r) ** (1 / 3)) / (3 * (1 - (1 - r) ** (1 / 3)))) ** (1.5)  # assuming n=3
    return np.where(r < 0.81, b1, b2)


# TODO: use density_core - return r instead of a and b
# def fit_density_profile(z, rho, drho_dz=None, T=-31, bdot=0.11 * 917, e1=0, e2=0):
#    #
#    if drho_dz is None:  # this is to facilitate that you can pass a smoothed calculation of the gradient.
#        drho_dz = np.gradient(rho, z)
#    overburden_load = cumtrapz(rho, z, initial=0)
#    sigma_zz = -overburden_load * g
#    w = (bdot - overburden_load * (e1 + e2)) / rho
#
#    # w = bdot / rho
#    # asssumed:
#    ezz = -w * drho_dz / rho - e1 - e2  # TODO: check
#
#    a = np.full_like(z, np.nan)
#    b = np.full_like(z, np.nan)
#    for ix in range(len(a)):
#        this_rho = rho[ix]
#        this_ezz = ezz[ix] / sec_per_year
#        this_sigma_zz = sigma_zz[ix]
#        if this_sigma_zz > -10:
#            continue
#
#        baratio = b_fun(this_rho) / a_fun(this_rho)
#        try:
#            sol = root_scalar(
#                lambda a: gagliardini_ezz(this_sigma_zz, a, a * baratio, T, e1 / sec_per_year, e2 / sec_per_year) - this_ezz,
#                x0=a_fun(this_rho),
#                bracket=[1e-6, 1e6],
#            )
#            if sol.converged:
#                a[ix] = sol.root
#                b[ix] = sol.root * baratio
#        except ValueError:
#            pass
#    return a, b


def singlecore_fit(core):
    sigma_zz = -core.overburden * g
    w = (core.bdot - core.overburden * (core.e1 + core.e2)) / core.rho
    ezz = -w * core.drho_dz / core.rho - core.e1 - core.e2

    a = np.full_like(core.z, np.nan)
    b = np.full_like(core.z, np.nan)
    for ix in range(len(a)):
        this_rho = core.rho[ix]
        this_ezz = ezz[ix] / sec_per_year
        this_sigma_zz = sigma_zz[ix]
        if this_sigma_zz > -10:
            continue

        baratio = b_fun(this_rho) / a_fun(this_rho)
        try:
            sol = root_scalar(
                lambda a: gagliardini_ezz(this_sigma_zz, a, a * baratio, core.T, core.e1 / sec_per_year, core.e2 / sec_per_year) - this_ezz,
                x0=a_fun(this_rho),
                bracket=[1e-6, 1e6],
            )
            if sol.converged:
                a[ix] = sol.root
                b[ix] = sol.root * baratio
        except ValueError:
            pass
    return r_fun(a, b)


def multicore_fit(cores, rho=np.arange(350, 890, 10.0)):
    # Takes a list of DensityCore's and tries makes the best fit a and b.
    # Assuming steady state.

    a = np.full_like(rho, np.nan)
    b = np.full_like(rho, np.nan)

    gagli_vec = np.vectorize(gagliardini_ezz)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # this is used to enforce limits on parameter search.

    N_cores = len(cores)

    for ix in range(len(rho)):
        this_rho = rho[ix]
        # -------------------- Prepare data for fitting.
        e1 = np.full(N_cores, np.nan)
        e2 = np.full(N_cores, np.nan)
        sigma_zz = np.full(N_cores, np.nan)
        T = np.full(N_cores, np.nan)
        e_zz = np.full(N_cores, np.nan)
        for cix, core in enumerate(cores):
            e1[cix] = core.e1 / sec_per_year
            e2[cix] = core.e2 / sec_per_year
            T[cix] = core.T
            overburden = np.interp(this_rho, core.rho, core.overburden)
            sigma_zz[cix] = -g * overburden
            drho_dz = np.interp(this_rho, core.rho, core.drho_dz)
            w = (core.bdot - overburden * (core.e1 + core.e2)) / this_rho
            e_zz[cix] = -w * drho_dz / this_rho - core.e1 - core.e2
            z = np.interp(this_rho, core.rho, core.z)

        if np.min(z) < 1:  # dont attempt using the model without any load.
            # near surface has no load and are affected by seasonal temperatures.
            continue

        if np.any(np.isnan(overburden)):
            # TODO: dont skip if we have more than 2 cores.
            continue

        # transform the parameter space so that it is impossible to exceed theoretical limits.
        x2a = lambda x: np.exp(x[0]) + 1  # a>=1
        x2b = lambda x: x2a(x) * sigmoid(x[1]) * 9 / 2  # b<=9a/2
        deviance = lambda x: gagli_vec(sigma_zz, x2a(x), x2b(x), T, e1=e1, e2=e2) * sec_per_year - e_zz
        res = least_squares(deviance, x0=[1, 1], method="lm")
        if res.success:
            x = res.x
            a[ix], b[ix] = x2a(x), x2b(x)
    return rho, a, b
