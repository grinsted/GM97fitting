import numpy as np
from scipy.optimize import root_scalar, least_squares
from scipy.special import erf
import numba

# from scipy.special import jv
# import density_core


sec_per_year = 365.25 * 24 * 60 * 60
rhow = 1000
# rhoi = 917
rhoi = 921
R = 8.31446261815324  # J/mol/K
g = 9.82


@numba.njit
def A_fun(T_kelvin):
    A1 = 3.985e-13 * np.exp(-60e3 / (R * T_kelvin))
    A2 = 1.916e3 * np.exp(-139e3 / (R * T_kelvin))
    return np.maximum(A1, A2)


@numba.njit
def thermal_conductivity(rho):
    return 2.1 * (rho / rhoi) ** 2  # from Arthern   (units= W/m/K)


@numba.njit
def r_fun(a, b):
    return 2 / (3 * a) + 3 / (2 * b)


@numba.njit
def ab_from_r(r, baratio):
    # DO NOT USE. USE poisson to b instead...
    a = (4 * baratio + 9) / (6 * baratio * r)
    b = a * baratio
    return a, b


@numba.njit
def poisson_to_b(nu, a):
    return a * (3 - 6 * nu) / (2 * (nu + 1))


@numba.njit
def poisson_from_ab(a, b):
    return (3 * a - 2 * b) / (6 * a + 2 * b)


@numba.njit
def sigmoidal_poisson(rho, rhoc=400, alpha=2):
    return 1 / (2 + 2 * ((rhoc * (rhoi - rho)) / (rho * (rhoi - rhoc))) ** alpha)


# @numba.njit
def forward_model(sigma, a, b, A, B=0, n=3):
    p = -np.trace(sigma) / 3
    tau = sigma + p * np.eye(3)
    te2 = 0.5 * np.tensordot(tau, tau)
    se2 = a * te2 + (b * p**2) / 3
    phi = A * se2 ** ((n - 1) / 2) + B
    return phi * (a * tau - 2 * b * p * np.eye(3) / 9)


@numba.jit
def inverse_model(e, a, b, A, n=3):
    ed = e - np.trace(e) * np.eye(3) / 3  # deviatoric strain rate
    e_e2 = np.tensordot(ed, ed) / (2 * a) + 3 * np.trace(e) ** 2 / (4 * b)
    c = A ** (-1 / n) * e_e2 ** ((1 - n) / (2 * n))
    sigma = c * (ed / a + 3 * np.trace(e) * np.eye(3) / (2 * b))
    return sigma


def leastsquares_ezz(sigma_zz, a, b, A, e1=0, e2=0, forward_model=forward_model):
    # TODO: make model parameters pass_through
    if b == 0:
        return -e1 - e2
    f = lambda sxy: forward_model(np.diag([sxy[0], sxy[1], sigma_zz]), a, b, A, 0)
    e0 = f([0, 0])[2, 2]  # ezz if sxx and syy = 0

    res = least_squares(
        lambda sxy: np.diag(f(sxy))[:-1] - np.array([e1, e2]),
        x0=np.array([sigma_zz * e1 / e0, sigma_zz * e2 / e0]),
        x_scale=1e3,  # np.max(np.abs(sigma_zz), 1e3),
        f_scale=np.abs(e0),
        tr_solver="exact",
        method="lm",
    )
    e = f(res.x)
    return e[2, 2]


@numba.njit(inline="always", fastmath=True)
def closed_form_cubic_root(k3, k2, k1, k0):
    # --- Degenerate cases ---
    if abs(k3) < 1e-50:
        if abs(k2) < 1e-50:
            if abs(k1) < 1e-50:
                return 0.0
            return -k0 / k1  # linear

        # quadratic
        disc = k1 * k1 - 4.0 * k2 * k0
        if disc < 0.0:
            disc = 0.0
        return (-k1 + np.sqrt(disc)) / (2.0 * k2)

    # --- Precompute inverses ---
    inv_k3 = 1.0 / k3
    inv_k3_2 = inv_k3 * inv_k3
    inv_k3_3 = inv_k3_2 * inv_k3

    # --- Powers (faster than **) ---
    k2_sq = k2 * k2
    k2_cu = k2_sq * k2

    # --- Depressed cubic coefficients ---
    p = (3.0 * k3 * k1 - k2_sq) * (inv_k3_2 / 3.0)
    q = (2.0 * k2_cu - 9.0 * k3 * k2 * k1 + 27.0 * k3 * k3 * k0) * (inv_k3_3 / 27.0)

    # --- Discriminant ---
    p3 = p * p * p
    delta = 0.25 * q * q + p3 / 27.0

    # --- Tolerance (scaled, cheap) ---
    tmp = k2 * inv_k3
    tol = 1e-10 * tmp * tmp

    # clamp instead of branching-heavy logic
    if delta < -tol:
        # strong violation → something is wrong numerically
        # in njit we avoid assert → return something safe
        delta = 0.0
    elif delta < 0.0:
        # small negative → numerical noise
        delta = 0.0

    sqrt_delta = np.sqrt(delta)

    # --- Cardano ---
    u = np.cbrt(-0.5 * q + sqrt_delta)
    v = np.cbrt(-0.5 * q - sqrt_delta)

    return u + v - tmp / 3.0


import numpy as np


@numba.njit(inline="always", fastmath=True)
def cubic_root_ultrafast(k3, k2, k1, k0):
    # might not be stable
    a = k2 / k3
    b = k1 / k3
    c = k0 / k3

    a3 = a / 3.0
    p = b - a * a3
    q = 2.0 * a3**3 - a3 * b + c

    D = (0.5 * q) ** 2 + (p / 3.0) ** 3
    sqrt_D = np.sqrt(D)

    u = np.cbrt(-0.5 * q + sqrt_D)
    v = np.cbrt(-0.5 * q - sqrt_D)

    return (u + v) - a3


@numba.njit(inline="always", fastmath=True)
def cubic_root_closest_real_to_zero(k3, k2, k1, k0):
    # -----------------------------
    # Scale for numerical stability
    scale = max(abs(k0), abs(k1), abs(k2), abs(k3))
    if scale > 0.0:
        k3 /= scale
        k2 /= scale
        k1 /= scale
        k0 /= scale
    # -----------------------------

    # --- Degenerate cases ---
    if abs(k3) < 1e-50:
        if abs(k2) < 1e-50:
            if abs(k1) < 1e-50:
                return 0.0
            return -k0 / k1  # linear

        disc = k1 * k1 - 4.0 * k2 * k0
        if disc < 0.0:
            disc = 0.0

        r1 = (-k1 + np.sqrt(disc)) / (2.0 * k2)
        r2 = (-k1 - np.sqrt(disc)) / (2.0 * k2)

        return r1 if abs(r1) < abs(r2) else r2

    # --- Normalize cubic ---
    inv_k3 = 1.0 / k3
    b = k2 * inv_k3
    c = k1 * inv_k3
    d = k0 * inv_k3

    # depressed cubic: x = y - b/3
    bb = b * b
    p = c - bb / 3.0
    q = (2.0 * b * bb) / 27.0 - (b * c) / 3.0 + d

    half_q = 0.5 * q
    delta = half_q * half_q + (p * p * p) / 27.0

    # -----------------------------
    # CASE 1: one real root (Δ >= 0)
    if delta >= 0.0:
        sqrt_delta = np.sqrt(delta)

        # stable Cardano form
        t = -half_q - np.sign(half_q) * sqrt_delta
        u = np.cbrt(t)

        # safe computation of v
        if abs(u) < 1e-30:
            v = np.cbrt(-half_q + sqrt_delta)
        else:
            v = -p / (3.0 * u)

        y = u + v
        return y - b / 3.0

    # -----------------------------
    # CASE 2: three real roots (Δ < 0)
    r = np.sqrt(-p / 3.0)

    # safe arccos argument
    arg = -q / (2.0 * r * r * r)
    if arg < -1.0:
        arg = -1.0
    elif arg > 1.0:
        arg = 1.0

    phi = np.arccos(arg)

    # three real roots
    x1 = 2.0 * r * np.cos(phi / 3.0) - b / 3.0
    x2 = 2.0 * r * np.cos((phi + 2.0 * np.pi) / 3.0) - b / 3.0
    x3 = 2.0 * r * np.cos((phi + 4.0 * np.pi) / 3.0) - b / 3.0

    # pick closest to zero
    ax1 = abs(x1)
    ax2 = abs(x2)
    ax3 = abs(x3)

    if ax1 < ax2:
        return x1 if ax1 < ax3 else x3
    else:
        return x2 if ax2 < ax3 else x3


@numba.njit
def cubic_real_roots_eig(k3, k2, k1, k0, tol=1e-12):
    roots = np.roots([k3, k2, k1, k0])  # this uses the more robust eigenvalue of the companion matrix method
    return roots[np.abs(roots.imag) < tol].real


@numba.njit(inline="always")
def gagliardini_ezz(sigma_zz, a, b, A, e1, e2):
    # assuming e1=0 and e2=0
    # assuming n=3 (and no additional linear term in the rheology!)
    if b == 0:
        return -e1 - e2
    r = r_fun(a, b)
    p = (e1 + e2) / (3 * a) - 3 * (e1 + e2) / (2 * b)
    Asig3 = A * sigma_zz**3
    k0 = Asig3 * ((e1**2 + e2**2 - e1 * e2) / (3 * a) + 3 * (e1 + e2) ** 2 / (4 * b)) + p**3
    k1 = -p * Asig3 - 3 * p**2 * r
    k2 = 0.5 * r * Asig3 + 3 * p * r**2
    k3 = -(r**3)
    return cubic_root_ultrafast(k3, k2, k1, k0)
    # return cubic_root_closest_real_to_zero(k3, k2, k1, k0)


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64, numba.float64, numba.float64, numba.float64)], target="parallel")
def gagliardini_ezz_vec(sigma_zz, a, b, A, e1, e2):
    return gagliardini_ezz(sigma_zz, a, b, A, e1, e2)


@numba.njit(inline="always")
def gagliardini_ezz_vec2(sigma_zz, a, b, A, e1, e2):
    N = len(sigma_zz)
    ezz = np.empty(N)  # faster than zeros

    for i in range(N):
        ezz[i] = gagliardini_ezz(
            sigma_zz[i],
            a[i],
            b[i],
            A,
            e1,
            e2,
        )

    return ezz


@numba.njit
def a_fun(rho):
    # zwinger 2007
    r = rho / rhoi
    a1 = np.exp(13.22240 - 15.78652 * r)
    a2 = (1 + (2 / 3) * (1 - r)) * (r ** (-1.5))  # assuming n=3
    return np.where(r < 0.81, a1, a2)


@numba.njit
def b_fun(rho):  # note the factor 3 difference between zwinger and JL
    r = rho / rhoi
    b1 = np.exp(15.09371 - 20.46489 * r)
    b2 = 0.75 * (((1 - r) ** (1 / 3)) / (3 * (1 - (1 - r) ** (1 / 3)))) ** (1.5)  # assuming n=3
    return np.where(r < 0.81, b1, b2) / 3


# def singlecore_fit(core):
#    sigma_zz = -core.overburden * g
#    w = (core.bdot - core.overburden * (core.e1 + core.e2)) / core.rho
#    ezz = -w * core.drho_dz / core.rho - core.e1 - core.e2
#
#    a = np.full_like(core.z, np.nan)
#    b = np.full_like(core.z, np.nan)
#    for ix in range(len(core.z)):
#        this_rho = core.rho[ix]
#        this_ezz = ezz[ix] / sec_per_year
#        this_sigma_zz = sigma_zz[ix]
#        if this_sigma_zz > -10:  # we must have some load for this to work.
#            continue
#
#        baratio = b_fun(this_rho) / a_fun(this_rho)
#        try:
#            sol = root_scalar(
#                lambda a: gagliardini_ezz(
#                    this_sigma_zz,
#                    a,
#                    a * baratio,
#                    A_fun(273.15 + core.T),
#                    core.e1 / sec_per_year,
#                    core.e2 / sec_per_year,
#                )
#                - this_ezz,
#                x0=a_fun(this_rho),
#                bracket=[1e-6, 1e6],
#            )
#            if sol.converged:
#                a[ix] = sol.root
#                b[ix] = sol.root * baratio
#        except ValueError:
#            pass
#    return r_fun(a, b)


def density_profile(
    Tm,
    Ts,
    bdot,
    rho_s=350,
    z=np.linspace(0, np.sqrt(100), 100) ** 2,
    e1=0,
    e2=0,
    A_fun=A_fun,
    a_fun=a_fun,
    b_fun=b_fun,
    thermal_conductivity_fun=thermal_conductivity,
):
    #
    # e1 = 0/sec_per_year
    # e2 = 0/sec_per_year
    # Tm = 273.15-30
    # Ts = 25
    # rho_s = 350
    # bdot = 300 / sec_per_year #kg/m2/s

    omega = np.pi * 2 / sec_per_year
    # c = 2009  # J/kg/K heat capacity. Note: should be a function of T.
    c = 185 + 6.89 * Tm  # J/kg/K heat capacity - fukusako 1990 eqn2 (90-273K)

    M = np.zeros_like(z)  # overburden
    rho = np.full_like(z, np.nan)
    rho[0] = rho_s
    logTamplitude = np.full_like(z, np.nan)
    logTamplitude[0] = np.log(Ts)

    for ii in range(0, len(z) - 1):
        dz = z[ii + 1] - z[ii]
        sigma_zz = -M[ii] * g
        w = (bdot - M[ii] * (e1 + e2)) / rho[ii]
        if w <= 0:
            break
        a = a_fun(rho[ii])
        b = b_fun(rho[ii])

        k = thermal_conductivity_fun(rho[ii])

        curTs = np.exp(logTamplitude[ii])
        if curTs > 1e-3:
            dlogAdT = (np.log(A_fun(Tm + curTs)) - np.log(A_fun(Tm))) / Ts  # The warm Q is the most important
            Qprime = dlogAdT * R * (Tm + curTs / 2) ** 2
            Abar = (
                A_fun(Tm + curTs)
                * (Tm + curTs)
                * erf(np.pi * np.sqrt(0.5 * Qprime * curTs / R) / (Tm + curTs))
                / np.sqrt(2 * np.pi * Qprime * curTs / R)
            )
            # print(curTs, Abar / A_fun(Tm))
        else:
            Abar = A_fun(Tm)

        ezz = gagliardini_ezz(sigma_zz, a, b, Abar, e1=e1, e2=e2)

        drho_dz = -rho[ii] * (e1 + e2 + ezz) / w

        # UPDATE RHO

        if rho[ii] - rhoi == 0:
            rho[ii + 1] = rho[ii]
        else:
            rho[ii + 1] = rhoi - np.exp(np.log(rhoi - rho[ii]) - (drho_dz * dz) / (rhoi - rho[ii]))

        if rho[ii + 1] < 0:
            print("negative rho?")
            1 / 0

        M[ii + 1] = M[ii] + 0.5 * (rho[ii] + rho[ii + 1]) * dz  # HERES THE TRAPEZOIDAL LOAD..

        dkdz = (thermal_conductivity_fun(rho[ii + 1]) - k) / dz
        gamma = rho[ii] * c * w - dkdz
        # decay the amplitude
        if (4 * rho[ii] * c * omega) ** 2 - gamma**2 > 0:
            kz2 = (np.sqrt((4 * rho[ii] * c * omega) ** 2 - gamma**2) - gamma**4) / (8 * k**2)
            ez = (np.sqrt(4 * k**2 * kz2 + gamma**2) - gamma) / (2 * k)
        else:
            print("NOT GOOD:", logTamplitude[ii])
            ez = 0
        if np.isnan(ez):  # TODO: add a real check ....
            print("first sqrt:", (4 * rho[ii] * c * omega) ** 2 - gamma**2)
            print("second sqrt:", 4 * k**2 * kz2 + gamma**2)
            ez = 0
        logTamplitude[ii + 1] = logTamplitude[ii] - ez * dz
    return z, rho
