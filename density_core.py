import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
import dataclasses
import maplookup
from pygam import LinearGAM, s  # , te

#
# This file has a class to hold a density profile along with
# additional metadata and derived data such as overburden.
#
#
# Aslak Grinsted 2023
#


@dataclasses.dataclass()
class DensityCore:
    site_name: str = ""
    lat: float = np.nan
    lon: float = np.nan
    T: float = np.nan  # celcius
    bdot: float = np.nan  # kg/m2
    e1: float = np.nan  # per year
    e2: float = np.nan  # per year

    _z: np.ndarray = dataclasses.field(default=np.nan, repr=False, init=False)
    _rho: np.ndarray = dataclasses.field(default=np.nan, repr=False, init=False)
    _raw_z: np.ndarray = dataclasses.field(default=np.nan, repr=False, init=False)
    _raw_rho: np.ndarray = dataclasses.field(default=np.nan, repr=False, init=False)
    _drho_dz: np.ndarray = dataclasses.field(default=np.nan, repr=False, init=False)
    _overburden: np.ndarray = dataclasses.field(default=np.nan, repr=False, init=False)
    # TODO: it would make sense to autocalculate drho_dz and overburden the moment
    # rho is assigned to. But we dont need perfect OOP.
    # - So we just use it as a data container for the moment.

    def __post_init__(self):
        if np.isnan(self.e1 + self.e2):
            self.e1, self.e2 = maplookup.get_strainrate(self.lat, self.lon, return_eigen_strainrate=True)
        if np.isnan(self.bdot):
            self.bdot = maplookup.get_accumulation(self.lat, self.lon)
        if np.isnan(self.T):
            self.T = maplookup.get_temperature(self.lat, self.lon)

    @property
    def z(self) -> np.ndarray:
        return self._z

    @property
    def rho(self) -> np.ndarray:
        return self._rho

    @property
    def drho_dz(self) -> np.ndarray:
        return self._drho_dz

    @property
    def overburden(self) -> np.ndarray:
        return self._overburden

    @property
    def raw_z(self) -> np.ndarray:
        return self._raw_z

    @property
    def raw_rho(self) -> np.ndarray:
        return self._raw_rho

    def set_density_profile(self, z, rho, is_smooth=False, dz_smoothing=0.5):
        # TODO: auto smooth if not smooth?
        if is_smooth:
            self._z = z.ravel()
            self._rho = rho.ravel()
        else:
            # make it smooth but keep values in raw
            self._raw_z = z.ravel()
            self._raw_rho = rho.ravel()
            self._z = np.arange(0.0, np.max(z), dz_smoothing)  # IT has to go to the surface!
            self._rho = concave_fit(self._raw_z, self._raw_rho, self._z)
        self._drho_dz = np.gradient(self._rho, self._z)
        self._overburden = cumtrapz(self._rho, self._z, initial=0)

    def plot(self, show_raw=False, **kwargs):
        h = plt.plot(self._rho, -self._z, label=self.site_name, zorder=2, **kwargs)
        if show_raw:
            plt.plot(self._raw_rho, -self._raw_z, ".", color=h[0].get_color(), ms=0.5, alpha=0.5, **kwargs)


# --------------------------------------------------------------


rhoi = 917  # used to calculate rho_hat
g = 9.82
secperyear = 365.25 * 24 * 60 * 60


# this is a helper function to make non-linear transformation of the x-axis of the current plot axis
def density_xscale():
    symlog = lambda x, thres: np.sign(x) * (np.log(1 + np.abs(x) / thres))
    symexp = lambda y, thres: np.sign(y) * thres * (np.exp(y) - 1)
    forward_transform = lambda rho: -symlog(rhoi - rho, 5.0)
    inverse_transform = lambda x: rhoi - symexp(-x, 5.0)
    plt.xscale("function", functions=(forward_transform, inverse_transform))


# this function is used to make a smooth density profile from noisy data.
def concave_fit(zp, rhop, z, rhos=None, n_splines=20):
    X = zp.ravel()
    y = rhop.ravel()
    weights = np.ones_like(y)
    if rhos:
        X = np.append(0, X)
        y = np.append(rhos, y)
        weights = np.append(100.0, weights)  # should this be a parameter
    mygam = LinearGAM(s(0, n_splines=n_splines, constraints="concave")).fit(X, y, weights=weights)
    # z = np.arange(np.min(zp),np.max(zp),dz)
    rho = mygam.predict(z)
    rho[z > np.max(zp)] = np.nan  # do not extrapolate!
    return rho
