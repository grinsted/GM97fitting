import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
import dataclasses
import maplookup
import pickle
from pygam import LinearGAM, s  # , te
from dataclasses import fields, asdict

#
# This file has a class to hold a density profile along with
# additional metadata and derived data such as overburden.
#
#
# Aslak Grinsted 2023
#

rhoi = 921  # used to calculate rho_hat in weird xscale function
g = 9.82


# --------------------------------------------------------------


# TODO: consider using a regular class instead of dataclass.
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

    def __post_init__(self):
        if np.isnan(self.e1 + self.e2):
            self.e1, self.e2 = maplookup.get_strainrate(self.lat, self.lon, source="dtu", smoothing_sigma=150, return_eigen_strainrate=True)
        if np.isnan(self.bdot):
            self.bdot = maplookup.get_accumulation(self.lat, self.lon) * 1000
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

    def set_density_profile(self, z, rho, is_smooth=False, dz_smoothing=0.5, rhos=None):
        six = np.argsort(np.asarray(z))
        if is_smooth:
            self._z = np.asarray(z)[six]
            self._rho = np.asarray(rho)[six]
        else:
            # make it smooth but keep values in raw
            self._raw_z = np.asarray(z)[six]
            self._raw_rho = np.asarray(rho)[six]
            self._z = np.arange(0.0, np.max(z), dz_smoothing)  # IT has to go to the surface!
            self._rho = concave_fit(self._raw_z, self._raw_rho, self._z, rhos=rhos)
        self._drho_dz = np.gradient(self._rho, self._z)
        self._overburden = cumtrapz(self._rho, self._z, initial=0)

    def plot(self, show_raw=False, **kwargs):
        h = plt.plot(self._rho, self._z, label=self.site_name, zorder=2, **kwargs)
        if show_raw:
            plt.plot(self._raw_rho, self._raw_z, ".", color=h[0].get_color(), **kwargs)
        plt.gca().set_ylim(np.sort(plt.gca().get_ylim())[::-1])

    # -------------------------
    # Serialization helpers
    # -------------------------

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.__dict__ = d
        return obj

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)


# this is a helper function to make non-linear transformation of the x-axis of the current plot axis
def density_xscale():
    symlog = lambda x, thres: np.sign(x) * (np.log(1 + np.abs(x) / thres))
    symexp = lambda y, thres: np.sign(y) * thres * (np.exp(y) - 1)
    forward_transform = lambda rho: -symlog(rhoi - rho, 5.0)
    inverse_transform = lambda x: rhoi - symexp(-x, 5.0)
    plt.xscale("function", functions=(forward_transform, inverse_transform))


# this function is used to make a smooth density profile from noisy data.
def concave_fit(
    zp,
    rhop,
    z,
    rhos=None,
    n_splines=20,
    contraints=["monotonic_inc", "concave"],
    lam=10,
):
    X = zp.ravel()
    y = rhop.ravel()
    weights = np.ones_like(y)
    if rhos:
        X = np.append(0, X)
        y = np.append(rhos, y)
        weights = np.append(100.0, weights)  # should this be a parameter
    mygam = LinearGAM(s(0, n_splines=n_splines, constraints=contraints), lam=lam).fit(X, y, weights=weights)
    # z = np.arange(np.min(zp),np.max(zp),dz)
    rho = mygam.predict(z)
    rho[z > np.max(zp)] = np.nan  # do not extrapolate!
    return rho
