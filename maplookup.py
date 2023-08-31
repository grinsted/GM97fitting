import numpy as np
import rioxarray as rio
import xarray as xr
import pyproj
from functools import lru_cache

#IV_rootfolder = "/Users/ag/HugeData/Greenland IV"
IV_rootfolder = "https://sid.erda.dk/share_redirect/D67kSJ8fSX/IV"
accumulation_file = "https://sid.erda.dk/share_redirect/D67kSJ8fSX/QGreenland_v1.0.1/Regional%20climate%20models/RACMO%20model%20output/Total%20precipitation%201958-2019%20%281km%29/racmo_precip.tif"
temperature_file = "https://sid.erda.dk/share_redirect/D67kSJ8fSX/CARRA/carra_annual_T.tif"

# this is a slow function intended for calculating the strain rate at single points
@lru_cache(maxsize=None)
def get_strainrate(lat, lon, source="measures", return_eigen_strainrate=False):
    source = source.lower()
    if "measures" in source:
        #https://sid.erda.dk/share_redirect/D67kSJ8fSX/IV/MEaSUREs/M%20Multi-year%20IV%20mosaic%20v1/greenland_vel_mosaic250_vx_v1.tif
        fvx = f"{IV_rootfolder}/MEaSUREs/M%20Multi-year%20IV%20mosaic%20v1/greenland_vel_mosaic250_vx_v1.tif"
    elif "dtu" in source:
        fvx = f"{IV_rootfolder}/DTU-Space/greenland_iv_50m_s1_20191216_20200125_zwally21_winter_v1.0_vx.tif"
    elif ("itslive" in source) or ("its-live" in source):
        fvx = f"{IV_rootfolder}/ITS_LIVE/GRE_G0120_0000_vx.tif"
    else:
        fvx = f"{IV_rootfolder}/MEaSUREs/M%20Multi-year%20IV%20mosaic%20v1/greenland_vel_mosaic250_vx_v1.tif"

    fvy = fvx.replace("_vx", "_vy")

    mapproj = pyproj.Transformer.from_crs("epsg:4326", "epsg:3413")
    x0, y0 = mapproj.transform(lat, lon)

    w = 1000  # only work on a small window.
    vx = rio.open_rasterio(fvx, band_as_variable=True).band_1
    vx = vx.rio.clip_box(minx=x0 - w, miny=y0 - w, maxx=x0 + w, maxy=y0 + w)
    vy = rio.open_rasterio(fvy, band_as_variable=True).band_1
    vy = vy.rio.clip_box(minx=x0 - w, miny=y0 - w, maxx=x0 + w, maxy=y0 + w)

    if "dtu" in source:
        vx = vx * 365.25
        vy = vy * 365.25

    dvxdx = vx.differentiate("x")
    dvxdy = vx.differentiate("y")
    dvydx = vy.differentiate("x")
    dvydy = vy.differentiate("y")
    epsilon = np.zeros((2, 2))
    epsilon[0, 0] = dvxdx.interp(x=x0, y=y0).values
    epsilon[1, 1] = dvydy.interp(x=x0, y=y0).values
    Lxy = dvydx.interp(x=x0, y=y0).values
    Lyx = dvxdy.interp(x=x0, y=y0).values
    epsilon[0, 1] = (Lxy + Lyx) / 2
    epsilon[1, 0] = (Lxy + Lyx) / 2
    if return_eigen_strainrate:
        return np.linalg.eig(epsilon)[0]
    else:
        return epsilon


@lru_cache(maxsize=None)
def get_accumulation(lat, lon):

    mapproj = pyproj.Transformer.from_crs("epsg:4326", "epsg:3413")
    x0, y0 = mapproj.transform(lat, lon)
    acc_racmo = rio.open_rasterio(accumulation_file, band_as_variable=True).band_1
    return acc_racmo.interp(x=x0, y=y0).values/1000


@lru_cache(maxsize=None)
def get_temperature(lat, lon):
    mapproj = pyproj.Transformer.from_crs("epsg:4326", "epsg:3413")
    x0, y0 = mapproj.transform(lat, lon)
    Tfile = rio.open_rasterio(temperature_file, band_as_variable=True).band_1
    return Tfile.interp(x=x0, y=y0).values-273.15
