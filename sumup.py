import pandas as pd
import density_core

# rootfolder= '/users/ag/hugedata/sumup/density'
rootfolder = "sumup"
df = pd.read_csv(f"{rootfolder}/sumup_densities_cleaned.zip")


def get_core(coreid=381, site_name=None):
    if not site_name:
        site_name = f"SUMup{coreid}"
    data = df[df.id == coreid]
    core = density_core.DensityCore(site_name=site_name, lat=data.lat.iloc[0], lon=data.lon.iloc[0])
    core.set_density_profile(data.midpoint_depth, data.density, is_smooth=False)
    return core
