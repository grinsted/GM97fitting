import pandas as pd
import density_core

# rootfolder= '/users/ag/hugedata/sumup/density'
rootfolder = "sumup"
df = pd.read_csv(f"{rootfolder}/sumup_densities_cleaned.zip")

known_citation = {
    12: "NEEM",
    26: "ngt03C93.2",
    27: "ngt06C93.2",
    28: "ngt14C93.2",
    29: "ngt27C94.2",
    30: "ngt37C95.2",
    31: "ngt42C95.2",
    42: "GISP2",
}


def get_core(coreid=381, site_name=None):
    data = df[df.id == coreid]
    if not site_name:
        if data.iloc[0].citation in known_citation:
            site_name = known_citation[data.iloc[0].citation]
        else:
            site_name = f"SUMup{coreid}"
    core = density_core.DensityCore(
        site_name=site_name, lat=data.lat.iloc[0], lon=data.lon.iloc[0]
    )
    core.set_density_profile(data.midpoint_depth, data.density, is_smooth=False)
    return core
