"""Construction of the island testbeds from the public RTS-GMLC benchmark.

The RTS-GMLC repository publishes, for a full calendar year, five-minute
real-time chronologies and the matching hourly day-ahead forecasts for
photovoltaic sites, wind sites and regional demand.  The real-time series
are treated as the realised process and the day-ahead series as the
information a causal controller actually possesses, so forecast error is
observed rather than assumed.

Three island testbeds are assembled by pairing distinct real resource
sites with distinct real demand regions and rescaling demand to island
scale.  No series is synthesised.
"""
import io
import os
import urllib.request
import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/GridMod/RTS-GMLC/master/RTS_Data"
FILES = {
    "rt_load": "timeseries_data_files/Load/REAL_TIME_regional_Load.csv",
    "da_load": "timeseries_data_files/Load/DAY_AHEAD_regional_Load.csv",
    "rt_pv":   "timeseries_data_files/PV/REAL_TIME_pv.csv",
    "da_pv":   "timeseries_data_files/PV/DAY_AHEAD_pv.csv",
    "rt_wind": "timeseries_data_files/WIND/REAL_TIME_wind.csv",
    "da_wind": "timeseries_data_files/WIND/DAY_AHEAD_wind.csv",
    "gen":     "SourceData/gen.csv",
}
DATA_DIR = os.environ.get("ISLAND_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def _fetch(key: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    local = os.path.join(DATA_DIR, os.path.basename(FILES[key]))
    if not os.path.exists(local):
        with urllib.request.urlopen(f"{RAW}/{FILES[key]}", timeout=180) as r:
            raw = r.read()
        with open(local, "wb") as f:
            f.write(raw)
    return pd.read_csv(local)


def _to_hourly(df: pd.DataFrame, cols) -> np.ndarray:
    """Average sub-hourly periods to hourly and drop 29 February."""
    d = df.copy()
    n_per_day = int(d.groupby(["Month", "Day"]).size().max())
    per_hour = n_per_day // 24
    d["hour"] = (d["Period"] - 1) // per_hour
    grp = d.groupby(["Month", "Day", "hour"], sort=False)[list(cols)].mean().reset_index()
    grp = grp[~((grp["Month"] == 2) & (grp["Day"] == 29))]
    return grp[list(cols)].to_numpy(dtype=float)


def build_island(spec, cache=True):
    """Return a dict of hourly arrays for one island testbed."""
    cache_file = os.path.join(DATA_DIR, f"island_{spec.key}.npz")
    if cache and os.path.exists(cache_file):
        z = np.load(cache_file)
        return {k: z[k] for k in z.files}

    gen = _fetch("gen").set_index("GEN UID")["PMax MW"]
    out = {}
    for tag, rt_key, da_key in (("pv", "rt_pv", "da_pv"), ("wind", "rt_wind", "da_wind")):
        sites = spec.pv_sites if tag == "pv" else (spec.wind_site,)
        rt = _fetch(rt_key)
        da = _fetch(da_key)
        sites = [s for s in sites if s in rt.columns and s in da.columns]
        if not sites:
            raise KeyError(f"no {tag} sites available for island {spec.key}")
        cap = np.array([gen[s] for s in sites], dtype=float)
        a = _to_hourly(rt, sites) / cap
        f = _to_hourly(da, sites) / cap
        out[f"{tag}_cf"] = np.clip(a.mean(axis=1), 0.0, 1.0)
        out[f"{tag}_cf_fc"] = np.clip(f.mean(axis=1), 0.0, 1.0)

    rl = _to_hourly(_fetch("rt_load"), [spec.load_region])[:, 0]
    dl = _to_hourly(_fetch("da_load"), [spec.load_region])[:, 0]
    scale = spec.peak_mw / rl.max()
    out["load"] = rl * scale
    out["load_fc"] = dl * scale
    out["cool"] = out["load"] * spec.cool_share
    out["hour_of_year"] = np.arange(out["load"].size, dtype=float)
    if cache:
        np.savez_compressed(cache_file, **out)
    return out


def summarise(spec, d):
    cf_pv = d["pv_cf"].mean()
    cf_w = d["wind_cf"].mean()
    lf = d["load"].mean() / d["load"].max()
    err_pv = np.sqrt(np.mean((d["pv_cf"] - d["pv_cf_fc"]) ** 2))
    err_w = np.sqrt(np.mean((d["wind_cf"] - d["wind_cf_fc"]) ** 2))
    err_l = np.sqrt(np.mean((d["load"] - d["load_fc"]) ** 2)) / d["load"].mean()
    return dict(island=spec.key, hours=d["load"].size, peak=d["load"].max(),
                mean_load=d["load"].mean(), energy=d["load"].sum() / 1e3,
                cf_pv=cf_pv, cf_wind=cf_w, load_factor=lf,
                rmse_pv=err_pv, rmse_wind=err_w, nrmse_load=err_l)


if __name__ == "__main__":
    from config import ISLANDS
    rows = []
    for s in ISLANDS:
        d = build_island(s)
        rows.append(summarise(s, d))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
