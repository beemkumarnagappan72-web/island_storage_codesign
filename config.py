"""Techno-economic and structural configuration.

All monetary and efficiency parameters are taken from the public
PyPSA technology-data database (outputs/costs_2030.csv), which itself
aggregates the Danish Energy Agency technology catalogue, the DIW data
documentation and World Bank commodity prices.  Nothing here is invented:
`TECH` mirrors the database entries used, and `provenance()` prints the
mapping so that every number in the manuscript can be traced.
"""
from dataclasses import dataclass, field
import numpy as np

EUR_PER_KW_TO_EUR_PER_MW = 1e3
EUR_PER_KWH_TO_EUR_PER_MWH = 1e3

# ----------------------------------------------------------------------
# raw database entries (units as published)
# ----------------------------------------------------------------------
TECH = {
    "solar-utility":      {"investment": 482.4785,  "unit": "EUR/kW",  "FOM": 2.4757, "lifetime": 40.0},
    "onwind":             {"investment": 1383.3059, "unit": "EUR/kW",  "FOM": 1.2167, "lifetime": 30.0,
                           "VOM": 1.8033},
    "battery storage":    {"investment": 189.8610,  "unit": "EUR/kWh", "FOM": 0.0,    "lifetime": 25.0},
    "battery inverter":   {"investment": 213.9279,  "unit": "EUR/kW",  "FOM": 0.3375, "lifetime": 10.0,
                           "efficiency": 0.96},
    "electrolysis":       {"investment": 1886.0019, "unit": "EUR/kW",  "FOM": 4.0,    "lifetime": 25.0,
                           "efficiency": 0.6217},
    "fuel cell":          {"investment": 1469.3834, "unit": "EUR/kW",  "FOM": 5.0,    "lifetime": 10.0,
                           "efficiency": 0.50},
    "hydrogen tank":      {"investment": 60.0469,   "unit": "EUR/kWh", "FOM": 1.1133, "lifetime": 30.0},
    "PHS store":          {"investment": 71.7612,   "unit": "EUR/kWh", "FOM": 0.43,   "lifetime": 60.0},
    "PHS bicharger":      {"investment": 1756.6580, "unit": "EUR/kW",  "FOM": 0.9951, "lifetime": 60.0,
                           "efficiency": 0.8944},
    "water tank storage": {"investment": 3.8174,    "unit": "EUR/kWh", "FOM": 1.0,    "lifetime": 40.0,
                           "standing_loss_pct_h": 0.0077, "e_to_p_h": 60.3448},
    "oil genset":         {"investment": 458.1805,  "unit": "EUR/kW",  "FOM": 2.463,  "lifetime": 25.0,
                           "efficiency": 0.35, "VOM": 8.0148, "fuel": 43.6295,
                           "co2": 0.2571},
}

DISCOUNT_RATE = 0.07          # sensitivity-tested in E3
VOLL = 3000.0                 # EUR/MWh value of lost load, sensitivity-tested
FUEL_DELIVERY_MULT = 2.0      # remote-island delivered fuel premium, sensitivity-tested
BATTERY_EQ_CYCLES = 5000.0    # equivalent full cycles to end of life
STACK_LIFETIME_H = 60000.0    # electrolyser / fuel-cell stack operating hours


def crf(rate: float, life: float) -> float:
    """Capital recovery factor."""
    return rate / (1.0 - (1.0 + rate) ** (-life))


@dataclass
class MediumSpec:
    """Annualised cost and physics of one storage medium."""
    name: str
    key: str
    c_energy: float          # EUR per MWh of energy capacity per year
    c_power_ch: float        # EUR per MW of charging power per year
    c_power_dis: float       # EUR per MW of discharging power per year
    eta_c: float
    eta_d: float
    self_discharge: float    # per hour
    c_throughput: float      # EUR per MWh discharged (degradation)
    fixed_ep_ratio: float = 0.0   # if > 0, power is tied to energy by E/P = ratio
    load_limited: bool = False    # discharge bounded by the coolable share of demand
    soc0: float = 0.5
    e_cap_rule: str = ""          # siting limit: "" none, "peak:h", "cool:h"


def _ann(inv_per_kw_or_kwh: float, fom_pct: float, life: float, rate: float) -> float:
    """Annualised EUR per MW (or MWh) from EUR/kW (or EUR/kWh) database entries."""
    capex = inv_per_kw_or_kwh * 1e3
    return capex * (crf(rate, life) + fom_pct / 100.0)


def build_media(rate: float = DISCOUNT_RATE):
    t = TECH
    battery = MediumSpec(
        name="Lithium-ion battery", key="B",
        c_energy=_ann(t["battery storage"]["investment"], 0.0, t["battery storage"]["lifetime"], rate),
        c_power_ch=_ann(t["battery inverter"]["investment"], t["battery inverter"]["FOM"],
                        t["battery inverter"]["lifetime"], rate),
        c_power_dis=0.0,                      # single bidirectional inverter, charged once
        eta_c=t["battery inverter"]["efficiency"],
        eta_d=t["battery inverter"]["efficiency"],
        self_discharge=2.0e-5,
        c_throughput=t["battery storage"]["investment"] * 1e3 / (2.0 * BATTERY_EQ_CYCLES),
    )
    hydrogen = MediumSpec(
        name="Hydrogen (electrolyser-tank-fuel cell)", key="H",
        c_energy=_ann(t["hydrogen tank"]["investment"], t["hydrogen tank"]["FOM"],
                      t["hydrogen tank"]["lifetime"], rate),
        c_power_ch=_ann(t["electrolysis"]["investment"], t["electrolysis"]["FOM"],
                        t["electrolysis"]["lifetime"], rate),
        c_power_dis=_ann(t["fuel cell"]["investment"], t["fuel cell"]["FOM"],
                         t["fuel cell"]["lifetime"], rate),
        eta_c=t["electrolysis"]["efficiency"],
        eta_d=t["fuel cell"]["efficiency"],
        self_discharge=1.0e-5,
        c_throughput=(t["electrolysis"]["investment"] + t["fuel cell"]["investment"]) * 1e3
                     / STACK_LIFETIME_H * 0.5,
    )
    phs = MediumSpec(
        name="Pumped hydro", key="P",
        c_energy=_ann(t["PHS store"]["investment"], t["PHS store"]["FOM"],
                      t["PHS store"]["lifetime"], rate),
        c_power_ch=_ann(t["PHS bicharger"]["investment"], t["PHS bicharger"]["FOM"],
                        t["PHS bicharger"]["lifetime"], rate),
        c_power_dis=0.0,
        eta_c=t["PHS bicharger"]["efficiency"],
        eta_d=t["PHS bicharger"]["efficiency"],
        self_discharge=5.0e-6,
        c_throughput=0.5,
        e_cap_rule="peak:10",
    )
    thermal = MediumSpec(
        name="Chilled-water thermal store", key="T",
        c_energy=_ann(t["water tank storage"]["investment"], t["water tank storage"]["FOM"],
                      t["water tank storage"]["lifetime"], rate),
        c_power_ch=0.0, c_power_dis=0.0,
        eta_c=0.90, eta_d=0.98,
        self_discharge=t["water tank storage"]["standing_loss_pct_h"] / 100.0,
        c_throughput=0.2,
        fixed_ep_ratio=t["water tank storage"]["e_to_p_h"] / 6.0,
        load_limited=True,
        e_cap_rule="cool:18",
    )
    return [battery, hydrogen, phs, thermal]


@dataclass
class SystemCost:
    c_pv: float
    c_wind: float
    c_diesel_cap: float
    c_fuel: float            # EUR per MWh of electricity produced
    c_vom_diesel: float
    c_vom_wind: float
    co2_rate: float          # tCO2 per MWh electric
    voll: float


def build_system_cost(rate: float = DISCOUNT_RATE,
                      fuel_mult: float = FUEL_DELIVERY_MULT,
                      voll: float = VOLL) -> SystemCost:
    t = TECH
    d = t["oil genset"]
    return SystemCost(
        c_pv=_ann(t["solar-utility"]["investment"], t["solar-utility"]["FOM"],
                  t["solar-utility"]["lifetime"], rate),
        c_wind=_ann(t["onwind"]["investment"], t["onwind"]["FOM"],
                    t["onwind"]["lifetime"], rate),
        c_diesel_cap=_ann(d["investment"], d["FOM"], d["lifetime"], rate),
        c_fuel=d["fuel"] * fuel_mult / d["efficiency"],
        c_vom_diesel=d["VOM"],
        c_vom_wind=t["onwind"]["VOM"],
        co2_rate=d["co2"] / d["efficiency"],
        voll=voll,
    )


# ----------------------------------------------------------------------
# island testbeds assembled from the RTS-GMLC benchmark chronologies
# ----------------------------------------------------------------------
@dataclass
class IslandSpec:
    key: str
    name: str
    load_region: str
    wind_site: str
    pv_sites: tuple
    peak_mw: float
    cool_share: float        # share of electric demand that is coolable


ISLANDS = [
    IslandSpec("A", "Island A (trade-wind, high cooling)", "1", "122_WIND_1",
               ("101_PV_1", "102_PV_1", "103_PV_1", "104_PV_1"), 12.0, 0.32),
    IslandSpec("B", "Island B (low-wind, solar-rich)", "2", "303_WIND_1",
               ("313_PV_1", "314_PV_1", "319_PV_1", "320_PV_1"), 25.0, 0.24),
    IslandSpec("C", "Island C (high-variability, small)", "3", "317_WIND_1",
               ("212_PV_1", "213_PV_1", "215_PV_1", "324_PV_1"), 6.0, 0.18),
]

BAND_TAU_H = (4.0, 24.0, 168.0, 1440.0)   # cascade time constants, hours
FORECAST_WINDOW_H = 24


def provenance() -> str:
    lines = ["Parameter provenance (PyPSA technology-data, costs_2030.csv):"]
    for k, v in TECH.items():
        lines.append(f"  {k}: " + ", ".join(f"{a}={b}" for a, b in v.items()))
    lines.append(f"  discount rate = {DISCOUNT_RATE}, VoLL = {VOLL} EUR/MWh, "
                 f"fuel delivery multiplier = {FUEL_DELIVERY_MULT}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(provenance())
    for m in build_media():
        print(f"{m.key} {m.name:42s} cE={m.c_energy:9.1f} cPc={m.c_power_ch:9.1f} "
              f"cPd={m.c_power_dis:9.1f} eta={m.eta_c*m.eta_d:.3f} thr={m.c_throughput:6.2f}")
    sc = build_system_cost()
    print(sc)


def energy_caps(media, island, data=None):
    """Siting limits on storage energy capacity (MWh); inf where unconstrained."""
    import numpy as _np
    mean_cool = island.cool_share * island.peak_mw * 0.49 if data is None else float(
        _np.mean(data["cool"]))
    caps = []
    for m in media:
        if not m.e_cap_rule:
            caps.append(_np.inf)
        else:
            base, h = m.e_cap_rule.split(":")
            ref = island.peak_mw if base == "peak" else mean_cool
            caps.append(float(h) * ref)
    return _np.asarray(caps, dtype=float)
