# Co-Design of Storage Portfolios and Duration-Structured Causal Dispatch for Remote Island Power Systems

Reference implementation and full experimental pipeline for the article of the
same name. Every number, table and figure in the paper is produced by this code
from public data; nothing is hard-coded.

## What the code does

An isolated island is sized from photovoltaic, wind, dispatchable and four
storage technologies (lithium-ion battery, hydrogen chain, pumped hydro,
chilled-water thermal store). The novelty is that the inner operating problem
is a **causal** dispatch policy rather than a clairvoyant schedule: a cascade of
first-order filters splits the residual demand into an exactly additive set of
timescale components, and a column-stochastic matrix allocates those components
to the storage media. The whole plant is written with softplus-smoothed
saturations so that capacities and policy weights are optimised together by
reverse-mode gradient descent, with the smoothing width annealed to zero.

Designs are then compared under four operating protocols, from clairvoyant
dispatch (conventional but not implementable) to each method's own controller.

## Data

Two public sources are downloaded automatically on first run:

* Hourly demand, photovoltaic and wind chronologies **and the matching
  day-ahead forecasts** from the RTS-GMLC benchmark
  (`github.com/GridMod/RTS-GMLC`).
* Techno-economic parameters from the PyPSA `technology-data` database
  (`github.com/PyPSA/technology-data`, `outputs/costs_2030.csv`).

Three island testbeds are assembled by pairing distinct demand regions with
distinct resource sites and rescaling to island magnitude. Scenario years are
drawn by a seasonally anchored moving-block bootstrap from disjoint pools of
odd and even calendar weeks, so design years and evaluation years share no
weather.

## Installation

```
pip install -r requirements.txt
```

Python 3.10 or later. JAX runs on the CPU; no accelerator is required.

## Reproducing the article

```
python run_all.py                # everything, a few hours on two cores
python run_all.py designs        # or one stage at a time
python run_all.py figures
```

Results are written to `../results` as pickles and figures to `../figs`.
Set `ISLAND_DATA_DIR` to control where the downloaded data is cached.

## Files

| File | Contents |
|---|---|
| `config.py` | technology parameters, island testbed definitions, annualisation |
| `data_loader.py` | download and assembly of the benchmark chronologies |
| `scenarios.py` | seasonally anchored block bootstrap, train/test separation, perturbations |
| `system.py` | differentiable plant: filter cascade, allocation policy, storage dynamics, cost |
| `codesign.py` | joint capacity and policy optimiser with smoothing annealing |
| `lp.py` | clairvoyant joint sizing and dispatch programmes |
| `mpc.py` | receding-horizon evaluation controller |
| `heuristics.py` | priority-list dispatch with genetic and particle-swarm sizing |
| `pipeline.py` | design construction and evaluation protocols |
| `stats.py` | Wilcoxon, bootstrap intervals, Holm correction, Friedman and Nemenyi |
| `run_main.py` | all planning methods on one testbed |
| `run_reserve.py` | reserve-constrained variants of the clairvoyant planners |
| `run_eval2.py` | restart selection and protocols II to IV |
| `run_p1.py`, `run_p2.py` | clairvoyant protocol and tuned receding-horizon protocol |
| `run_foresight.py` | duration and reliance sweeps of the foresight capture ratio |
| `run_extra.py` | ablation, sensitivity and policy transfer |
| `run_robust4.py` | robustness of every design under its own controller |
| `figures.py`, `fig_arch.py`, `figstyle.py` | all figures, grayscale |
| `run_all.py` | orchestration |

## Notes

* Linear programmes are solved with HiGHS through SciPy. The annual joint
  sizing programme has of the order of 130 000 variables and takes a few
  minutes; the horizon programmes solve in milliseconds.
* The gradient-based design is run from several random restarts and the restart
  with the lowest objective on the design years is selected. No evaluation year
  participates in any selection.
* Reported results use the exact, non-smoothed plant (`mu = 0`).

## License

MIT. See `LICENSE`.
