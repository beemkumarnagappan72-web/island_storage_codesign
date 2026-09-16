"""Reproduce every result in the article.

The stages are ordered by dependency.  Running the whole sequence takes a
few hours on two cores; individual stages can be run on their own with
`python run_all.py <stage>`.
"""
import subprocess, sys, time

ISLANDS = ["A", "B", "C"]
STAGES = [
    ("designs", [["run_main.py", k] for k in ISLANDS]),
    ("reserve", [["run_reserve.py", k] for k in ISLANDS]),
    ("protocols", [["run_eval2.py", k] for k in ISLANDS]),
    ("clairvoyant", [["run_p1.py", k] for k in ISLANDS]),
    ("horizon", [["run_p2.py", k] for k in ISLANDS]),
    ("foresight", [["run_foresight.py", k] for k in ISLANDS]),
    ("ablation", [["run_extra.py", "ablation", k] for k in ISLANDS]),
    ("robust", [["run_robust4.py", k] for k in ISLANDS]),
    ("sensitivity", [["run_extra.py", "sensitivity", "A"]]),
    ("transfer", [["run_extra.py", "transfer", "A"]]),
    ("figures", [["figures.py"], ["fig_arch.py"]]),
]


def main(only=None):
    for name, jobs in STAGES:
        if only and name != only:
            continue
        for job in jobs:
            t0 = time.time()
            print(f"=== {name}: {' '.join(job)}", flush=True)
            subprocess.run([sys.executable] + job, check=True)
            print(f"=== {name} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
