# Savings-score stress test

I built a rough version of the kind of score Third Space describes on their
site (thirdspacerisk.com, read 2026-09-24), then simulated what happens when bars
act on it. Nothing here is a claim about their actual product or about real
premiums. The premiums and losses are generated; only the feature distributions
for NYC bars come from public inspection and liquor-license data.
[`REPORT.md`](REPORT.md) has the findings, and [`audit/AUDIT.md`](audit/AUDIT.md)
tests which of them hold up.

## Run

Windows / PowerShell, Python 3.11:

```powershell
py -m pip install uv                       # or use any Python 3.11 install
py -m uv venv --python 3.11 .venv
py -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe run_all.py        # all numbers, results/*.csv, figures/*.png
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`run_all.py` takes about 5 minutes on 12 cores (700 stress runs across 20 seeds,
in parallel single-threaded workers). Two consecutive runs give byte-identical
`results/*.csv`. Each stage also prints on its own: `python src/data.py`,
`python src/score.py`, `python src/stress.py`.

The public data is cached in `data/raw/`. Delete it to re-fetch; the NYC data
updates daily, so a re-fetch can shift numbers slightly.

The audit scripts run from `audit/` with the same venv; the order is at the end
of `audit/AUDIT.md`.

## Layout

| file | what |
|---|---|
| `src/data.py` | real-data loaders (cached), synthetic venues, ground-truth loss, carrier simulator |
| `src/score.py` | Savings Score model, per-venue and per-feature scores, promised-vs-causal check |
| `src/stress.py` | adoption sweep, carrier re-pricing, proxy-effect sweep, metrics and tables |
| `tests/test_sanity.py` | ground-truth and metric sanity tests |
| `run_all.py` | regenerates everything |
| `results/` | per-period, per-run and calibration CSVs from the stress test |
| `figures/` | four charts |
| `data/raw/` | cached public data |
| `REPORT.md` | writeup and the full list of assumptions |
| `audit/` | audit of the headline claims: harness, experiments, results, figures, `AUDIT.md` |

## Data provenance

| column | source |
|---|---|
| `prior_violations` | real marginal: NYC DOHMH critical violations 2023-01 to 2026-08 of a bootstrapped real bar ([43nn-pn8j](https://data.cityofnewyork.us/resource/43nn-pn8j)) |
| `nightlife_density` | real marginal: NYS SLA active on-premises licenses in that bar's ZIP ([9s3h-dpkz](https://data.ny.gov/resource/9s3h-dpkz)) |
| `capacity`, `late_night_hours`, `staff_training_score`, `security_staff`, `incident_response_time`, `camera_coverage`, `license_compliance` | synthetic |
| `management_quality`, `unobserved_risk` | synthetic latent, hidden from the carrier and the score |
| `true_expected_loss`, `claim_count`, `loss`, `premium`, adoption | synthetic |

The same labels are in the code as `data.COLUMN_SOURCES`.
