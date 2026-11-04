# Fed-HybridPFN

Code for the paper **"Fed-HybridPFN: Differentially Private In-Context Learning for Cross-Agency Social Analytics over Unlinked Hybrid Data Silos"** (submitted to *IEEE Transactions on Computational Social Systems*).

## About the paper

Agencies and lenders that need to predict income, employment or default risk often hold only a few dozen labelled cases. The outcomes and predictive attributes they need sit with other organisations that cover other regions, record other columns and share no record identifier with them. The data are split both horizontally and vertically (a *hybrid* split), and the records cannot be linked.

Fed-HybridPFN handles this in a single round of communication:

1. **Private release.** Each label-rich agency publishes, once, a set of class-conditional prototypes (noisy means) under zero-concentrated differential privacy (zCDP). Empirical Bayes shrinkage denoises the release, and an adaptive rule sets the number of prototypes from the class size and the budget.
2. **SoftJoin completion.** The server fills in the columns each agency does not record with unbalanced optimal transport (Sinkhorn) between prototypes of the same class, using the shared "bridge" columns as the matching cost.
3. **In-context prediction.** The target agency queries a frozen tabular foundation model (TabICLv2) with its own labels plus a synthetic context drawn from the completed prototypes. The context is sized to the resolution of the release, with a small-class shortcut and a local hedge.

The paper proves record-level and person-level privacy, shows that class matching moves the linked joint law closer to the truth by a conditional mutual information term, and bounds the completion error by the transport cost.

**Main results.** On 1.7 million ACS census records from ten US states, Fed-HybridPFN reaches AUROC 0.836 (ACSIncome) and 0.877 (ACSEmployment) at ε = 1 with 64 target labels. It beats DP federated averaging and local learning in all ten seeds, with one upload of about 3 kB per agency. On two credit benchmarks with small lenders (Taiwan default, Give Me Some Credit) it is the only private method that beats local learning (paired p < 0.01).

## Repository layout

```
code/      Python source (this folder)
results/   raw JSONL logs, audit JSON files and summary JSON used for the tables and figures
matlab/    figure script (matlab_fig_1.m), per-panel data (data/*.json), rendered figures/*.pdf
paper/     single-file LaTeX source (main.tex) with refs.bib, IEEEtran.cls/.bst and figures/
```

## Code files

### Core method and baselines
| File | Description |
|---|---|
| `fedhybrid.py` | Core of the method: zCDP accounting (ε ↔ ρ conversion), DP prototype release with the adaptive count rule and empirical Bayes shrinkage, SoftJoin (unbalanced Sinkhorn) completion, top-k pruned matching, and synthetic context generation. |
| `methods.py` | Federation and all compared methods: Local GBDT, Local TFM, DP FedAvg (with ghost norm clipping) and DP FedAvg + fine-tuning, pooled DP summary, DP synthetic GBDT, pooled raw data (non-private upper reference) and Fed-HybridPFN. Also the context-size gates and cross-fitted temperature scaling. |
| `data.py` | Loaders for ACSIncome / ACSEmployment (via `folktables`, 2018 1-year person files, ten states), Taiwan credit default and GMSC. Public, schema-only encoding (uses no privacy budget) and construction of the hybrid silos, including the disjoint-population control. |
| `run.py` | Default configuration (`DEFAULT`, per-dataset overrides) and the single-run driver. Metrics: AUROC, AUPRC, F1, NLL, Brier, ECE, worst-group AUROC, demographic parity and equalised odds gaps. |

### Experiments
| File | Description |
|---|---|
| `experiments.py` | Named suites: `main` (10 seeds), `ablation`, `gates`, `disjoint`, `ens` (5 seeds), `eps`, `eps_credit`, `kappa`, `nlab`, `phase` (3 seeds), plus `shift`, `msweep`, `mrule`, `scale`. Usage: `python3 experiments.py SUITE WORKER NWORKERS`. Finished runs are skipped on restart. |
| `experiments_v3.py` | Suites for the final method: `main3`, `abl3`, `fact3` (factorial release ablation), `tfm3` (TabICL v1.1 backbone), `tau3`, `donors3` (number of donors), `epsc3`, `kappa3`. |
| `experiments_v6.py` | `gmsc20` (GMSC seeds 10 to 19), `calib6` (temperature scaling, reported as a negative result), `prune6` (top-k pruned matching at K = 10, 20, 40). |
| `queue.sh`, `queue3.sh`, `queue6.sh` | Run the suites above with two parallel workers (`sh queue.sh 0` and `sh queue.sh 1`). |
| `waiter.sh` | Helper that starts the `abl3` suite once `queue3.sh` has finished. |
| `pilot.py` | Pilot of the small-sample refinements (small-class shortcut, local hedge) on the credit tasks. |
| `dev1.py` to `dev4.py` | Short development scripts for the gate, codebook and resolution-rule variants (not used for the reported tables). |

### Audits and diagnostics
| File | Description |
|---|---|
| `audits.py` | Label-free audits: `shrink` (prototype MSE of five estimators vs class size), `mrule` (fixed, linear, adaptive and oracle prototype counts), `cmi` (plug-in I(U;V\|S) and I(U;V\|S,Y)), `shift` (completion error, transported mass and marginal violation under selection), `tau` (completion vs marginal penalty τ), `mia` (black-box membership inference). |
| `diagnostics.py` | Completion error of SoftJoin against the hidden ground truth and membership inference against the deployed predictor. |
| `test_audit.py` | Unit tests for privacy accounting (zCDP vs exact RDP conversion), the Sinkhorn solver and SoftJoin completion. Run with `pytest test_audit.py`. |

### Tables, figures and checks
| File | Description |
|---|---|
| `aggregate.py`, `aggregate_v3.py` | Build the LaTeX tables and the seed-level paired t tests (seed = unit of replication; target silos are averaged within a seed). |
| `make_figs.py` | Writes one JSON per figure panel to `../matlab/data/` from the logs (seed means as logged, no smoothing). |
| `check_style.py` | Scans the LaTeX prose for stray hyphens/dashes and a list of banned words. Usage: `python3 check_style.py ../paper`. |

## Requirements

Python 3.10 or later and:

```
numpy scipy pandas scikit-learn torch pot tabicl folktables pyarrow xlrd pytest
```

The figures are drawn with MATLAB or GNU Octave (`matlab/matlab_fig_1.m`).

## Data

Raw data are not included. Place them in `data_raw/` next to `code/`:

- **ACS 2018 1-year person files**: downloaded automatically by `folktables` on first use.
- **Taiwan default of credit card clients**: UCI repository, saved as `data_raw/taiwan/default of credit card clients.xls`.
- **Give Me Some Credit (GMSC)**: Kaggle, converted to `data_raw/gmsc.pq`.

## Reproducing the paper

```bash
cd code
sh queue.sh 0 & sh queue.sh 1 & wait        # baseline suites
sh queue3.sh 0 & sh queue3.sh 1 & wait      # final-method suites
sh queue6.sh 0 & sh queue6.sh 1 & wait      # GMSC seeds 10-19, calibration, pruning
python3 audits.py shrink income; python3 audits.py mrule income
python3 audits.py cmi income employment; python3 audits.py shift income
python3 audits.py tau income; python3 audits.py mia taiwan income
python3 aggregate_v3.py                     # tables (written to ../paper/table_*.tex)
python3 make_figs.py                        # figure data
cd ../matlab && octave --eval "matlab_fig_1('data','figures')"
cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The paper source is a single `main.tex`. The aggregation scripts write each table to its own `table_*.tex` file. To update the paper, paste the regenerated table into the matching `begin table_*` block of `main.tex`.

## Citation

```bibtex
@article{fedhybridpfn2026,
  title   = {Fed-HybridPFN: Differentially Private In-Context Learning for Cross-Agency Social Analytics over Unlinked Hybrid Data Silos},
  author  = {Author1 and Author2 and Author3},
  journal = {IEEE Transactions on Computational Social Systems},
  year    = {2026},
  note    = {under review}
}
```
