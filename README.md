# Neural-Operator Surrogates for Bayesian Identification of Corrosion Thickness Loss from DIC Strain Fields

Code, configurations and result summaries for the paper. The project replaces finite-element
(FE) analysis inside Bayesian identification of local thickness loss in corroded steel plates.
A physics-informed neural operator (PINO, built on a Fourier neural operator) maps a thickness
field to the plate's strain field. Posteriors over the corrosion parameters are then sampled
from digital image correlation (DIC) measurements, with the surrogate's own error included in
the likelihood.

The test case is the plate of Sato & Nishio (2025): 300 × 250 × 3.2 mm free length, in uniaxial
tension. A real-data check uses the public DIC record of a steel plate with a hole
(Zenodo 11257192).

## Method in brief

1. **Forward model.** A 2D plane-stress membrane FE solver (FEniCSx) with a spatially varying
   thickness τ(x, y). Its output is the strain perturbation relative to the intact plate. A 3D
   solid model (hexahedra, several layers through the thickness) checks the membrane assumption
   for corrosion on one side only.
2. **Thickness priors and datasets.** Parametric pits (centre, radius, depth), Karhunen–Loève
   (KL) random fields, and triangular pits held out as an out-of-distribution set. Sets A (pit),
   B (KL), C (triangle, OOD), D (for estimating surrogate error) and U (unlabelled, for the
   physics loss).
3. **Virtual DIC.** Speckle images are rendered from FE displacements and analysed with a GPU
   subset-DIC engine (first-order shape functions, ZNSSD, IC-GN, cubic B-spline interpolation)
   followed by a virtual strain gauge. From this the code fits the DIC observation operator H
   (Gaussian filter, ℓ_f ≈ 2.6 mm) and a stationary DIC noise covariance Σ_DIC.
4. **Surrogates.** DeepONet, FNO and PINO (FNO + discrete Q1 equilibrium residual on labelled
   and unlabelled inputs), compared on data efficiency (N = 100 / 500 / 2000).
5. **Inference.** Gaussian likelihood with Σ_DIC + Σ_G, where Σ_G is a low-rank surrogate-error
   covariance estimated on held-out FE solves and added through the Woodbury identity. On a
   rectilinear DIC grid Σ_DIC is a Kronecker product, so the full-field likelihood is exact.
   Sampling uses adaptive TMCMC, with NUTS (Pyro) as a cross-check.
6. **Benchmark.** Interval coverage and Wasserstein-1 distance to an FE-in-the-loop reference
   posterior, for two sensor designs (a 56-point grid and the full DIC field), with and without
   Σ_G, and with data from the full rendered-image pipeline.
7. **Applications.** Identification of the hole on the public Zenodo plate, and net-section
   capacity from posterior samples, checked with CalculiX.

## Repository layout

```
pino/                     library
  config.py, grid.py      layered YAML config, operator grid
  thickness.py, data.py   thickness priors (pit, KL, triangle), HDF5 datasets
  fe/plate2d.py           2D membrane FE solver (FEniCSx)
  fe/plate3d.py           3D solid FE solver (one-sided loss, bending check)
  vdic/                   speckle rendering, GPU subset DIC, virtual strain gauge, noise model
  surrogates/             DeepONet / FNO / PINO models, Q1 residual, training loop
  inference/              observation operator, likelihoods (Woodbury, Kronecker), Σ_G,
                          GP model discrepancy, TMCMC, NUTS
  capacity/capacity.py    net-section capacity from thickness samples
experiments/              one script per study step (see table below)
configs/                  base.yaml (full spec) + laptop / quick / improve overlays
tests/test_gates.py       numerical gates (FE vs residual, likelihood vs dense MVN, TMCMC, ...)
scripts/                  pipeline runner, detached launchers, Kaggle bundling
kaggle/                   notebook generators for running training on Kaggle GPUs
docker/                   container image (FEniCSx 0.9, PyTorch 2.7/CUDA 12.6, neuraloperator,
                          Pyro, CalculiX)
data/vdic/                fitted DIC calibration, filter width and noise model (JSON)
results/laptop/           summary tables, metrics and figures of the reported run
```

Datasets (`data/*.h5`, ~4 GB), checkpoints and posterior samples are not stored in the
repository. The scripts regenerate them, and every result JSON records the hash of the config
that produced it.

## Requirements

Everything runs in one Docker image:

```bash
docker build -t pino-surrogate:latest -f docker/Dockerfile docker
```

`docker/run.sh <command>` runs a command in the container with the repository mounted at
`/work`. GPU access: on a Linux host with nvidia-container-toolkit, set `GPU_MODE=nvidia`; on
WSL2 the script passes `/dev/dxg` automatically. A CUDA GPU with ≥ 4 GB is enough for every
stage at the `laptop.yaml` budget.

## Reproducing the results

```bash
./docker/run.sh python3 -m pytest -q tests -p no:cacheprovider   # numerical gates
./docker/run.sh bash scripts/pipeline.sh quick.yaml                # smoke test of all stages (~1-2 h)
./docker/run.sh bash scripts/pipeline.sh laptop.yaml               # reported run (~40 h on a 4 GB GPU)
./docker/run.sh bash scripts/pipeline.sh laptop.yaml study bench   # selected stages only
```

Stages, in order: `verify bending dataset vdic tests study bench nuts zenodo capacity`. Each
stage can be resumed: rerunning skips finished datasets, training runs and posteriors.
`scripts/launch_detached.sh laptop.yaml [stages]` runs the same pipeline in a background
container, logging to `logs/<tag>/`.

Configs are layered: `--config laptop.yaml` is applied on top of `base.yaml`.

| Config | Purpose |
| --- | --- |
| `base.yaml` | Full specification: geometry, priors, DIC settings, dataset sizes, 60k training steps, 100 benchmark cases. |
| `laptop.yaml` | The reported run: same model and priors, smaller budgets (8k steps, 2 seeds, 30–40 benchmark cases). |
| `quick.yaml` | Smoke test only; its numbers are not results. |
| `improve.yaml` | Follow-up studies T1–T5 (layer on `laptop.yaml`). |

Training (step 5) needs only PyTorch and neuraloperator, so it can run on Kaggle:
`scripts/kaggle_bundle.sh laptop.yaml` builds the dataset folder, `kaggle/make_notebook.py`
writes the notebook (2× T4, resumable across 12 h sessions), and `scripts/kaggle_import.sh`
merges the output back in.

### Steps and code

| Step | Code | Output |
| --- | --- | --- |
| 1 2D solver verification | `pino/fe/plate2d.py`, `experiments/step1_verify.py` | `results/<tag>/step1/verify.json` |
| 1b one-sided bending (3D) | `pino/fe/plate3d.py`, `experiments/step1b_bending.py` | `step1/bending_*.json` |
| 2 datasets A–D, U | `pino/thickness.py`, `pino/data.py`, `experiments/step2_dataset.py` | `data/*.h5` |
| 3 virtual DIC, H, Σ_DIC | `pino/vdic/`, `experiments/step3_vdic.py` | `data/vdic/*.json`, `step3/noise_model.png` |
| 4 residual, models, training | `pino/surrogates/` | — |
| 5 data-efficiency study | `experiments/step5_study.py` | `step5/summary.{md,csv}` |
| 6 NUTS cross-check | `pino/inference/nuts.py`, `experiments/step6_nuts.py` | `step6/nuts.json` |
| 7 identification benchmark | `pino/inference/`, `experiments/step7_benchmark.py` | `step7/pit/{metrics.md,calibration.png,detection_map.png}` |
| 7 positive control | `experiments/positive_control.py` | `step7/positive_control*.json` |
| 8a public plate with a hole | `experiments/step8a_zenodo.py` | `step8a/{result.json,posterior.png}` |
| 8c net-section capacity | `pino/capacity/capacity.py`, `experiments/step8c_capacity.py` | `step8c/capacity.json` |
| T1 TMCMC particles, SBC | `experiments/improve_t1_*.py` | `improve/t1/` |
| T2 surrogate error + GP discrepancy on 8a | `pino/inference/discrepancy.py`, `experiments/improve_t2_zenodo.py` | `improve/t2/` |
| T3 when Σ_G matters (regime study) | `experiments/improve_t3_regime.py`, `kaggle/make_notebook_t3.py` | `improve/t3/` |
| T5 3D forward model, both faces | `experiments/improve_t5_*.py` | `improve/t5/` |

Follow-up jobs run as, for example:

```bash
./docker/run.sh python3 experiments/improve_t1_particles.py --config laptop.yaml --config improve.yaml
```

## Results (`laptop.yaml`)

Tables and figures are in `results/laptop/`.

**Surrogate data efficiency (step 5).** Relative L2 error of the strain perturbation after H,
mean of 2 seeds:

| Trained on | N | DeepONet | FNO | PINO | PINO on C (OOD) |
|---|---|---|---|---|---|
| A (pit) | 100 | 0.220 | 0.081 | 0.079 | 0.614 |
| A (pit) | 500 | 0.106 | 0.021 | 0.021 | 0.511 |
| A (pit) | 2000 | 0.095 | 0.013 | 0.013 | 0.491 |
| B (KL) | 100 | 0.297 | 0.086 | 0.084 | 0.484 |
| B (KL) | 500 | 0.162 | 0.020 | 0.020 | 0.226 |
| B (KL) | 2000 | 0.146 | 0.012 | 0.012 | 0.185 |

PINO and FNO are equal within seed scatter at every N: the physics residual adds nothing
measurable at these dataset sizes. Training on KL fields generalises much better to the
triangle set than training on pits. Inference costs ~6 ms per field on a T4.

**Identification benchmark (step 7, 30 pit cases).** On the 56-point grid, 90 % coverage is
0.85–0.93 for every parameter, and the posteriors are within W1 ≤ 0.024 (in prior-width units)
of the FE-in-the-loop reference, with or without Σ_G. At the 30 µε DIC noise level the
surrogate error is far below the noise, so Σ_G does not change the posteriors. The grid
data are only weakly informative (intervals 72–87 % of the prior width); the full field halves
the widths. With data from the full rendered-image pipeline, posteriors lie 0.07–0.21 prior
widths from the reference: the image pipeline adds error that Σ_DIC does not fully capture.
One posterior (TMCMC, 1000 particles) takes 1–5 min with the surrogate, against 4 min (grid)
to 12 min (full field) for 2D FE in the loop on 10 CPU cores.

**NUTS (step 6).** R-hat < 1.01; marginal W1 to TMCMC ≤ 0.053 prior widths.

**Plate with a hole (step 8a, Zenodo data).** Hole centre within 1.9 mm of nominal, but the
intervals are over-confident (r = 3.35 [3.34, 3.37] mm vs 4 mm). Follow-up T2 adds Σ_G and a
Gaussian-process discrepancy: the evidence rises by 1100–1500 nats and the intervals widen
2–30×, but no variant covers the nominal hole jointly. The far-field strain implies
E ≈ 214–248 GPa rather than the 185 GPa reference value.

**Capacity (step 8c, T1).** 90 % coverage of the net-section capacity is 0.76 on the
benchmark. Raising the particle count from 1000 to 4000 changes nothing. Simulation-based
calibration with an exact model gives 0.96 coverage and uniform PIT (KS p = 0.87), so the
under-coverage comes from near-intact truths at the intact-capacity bound and from small-sample
noise, not from the sampler. The one-sided lower bound (q05) holds in 26 of 27 cases.

**Membrane assumption (step 1b, T5).** With corrosion on one side only, the plate bends: the
strain on either face differs from the membrane strain by 1.3–1.6× the damage signal, in
opposite directions on the two faces. Their mean matches the membrane model to within 3 %.
T5 (in progress) adds a 3D solid forward model and surrogates for both faces.

## Implementation notes

- **Load boundary condition.** The right-edge load is a uniform line load of total force P,
  independent of the local thickness, so the FE right-hand side equals the residual's load
  term exactly.
- **Nugget floor.** The fitted DIC nugget is tiny and makes the full-field covariance nearly
  singular (condition number ~1e12), so float32 surrogate round-off dominated the likelihood.
  The likelihood uses a white-noise floor of 0.1 σ (≈ 3 µε).
- **TMCMC** tunes its proposal scale towards 23 % acceptance and keeps sweeping until 60 % of
  particles have moved.
- **Σ_G across load steps.** The surrogate error scales with P, so Σ_G is stacked over loads;
  the variance missed by its top 20 PCs is added to the diagonal.
- **FNO padding** is 144 × 168 instead of a 0.125 fraction, which would give a grid size with
  the prime factor 71 and a ~10× slower cuFFT.
- **Relative L2** is pooled over the test set (‖e‖/‖t‖ over all samples), because near-intact
  samples make per-sample ratios meaningless.
- **DIC calibration target.** `dic.calib_target_std_ue = 30` µε is a placeholder for the
  scatter of the target experiment; changing it means rerunning the `vdic` stage and every
  stage after it.

## Data and references

- Sato & Nishio (2025): plate geometry and test setup.
- Tröger, Hartmann, Anton & Wessels, *Digital image correlation measurement of linear elastic
  steel specimen*, Zenodo, record 11257192. `experiments/step8a_zenodo.py` downloads it into
  `data/zenodo_11257192/`. Please cite the dataset if you use it.
- Anton et al., arXiv:2405.18311 (traction and reference moduli for the Zenodo plate).

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## License

The code is released under the MIT License (see `LICENSE`). The Zenodo dataset used in step 8a
is not part of this repository and keeps its own license.
