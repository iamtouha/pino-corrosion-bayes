# PINO surrogates for Bayesian corrosion identification

This repository contains the code for the plan *PINO Surrogates for Bayesian Corrosion
Identification: Plan, Theory and Implementation* (Steps 0–8). The code trains neural
operators that replace the FE analysis in a Bayesian identification of local thickness loss.
The data are DIC strain fields of a steel plate in tension. The plate is the plate of
Sato and Nishio (2025): free length 300 mm, width 250 mm, thickness 3.2 mm.

For a summary of the method and the results, read `REPORT.md`. This README gives the
procedures, the implementation notes and the detailed results.

Writing style: this README uses a controlled-language style similar to ASD-STE100
(Simplified Technical English).

All code runs in one Docker image. The image contains FEniCSx 0.9, PyTorch 2.7 (CUDA 12.6),
neuraloperator 1.0.2, Pyro 1.9, CalculiX, h5py, scipy and matplotlib.

## Quick start

1. Build the image:
   ```bash
   docker build -t pino-surrogate:latest -f docker/Dockerfile docker
   ```
2. Run the tests that do not need a trained model:
   ```bash
   ./docker/run.sh python3 -m pytest -q tests
   ```
3. Run the full pipeline at smoke-test size (approximately 1–2 h):
   ```bash
   ./docker/run.sh bash scripts/pipeline.sh quick.yaml
   ```
4. Start the real run in a detached container (approximately 2 days on a laptop GPU):
   ```bash
   ./scripts/launch_detached.sh laptop.yaml
   ```
5. Monitor the run:
   ```bash
   tail -f logs/laptop/pipeline.log
   ```

`docker/run.sh` mounts the repository at `/work`. On WSL2, the script gives the GPU to the
container through `/dev/dxg`. The nvidia-container-toolkit is not necessary. On a usual
Linux host, set `GPU_MODE=nvidia`.

## Configurations

| File | Use |
| --- | --- |
| `configs/base.yaml` | All numbers of the plan: geometry, priors, DIC settings, data set sizes, 60 000 training steps, 100 benchmark cases. On a 4 GB laptop GPU, this budget needs more than 3 weeks. |
| `configs/laptop.yaml` | The same method with smaller budgets for an RTX 3050 Ti laptop GPU (approximately 2 days): 8000 steps, 2 seeds, 30 benchmark cases, PINO only in the benchmark. |
| `configs/quick.yaml` | A smoke test of all stages (small data, some hundred steps). Do not use its results for conclusions. |
| `configs/improve.yaml` | Settings of the improvement runs (tasks T1–T7). Use it on top of `laptop.yaml`. |

The configurations are layers. `--config laptop.yaml` applies on top of `base.yaml`. A
later file overrides an earlier file. The results go to `results/<tag>/`. The data go to
`data/` (`data/quick/` for the smoke test). Each result JSON file contains the hash of its
configuration.

## Code and outputs for each step

| Plan step | Code | Output |
| --- | --- | --- |
| 1 2D plate solver and gate | `pino/fe/plate2d.py`, `experiments/step1_verify.py` | `results/<tag>/step1/verify.json` |
| 1b one-sided bending (3D) | `pino/fe/plate3d.py`, `experiments/step1b_bending.py` | `step1/bending_*.json` (decision: membrane or shell) |
| 2 data sets A–D, U | `pino/thickness.py`, `pino/data.py`, `experiments/step2_dataset.py` | `data/*.h5` |
| 3 virtual DIC, H, Σ_DIC | `pino/vdic/engine.py` (rendering, GPU subset DIC, VSG), `pino/vdic/noise.py`, `experiments/step3_vdic.py` | `data/vdic/{calibration,filter_fit,noise_model,noise_fit}.json`, `step3/noise_model.png` |
| 4 residual, models, training | `pino/surrogates/{residual,models,train}.py` | — |
| 4.2 and 6.2 gates | `tests/test_gates.py` | pytest |
| 5 data-efficiency study | `experiments/step5_study.py` | `step5/summary.{md,csv}`, `step5/ckpt/*.pt` |
| 6 observation, likelihood, TMCMC, NUTS | `pino/inference/{problem,likelihood,tmcmc,nuts}.py`, `experiments/step6_nuts.py` | `step6/nuts.json` |
| 7 benchmark | `experiments/step7_benchmark.py` | `step7/pit/{metrics.md,calibration.png,detection_map.png}` |
| 8a public plate with a hole | `experiments/step8a_zenodo.py` | `step8a/{result.json,posterior.png}` |
| 8b own coupons | needs physical tests | use the 8a code path (grid operator, geometry from the configuration) as a template |
| 8c capacity | `pino/capacity/capacity.py`, `experiments/step8c_capacity.py` | `step8c/capacity.json` |
| Improvement runs | `experiments/improve_*.py`, `pino/inference/discrepancy.py` | `results/laptop/improve/<task>/` |

## Implementation notes and deviations

Read these notes before you change the code.

- **Load boundary condition.** The FE model applies a uniform line load with a total force P
  on the right edge. This load does not depend on the local thickness. Thus, the FE
  right-hand side is equal to the vector `f_hat` of the residual. The FE example of the plan
  multiplies the traction by τ. That version breaks the gate "FE data = physics loss" if
  damage reaches the loaded edge.
- **DIC engine.** The code does not use DICe, because DICe needs Trilinos on Linux.
  `pino/vdic/engine.py` is a GPU version of the same local method: first-order subset shape
  function, ZNSSD criterion, IC-GN iterations, integer ZNCC seeds and **cubic B-spline
  interpolation**. Bicubic (Keys) interpolation caused a bias of 0.02–0.03 px. This bias gave
  strain errors of ±40 µε, which is more than the damage signal. B-splines decrease the bias
  by a factor of approximately 10 (to approximately 0.002 px). The derivative over a 5 mm VSG
  window still has pattern-dependent errors of approximately 10 µε. These errors are part of
  the fitted Σ_DIC.
- **Filter width ℓ_f.** Pits with r ≥ 8 mm are too smooth to identify ℓ_f. A Gaussian filter
  of 2–3 mm changes their peak strain by approximately 2 µε, which is less than the DIC
  error. Thus, the code fits ℓ_f from the DIC transfer function of sinusoidal strain fields
  (λ = 6–32 mm, mean over speckle patterns). The result is ℓ_f ≈ 2.6 mm. This value agrees
  with a 7.2 mm subset and a 5 mm VSG window. The deep pits remain as a validation check.
- **Calibration target (3.5).** The histogram of Fig. 8 of the reference paper is not
  available. `dic.calib_target_std_ue = 30` is a **placeholder** (±60 µε at 50 kN, read as
  ±2σ). Replace it with the value from the figure. Then run the `vdic` stage and all later
  stages again.
- **Σ_DIC.** The model is a stationary squared-exponential kernel plus a nugget for each
  strain component. On a rectilinear DIC grid, this covariance is a Kronecker product. Thus,
  the full-field likelihood is exact and uses only 1D eigendecompositions. The code adds
  Σ_G with the Woodbury identity. The low-rank `loglik` of the plan is also in the code. A
  test compares it with a dense MVN.
- **Nugget floor in the likelihood.** The fitted nugget of the virtual DIC is approximately
  10^-6 of σ, because the VSG output is smooth. The full-field SE covariance then has a
  condition number of approximately 10^12. In this condition, float32 round-off in the
  surrogate controls the log-likelihood. The log-likelihood changed by 300–700 for a change
  of δ of 10^-5, and TMCMC collapsed onto one particle. Thus, the likelihood uses a white-noise
  standard deviation of at least `inference.nugget_floor_frac` = 0.1 σ (approximately 3 µε).
  With this floor, the same scan changes by 0.2. The noise emulator keeps the fitted model.
  With Σ_G, the nugget was already sufficiently large. Thus, only the full-field condition
  without Σ_G failed.
- **TMCMC** is adaptive. After each MH sweep, the code tunes the proposal scale to 23 %
  acceptance. A stage continues (maximum 5 × n_mh sweeps) until 60 % of the particles have moved.
- **Σ_G across loads.** The surrogate error at unit load is fully correlated across the load
  steps, because it scales with P. Thus, Σ_G is stacked over the loads. The variance that
  its 20 largest principal components do not contain goes into the nugget.
- **Set D** contains 500 new pit samples and 500 new KL samples (one Σ_G for each family).
- **FNO padding** gives a 144 × 168 grid (approximately 12–14 %). The plan value 0.125 gives
  142 × 170 = 2·71 × 2·5·17. The prime factor 71 makes cuFFT approximately 10 times slower.
  The positional embedding is off, because x and y are already input channels.
- **Relative L2 error** is pooled over the batch or the test set (‖e‖/‖t‖ over all samples).
  Intact and almost intact samples have w ≈ 0. Thus, a ratio for each sample has no meaning.
  The code also gives the median of the per-sample error.
- **8a boundary conditions.** The specimen was clamped at x = 0. The operator uses a clamped
  left edge (`realdata.bc`) and three rigid-body parameters. It ignores the observations
  within 2 mm of the two ends. The traction (106.26 N/mm²) and the reference moduli come
  from Anton et al. (arXiv:2405.18311, Sec. 6). The hole is a region with τ = 10^-3. Its
  strain agrees with a meshed hole within 1 % at 2 mm or more from the edge (1.06 % at 1 mm).
- **8b** needs coupons and a testing machine. It cannot run here. **8d** waits for lab data.
- **Bending (1b): the decision is "shell".** A 3D solid model (Q1 hexahedra, 4 layers) shows
  that one-sided loss bends the plate. After H, the strain on each face differs from the 2D
  membrane strain by more than the damage signal: 1.24–1.41 times in the maximum and
  1.57–1.74 times in the L2 norm (centred pits, δ = 0.05–0.4 mm). The two faces move in
  opposite directions. The mean of the two faces agrees with the membrane model within
  approximately 3 % of the signal. The baseline does not have a shell or 3D surrogate. Thus,
  the synthetic benchmark (Steps 2–7) is self-consistent and tests the method. But for
  single-face DIC of one-sided corrosion, the method needs a shell or 3D forward model, or DIC
  on both faces (averaged). Improvement task T5 adds the 3D path.
- **3D solver, mid-thickness supports (fixed 2026-11-23).** `pino/fe/plate3d.py` holds the
  mid-thickness nodes of the two short faces in z. The mesh follows τ, so these nodes are at
  0.5·t0·τ. The first version searched at 0.5·t0. It found no nodes when a pit reached
  x = 0 or x = L, and the solve stopped. The Step 1b pits are at the centre, so their results
  do not change. The test `test_plate3d_edge_pit_and_intact` covers this case.

## Measured performance on this computer

The computer has an RTX 3050 Ti laptop GPU (4 GB) under WSL2. The GPU throttles thermally.

- 2D FE solution: 0.2 s (38 000 unknowns, MUMPS). 11 300 solutions take approximately 10 min on 10 workers.
- 3D FE solution (4 layers): 42–48 s on one core.
- Virtual DIC, full field (217 × 200 points, 5000 × 4200 px): approximately 56 s for each image pair on the GPU.
- FNO training: batch 16 takes approximately 0.3–0.4 s for each step, with a peak of 2.9 GB.
  PINO takes approximately 0.8 s for each step, with a peak of 3.0 GB. The code back-propagates
  the labelled and the unlabelled residual terms one after the other. In one graph, the PINO
  peak was 5.5 GB. On a 4 GB GPU under WSL, the driver then moves memory to the system RAM without an error
  message. A step then took 4.8 s (6 times slower).
- Surrogate log-likelihood in TMCMC: approximately 7 ms for each evaluation at batch 16.
  Batch 64 is slower, because it is more than the GPU memory.

**CAUTION:** Keep the peak GPU memory below 3.5 GB. Above 4 GB, the WSL driver moves memory to the system RAM without an error message. The
speed then decreases by a factor of approximately 6.

## Procedure: run the laptop configuration

The real run (`configs/laptop.yaml`) runs in the detached container `pino-laptop`.

1. Monitor the stage lines (start, OK, FAIL):
   ```bash
   tail -f logs/laptop/pipeline.log
   ```
2. Monitor the details of the current stage:
   ```bash
   tail -f logs/laptop/step5_study.log
   ```
3. To stop the run:
   ```bash
   docker stop pino-laptop
   ```
4. To start the run again:
   ```bash
   docker rm pino-laptop; ./scripts/launch_detached.sh laptop.yaml
   ```
5. To run only some stages:
   ```bash
   ./docker/run.sh bash scripts/pipeline.sh laptop.yaml study bench
   ```

A restart does not repeat finished work. This applies to the data sets and to the parts of Step 3 (calibration, filter, noise model).
It also applies to each training run of Step 5 and each posterior of Step 7. Steps 1 and 1b run again (approximately 8 min). NUTS, 8a and 8c also run again. An
interrupted noise fit starts again at its first field.

The expected time on this laptop is approximately 40 h: noise fit approximately 6 h, study
approximately 28 h, benchmark 6–10 h, and NUTS, 8a and 8c approximately 3 h. Step 5 can run
on Kaggle (next section). `laptop.yaml` limits NUTS to 2 chains × 500 samples + 500 warm-up
and a tree depth of 8 or less. The quick run measured 5–9 s for each NUTS iteration. Thus, the
plan value of 4 × 1500 would take approximately 10 h for each case.

## Procedure: run Step 5 (training) on Kaggle

Step 5 uses approximately 70 % of the laptop budget. It needs only PyTorch and
neuraloperator, not FEniCSx. Thus, it can run on the free Kaggle GPUs, and the other steps
stay local.

1. Make the upload folder (approximately 2.3 GB: sets A, B and C; set U without its fields;
   the DIC JSON files; the code; finished Step 5 runs, if there are any):
   ```bash
   ./scripts/kaggle_bundle.sh laptop.yaml
   ```
2. On kaggle.com, select **Datasets → New Dataset**. Put the contents of `kaggle/upload/` in
   the dataset. Alternatively, put your user name in `dataset-metadata.json` and run
   `kaggle datasets create -p kaggle/upload --dir-mode tar`. Use the name `pino-step5`.
3. Select **Notebooks → New → File → Import Notebook** and import `kaggle/pino_step5.ipynb`.
   Add the dataset as input. Set **Accelerator: GPU T4 x2** and **Internet: on**. Then select
   **Save Version → Save & Run All (Commit)**. The notebook runs without the browser. With two
   GPUs, family A trains on one GPU and family B on the other.
4. A session has a limit of 12 h. The notebook starts no new run after `DEADLINE_H` = 10.75 h.
   If runs remain, make a new version and add the output of the previous version as input.
   The notebook copies the finished runs and does not repeat them.
5. Download the output (zip) of the version. Merge it, and then finish the pipeline locally:
   ```bash
   ./scripts/kaggle_import.sh ~/Downloads/output.zip
   ./scripts/launch_detached.sh laptop.yaml study bench nuts zenodo capacity
   ```

**CAUTION:** Select GPU T4 x2. Recent PyTorch CUDA builds do not contain kernels for the P100
(sm_60). The notebook stops if the GPU has no kernels. `kaggle/make_notebook.py` makes the
notebook again.

## Laptop-run results (`laptop.yaml`, 2026-11-23)

Step 5 ran on Kaggle (2 sessions, 2 × T4, 36 runs). Steps 6–8 ran locally after Step 5
(`study bench nuts zenodo capacity`, 10 h 05 min). All stages passed. The tables are in
`results/laptop/step5/summary.md` and `results/laptop/step7/pit/metrics.md`. The plots are
in the same folders. `REPORT.md` discusses these results.

**Step 5, data efficiency: gate PASS.** The PINO error at N = 2000 on its own family is
0.013 (A) and 0.012 (B). The gate is 0.10. The table gives the relative L2 error of the
perturbation strain after H (mean of 2 seeds).

| Trained on | N | DeepONet | FNO | PINO | PINO on C (triangle, outside the training distribution) |
|---|---|---|---|---|---|
| A (pit) | 100 | 0.220 | 0.081 | 0.079 | 0.614 |
| A (pit) | 500 | 0.106 | 0.021 | 0.021 | 0.511 |
| A (pit) | 2000 | 0.095 | 0.013 | 0.013 | 0.491 |
| B (KL) | 100 | 0.297 | 0.086 | 0.084 | 0.484 |
| B (KL) | 500 | 0.162 | 0.020 | 0.020 | 0.226 |
| B (KL) | 2000 | 0.146 | 0.012 | 0.012 | 0.185 |

- **The PINO is equal to the FNO at all N.** The differences are in the range of the seed
  scatter. The selection of λ_r at N = 500 found a value where the residual decreases but
  the data loss does not change. With the Q1 residual, the physics term has no measurable
  effect at these data sizes. Training on B (KL fields) gives a much better result on C than
  training on A.
- FNO and PINO inference takes approximately 6 ms for each field on a T4 (5.5–6.2 s for 1000 fields).

**Step 7, benchmark (pit family, 30 cases, 3 intact, full-field design on the first 6).**

| Condition | Design | 90 % coverage xc / yc / r / δ | W1 to FE ref. (max. of means) | Gates (cov90, W1) |
|---|---|---|---|---|
| PINO N=100, with / without Σ_G | grid56 | 0.93 / 0.85 / 0.93 / 0.85–0.89 | ≤ 0.024 | pass, pass |
| PINO N=2000, with / without Σ_G | grid56 | 0.93 / 0.85 / 0.85–0.89 / 0.89 | ≤ 0.020 | pass, pass |
| PINO N=100 / 2000 | full | 1.0 (3 damaged cases) | ≤ 0.072 | fail (n = 3), mixed |
| PINO N=2000, full image pipeline | grid56 / full | 0.67–1.0 (3 cases) | 0.07–0.21 | fail |
| FE in the loop (reference) | grid56 / full | 0.67–1.0 (3 cases) | — | — |

- **Σ_G has no effect** in this setup. With the 30 µε (placeholder) DIC noise, the surrogate
  error (1.3–8 %) is much smaller than the noise. Thus, the two likelihoods give the same
  posteriors. The expected result of the plan (a coverage loss without Σ_G) does not occur.
  It needs a regime where the surrogate error is near the noise level (lower noise, or data
  outside the training distribution).
- **The 56-point grid is weakly informative.** The 90 % intervals are 72–87 % of the prior
  intervals. Intact plates still get P(δ > 0.01) ≈ 0.89 (prior 0.93). The detection map shows
  that pits with a depth of approximately 0.3 mm or more and r ≳ 20 mm get P ≈ 1. The full
  field decreases the interval widths by approximately 50 %.
- **Full image pipeline against ideal data.** Posteriors from rendered-image DIC are 0.07–0.21
  of the prior width from the FE reference. With emulator noise, the distance is 0.07 or less.
  Thus, the image pipeline adds error that Σ_DIC does not fully contain.
- Cost of one posterior (TMCMC, 1000 particles, 7000–14 000 evaluations): 55–300 s with the
  surrogate (the time depends on the GPU clock). FE in the loop on 10 CPU workers takes 4 min
  (grid56) or 12 min (full).

**Step 6, NUTS** (2 chains × 500 + 500 warm-up, cases 3 and 4, 24 min each): R-hat 1.007 and
1.001 (less than 1.01). ESS 197 and 159 (less than 400). Thus, NUTS is not accepted at this
limited size. The marginal W1 to the TMCMC posteriors is 0.053 of the prior width or less.
The two samplers agree.

**Step 8a, Zenodo plate with a hole: gate PASS.** The hole centre is (38.9, 8.5) mm against
(40, 10). The error (1.9 mm) is less than r. E = 185.4 GPa [184.8, 185.9] against the
reference of 185 GPa (0.14 %). This result replaces the 224–227 GPa of the earlier FE
least-squares fit. But the intervals are too narrow: r = 3.35 [3.34, 3.37] against 4 mm,
and ν = 0.248. Thus, the posterior is over-confident. The held-out strain error of the
surrogate is 4.3 %, and this likelihood has no surrogate-error term. See T2 below: the
agreement of E is not strong evidence.

**Step 8c, capacity.** The data are the posterior samples of 54 damaged grid56 cases
(N = 100 and 2000 with Σ_G). Their net-section capacity has a 90 % coverage of **0.76**. This is less than
the parameter coverage (0.85–0.93). CalculiX on the deepest case gives 192.3 kN for the true
damage and 191.6–194.4 kN for 8 posterior samples (median 192.8). See T1 below for the cause.

**Open points:** the bending decision (1b) is "shell", and the baseline has no shell model.
The 30 µε DIC calibration target is a placeholder. NUTS at full size (4 × 1500) did not run.

## Improvement runs (from 2026-11-23)

The improvement runs use the configuration layer `configs/improve.yaml`
(`--config laptop.yaml --config improve.yaml`). The tag stays `laptop`. Thus, the runs use
the Step 5 checkpoints and the data sets of the baseline. The outputs go to
`results/laptop/improve/<task>/`. The baseline results do not change.

To start an improvement job in a detached container:
```bash
./scripts/launch_improve.sh <task> python3 experiments/improve_<task>_*.py --config laptop.yaml --config improve.yaml
```
The container name is `pino-improve-<task>`. The log is `logs/laptop/improve_<task>.log`.
To run a job without the GPU, set `GPU_MODE=none`.

New tests in `tests/test_gates.py`:
- the Woodbury likelihood against a dense MVN,
- the type-II ML recovery of GP hyperparameters,
- TMCMC with a GP-discrepancy likelihood against the analytic posterior,
- the 3D solver with a pit at the loaded edge and with an intact plate.

### T1: TMCMC particle count and capacity coverage — done

`experiments/improve_t1_particles.py` reruns the Step 7 posterior (PINO N = 2000 with Σ_G,
grid56, the Step 7 data) on the first 10 damaged cases (3–12). It uses n = 1000 with
n_mh = 2 (laptop) and n = 4000 with n_mh = 3 (base). The seeds are case + 0 and case + 1000.
The capacity is `net_section` on **all** particles and on the 200-sample subsample of
Step 8c. On the shared GPU, a posterior took 76–93 s at n = 1000 and approximately 13 min at
n = 4000 (38 000 evaluations).

| n | seed | 90 % cov. xc / yc / r / δ | δr² | capacity (all particles) | capacity (200 sub) | unique frac. |
|---|---|---|---|---|---|---|
| 1000 | +0 | 0.90 / 0.70 / 0.90 / 1.00 | 0.80 | 0.80 | 0.70 | 0.86 |
| 1000 | +1000 | 0.90 / 0.80 / 0.90 / 1.00 | 0.80 | 0.80 | 0.70 | 0.85 |
| 4000 | +0 | 0.90 / 0.70 / 0.80 / 0.90 | 0.70 | 0.70 | 0.80 | 0.86 |

- **More particles change nothing.** The n = 1000 and n = 4000 posteriors differ by
  W1 ≈ 0.02 of the prior width (mean over the cases). This is less than the seed scatter at
  n = 1000 (0.03–0.045). The capacity PIT of each case is the same within ±0.03 in all three
  runs (for example, case 5: 0.96, 0.94, 0.97). The coverage differences come from cases at
  the 90 % limit. **Decision: keep n = 1000** for all later inference. n = 4000 costs 8–10
  times more.
- The Step 8c procedure decreases the coverage. With all 1000 particles, the baseline
  capacity coverage on 27 damaged cases is 0.81 (N = 2000) and 0.78 (N = 100). With the
  200-sample subsample, it is 0.78 and 0.74 (0.76 together).
- **Location of the misses.** In the 27 baseline cases, 4 of the 5 misses have the truth
  *above* q95. All 4 are almost intact plates (true capacity 318.9–319.9 kN, intact
  320 kN, δ·r ≤ 0.34 mm). The intact value is an upper limit of the capacity. The grid56 data
  are weakly informative, so the posteriors of these plates stay near the prior. Thus, an
  equal-tailed interval puts these truths in its upper tail. Under the prior, only
  approximately 3 % of the plates lose less than 0.3 kN. The truth set has 3 of 27 (Poisson
  p ≈ 0.04). With 27 cases, the standard error of a 0.9 coverage is 0.058. Thus, 0.81 is
  1.5 standard errors below 0.9. The one-sided lower bound (truth ≥ q05) holds in 26 of 27 cases.
- **SBC** (`experiments/improve_t1_sbc.py`, `results/laptop/improve/t1/sbc/`). The check uses
  50 replications. Each truth comes from the prior. The data are the PINO N = 2000 prediction
  plus a sample of the noise covariance of the likelihood. TMCMC uses n = 1000. With this
  exact model, the intervals are calibrated:

  | Quantity | 90 % cov. | Lower 5 % bound cov. | PIT KS p |
  |---|---|---|---|
  | xc / yc / r / δ | 0.92 / 0.92 / 0.90 / 0.96 | 0.94 / 0.96 / 0.96 / 0.96 | 0.94 / 0.36 / 0.65 / 0.10 |
  | δr² | 0.98 | 1.00 | 0.72 |
  | Net-section capacity | 0.96 | 0.98 | 0.87 |

  The standard error of a coverage at 50 replications is 0.04. Almost intact truths (loss
  less than 1 kN, 4 of 50) have a capacity coverage of 0.75. This is the same limit effect
  as in Step 7. The plan had 100 replications. We stopped at 50 to give the GPU to T3.
- **Decision.** The capacity under-coverage is not a sampling artefact. It is also not a
  property of TMCMC or of the derived quantity. Thus, n = 1000 stays. The Step 7 value of
  0.76 has three causes: the 200-sample procedure (−0.03), a truth set with more almost
  intact plates than the prior gives, and the binomial noise of 27 cases. For engineering
  use, give the one-sided lower bound (q05: 26 of 27 in Step 7, 0.98 in SBC).
- The second seed at n = 4000 did not run. We stopped it to give the GPU to SBC. The
  posteriors of each case already agree between the seeds at n = 1000 and n = 4000 within the
  seed scatter at n = 1000.

### T2: over-confidence of Step 8a (Zenodo plate with a hole) — done, not accepted

`experiments/improve_t2_zenodo.py` and `pino/inference/discrepancy.py`. The runs use the same data as Step 8a (1072 interpolated points, u_x and u_y). They also use
the same PINO (`results/laptop/step8a/pino_hole.pt`, held-out strain error 4.3 %), the same
priors and the same rigid-body terms. TMCMC uses n = 2000 and n_mh = 3 (15–30 min for each posterior on the
shared GPU).

- **Σ_G** comes from 300 new FE solutions of the hole family (rank 20 plus the remaining
  diagonal). Its RMS is 0.8 µm in u_x and 1.8 µm in u_y at E = 185 GPa.
- **Discrepancy:** a zero-mean Gaussian process over (x, y) for each component. The kernel is
  a sum of 1, 2 or 3 SE terms plus white noise. Type-II ML on the residual of the current MAP
  gives the hyperparameters (with Σ_G in the covariance). There are two empirical-Bayes passes.
- **Taper:** one asymmetry parameter g (a linear gradient of the axial strain, as a kinematic correction).

The table is `results/laptop/improve/t2/summary.md`. The plots are `intervals.png` and `map_residuals.png`.

| Condition | xc [90 %] | yc [90 %] | r [90 %] | E GPa [90 %] | logZ |
|---|---|---|---|---|---|
| iid (= Step 8a) | 38.10 [37.94, 38.28] | 8.90 [8.77, 9.00] | 3.36 [3.35, 3.37] | 187.0 [186.5, 187.4] | 10768 |
| iid + Σ_G | 38.77 [38.60, 38.93] | 7.26 [7.18, 7.33] | 2.86 [2.83, 2.89] | 183.7 [182.3, 184.7] | 12115 |
| taper + Σ_G | 37.97 [37.59, 38.40] | 10.96 [10.79, 11.12] | 2.58 [2.56, 2.59] | 191.3 [190.3, 192.6] | 12073 |
| GP(1 SE) + Σ_G, pass 0 / 1 | 39.69 [37.88, 40.61] / 38.21 [38.05, 38.39] | 9.52 [8.91, 11.13] / 9.45 [9.30, 9.59] | 4.41 [4.30, 4.66] / 3.89 [3.83, 3.93] | 222.5 [212.1, 227.6] / 216.2 [215.0, 218.0] | 13244 / 13257 |
| GP(2 SE) + Σ_G, pass 0 / 1 | 39.15 [38.86, 39.43] / 38.48 [37.80, 39.21] | 8.53 [8.21, 8.81] / 9.77 [8.66, 10.68] | 4.20 [4.11, 4.33] / 4.39 [4.18, 4.63] | 197.7 [196.0, 199.7] / 214.2 [198.8, 226.0] | 13284 / 13319 |
| GP(2 SE) + taper + Σ_G, pass 0 / 1 | 39.46 [37.67, 41.40] / 39.28 [39.03, 39.49] | 10.29 [9.53, 11.89] / 8.53 [8.35, 8.74] | 3.39 [2.59, 3.73] / 4.39 [4.20, 4.50] | 218.1 [204.4, 224.6] / 226.2 [223.7, 227.6] | 13284 / 13292 |
| GP(3 SE) + Σ_G, pass 0 / 1 | 38.56 [38.05, 38.80] / 39.11 [38.55, 39.40] | 8.88 [8.26, 12.84] / 8.62 [8.22, 9.34] | 3.56 [3.00, 3.70] / 4.25 [4.12, 4.35] | 164.8 [156.0, 228.4] / 159.0 [152.9, 174.3] | 13321 / 13330 |

**Result: not accepted.** No condition has (40, 10) mm and r = 4 mm in the 90 % intervals
together. No condition also keeps E within 5 % of 185 GPa.

- Σ_G alone makes the result *worse* (r 2.86, yc 7.26). The low-rank surrogate-error modes
  absorb most of the u_y misfit (σ_y decreases from 8.1 to 1.9 µm), and the hole parameters
  move further.
- The GP discrepancy increases the evidence by 1100–1500 nats over iid + Σ_G. It makes the
  intervals 2–30 times wider (r: 0.02 → 0.1–1.1 mm). Thus, the over-confidence of Step 8a
  decreases. But the posteriors change between the empirical-Bayes passes (for example,
  GP 1 SE: xc 39.7 → 38.2), and the hole stays biased (r 3.4–4.4, centre 0.5–1.9 mm off).
  The ML length scales are short (0.5–2.6 mm, correlated DIC and interpolation noise). Only
  the 3-term kernel keeps a long-range term (ℓ ≈ 19–26 mm, 7–9 µm in u_x). Then E is almost
  not identifiable (153–228 GPa). This is the usual Kennedy–O'Hagan confounding between a
  smooth discrepancy and a global stiffness.
- The data do not support the taper parameter (logZ is lower than without it; g = −0.19).
- **The data do not agree with E = 185 GPa.** Away from the hole, the axial strain of the
  gross section is 496 µε (x = 2–12 mm) and 429–437 µε (x = 12–25 mm and 47–78 mm) at
  106.26 MPa. This gives an apparent E of 214–248 GPa. E = 185 GPa needs 574 µε. The "E gate
  pass" of Step 8a came from an iid fit. That fit accepts a far-field misfit (−60 to −120 µε
  in the MAP residual) for a better fit near the hole. The GP posteriors (198–226 GPa) agree
  with the far-field strain.
- The raw DIC shows the hole near its nominal position. The largest empty circle is at
  (39.76, 10.27) with R = 3.66 mm. A circle through the innermost points is at (39.61, 9.91)
  with R = 4.11 mm. Thus, the low r comes from a model misfit at the hole edge. These causes are possible (not verified):
  - local yield: the net-section stress is 177 MPa, and K_t ≈ 2.2 gives approximately 390 MPa at
    the edge. This is more than the nominal yield of S235.
  - the DIC facets (25 px) and the linear interpolation smooth the field at the edge.
  - the τ = 10^-3 representation of the hole has a 1 % strain error at 2 mm or more from the
    edge and 11 % at the peak.
- Possible next steps: a non-stationary discrepancy (variance that increases near the hole
  edge), a band without observations at the edge, or an elasto-plastic forward model.

### T3: when does Σ_G have an effect? — running

`experiments/improve_t3_regime.py`. The study gives the 90 % coverage with and without Σ_G,
and W1 to an FE reference, as a function of the ratio (surrogate error / DIC noise).

- **Cross-family surrogates:** pit inference with the KL-trained PINO N = 100 and N = 2000.
- **Noise sweep:** the A-trained PINO N = 100 and N = 2000 at noise scales 1, 1/3 and 1/10.
  The data and the likelihood use the same scale. The data are
  `clean + s · (d − clean)` from the Step 7 data, so the noise samples stay the same.
- 20 damaged cases (3–22) for each condition, TMCMC n = 1000 (decision of T1).
- FE-in-the-loop references on 3 damaged cases for each noise scale and design. At scale 1,
  the run uses the Step 7 references again.
- The grid56 posteriors and the FE references run locally. The full-field posteriors run on
  Kaggle (notebook `kaggle/pino_t3.ipynb`, dataset of `scripts/kaggle_bundle_t3.sh`). The
  browser upload limit is 10 MB, so the dataset contains 9 MB parts and a `MANIFEST.json`.
  The notebook joins the parts and checks their SHA-256 values.
- Options: `--surrogate-only` (no FE, for Kaggle), `--fe-only` (only the FE references),
  `--design`, `--shard i/k`, `--deadline-hours`.
- Output: `results/laptop/improve/t3/regime.{json,md,png}`.

First result (KL-trained N = 100, scale 1, grid56): the coverage with and without Σ_G is the
same (mean 0.86). W1 to the FE reference is 0.02–0.03.

### T5: 3D forward model — code ready, runs not started

- `experiments/improve_t5_profile.py`: accuracy and cost of the 3D solver for nz = 2, 4, 6
  and three CG tolerances; parallel throughput and memory for each worker.
- `experiments/improve_t5_dataset.py`: sets A3 (2300) and D3_pit (300). Each 3D solution
  stores the fields of **both faces** (`w_flat`, `w_corroded`). The user selected both faces
  (2026-11-23). The configuration option is `forward.observed_face`.
- `experiments/improve_t5_train.py`: one FNO for each face. Gate: test error less than 5 %
  after H. The script also gives the error of the 2D membrane surrogate on the 3D data.
- `experiments/improve_t5_infer.py`: posteriors with the 3D-face surrogate and with the
  membrane surrogate on synthetic 3D data; an FE reference by importance reweighting with
  the 3D-FE likelihood (ESS given); the timing table.

## Smoke-test results (`quick.yaml`, not conclusions)

The quick run (400 training samples, 400 steps, 4 benchmark cases) runs all stages. All
stages pass. The numbers below show only that the code paths work.

- **Step 1 gate: PASS.** A uniform plate agrees with the closed form within approximately
  10^-11. The load and modulus scaling is exact. A change of the mesh from 2 mm to 1 mm
  changes the pit signal by 0.17 %. One solution takes 0.2 s.
- **Step 3:** a calibrated read noise of σ = 8.0 grey levels gives a scatter of 30 µε at a
  radius of 60 px (target 30, placeholder). The scatter is 65, 30 and 17 µε at 30, 60 and
  90 px, which is the expected trend. ℓ_f = 2.60 mm. Σ_DIC: σ ≈ 28 / 28 / 21 µε
  (εxx / εyy / γxy), ℓ_n ≈ 2.4 mm, nugget ≈ 0. The low-strain bias of von Mises (+12 µε) occurs again.
- **Step 5 (400 steps):** the PINO error on set A is 0.144. The gate (less than 0.10) fails,
  as expected at this budget. FNO 0.150, DeepONet 0.71. The PINO is equal to the FNO at this size.
- **Step 7:** with the 56-point grid, the posteriors stay near the prior for most pits.
  δ ≤ 0.15 of 3.2 mm is a loss of 0.5 mm or less, and its strain signal is near Σ_DIC. The
  full-field design is much more informative.
- **False alarm of the realism check (solved):** data from the full image pipeline on an
  *intact* plate first gave a confident pit (δ = 0.149, P(δ > 0.01) = 1.0). The cause was the
  almost singular full-field noise covariance (see "Nugget floor"). The sampler collapsed
  (14 % unique particles) onto an incorrect pit. A second run used the 0.1 σ floor (same data, same surrogate, with Σ_G). The results were:
  δ median 0.014, 90 % interval [0.001, 0.045], P(δ > 0.01) = 0.62 against a prior of 0.93,
  77 % unique particles, 313 s instead of 2907 s. On the 56-point
  grid, the floor has no effect (0.84 and 0.83). This is the expected result for points that
  are far apart. The quick Step 7 tables are from the time before the floor. The laptop run uses the floor.
- **Positive control** (`experiments/positive_control.py`, one pit with r = 40 mm and
  δ = 0.15 at (150, 125)): with the floor, the full-field posterior is centred within 0.5 mm.
  The 90 % interval is ±2.5 mm, and r = 40.8 [39.0, 43.0]. This result is the same for
  surrogate data and for FE data. The 56-point grid finds the pit with location intervals of approximately ±12 mm.
- **Step 6 NUTS:** the sampler runs. R-hat 1.03, ESS 27–44 at 2 × 100 samples (not accepted at this size).
- **Step 8a (Zenodo plate with a hole):** the check with a meshed hole passes (0.96 % strain L2
  at 2 mm or more from the edge). An FE least-squares fit finds the hole at (39.9, 9.95) mm
  with r = 4.27 mm (truth (40, 10), r = 4). But it gives E = 224–227 GPa (199 GPa with the
  weights of the authors) against the reference of 185 GPa. The data show a left/right
  asymmetry (approximately 520 against 430–460 µε). A symmetric clamped model cannot fit this
  asymmetry. The under-trained quick PINO gives a narrow posterior at an incorrect location
  ((53, 13), E 215 GPa, ν at its lower limit). The surrogate error controls this result.
- **Step 8c:** CalculiX gives 197.2 kN for the intact plate against 196 kN from a hand
  calculation. A deep pit gives 194.1 kN. The posterior samples of one benchmark case give
  196.1–196.9 kN against a truth of 196.8 kN. The net-section 90 % coverage is 1.0.
