"""Writes kaggle/pino_t3.ipynb: surrogate-only T3 posteriors (follow-up studies, README) on Kaggle 2x T4.
Run: python3 kaggle/make_notebook_t3.py. Input dataset: kaggle/upload_t3 (scripts/kaggle_bundle_t3.sh)."""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)}

cells = [
    md("""
# PINO surrogate — T3 Sigma_G regime study, surrogate-only posteriors

1. Add the dataset built by `scripts/kaggle_bundle_t3.sh` (or its 9 MB parts in kaggle/upload_t3_parts
   with MANIFEST.json; rebuilt and checksummed here) (**Add Input → Your Datasets → pino-t3**).
   To continue an earlier session, also add that notebook version's **output** as an input.
2. Settings: **Accelerator = GPU T4 x2**, Internet **on** (pip install neuraloperator).
3. **Save Version → Save & Run All (Commit)**. Two shards run, one per GPU; no new posterior
   starts after `DEADLINE_H`, and every finished posterior is one file in the output.
4. Download the output and unpack `results/laptop/improve/t3/` into the repo (files are not
   overwritten: `cp -rn`).
"""),
    code("""
CFGS = ["laptop.yaml", "improve.yaml"]
DESIGN = "full"          # the expensive design; grid56 runs locally
DEADLINE_H = 10.5
"""),
    code("""
import glob, os, shutil, subprocess, sys, tarfile, time, tempfile
from pathlib import Path
import torch
n_gpu = torch.cuda.device_count(); assert n_gpu > 0, "set Accelerator to GPU T4 x2"
for i in range(n_gpu):
    cap = torch.cuda.get_device_capability(i)
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        raise SystemExit(f"no kernels for {torch.cuda.get_device_name(i)}; use GPU T4 x2")
cons = Path(tempfile.mkdtemp()) / "constraints.txt"; cons.write_text(f"torch=={torch.__version__}\\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", str(cons), "neuraloperator==1.0.2",
                "torch-harmonics==0.7.3", "tensorly==0.10.0", "tensorly-torch==0.5.0"], check=True)
"""),
    code("""
REPO, WORK, BUNDLE = Path("/tmp/pino-surrogate"), Path("/kaggle/working"), Path("/tmp/bundle")
shutil.rmtree(REPO, ignore_errors=True); REPO.mkdir(parents=True)
# the dataset holds files split into <= 9 MB parts (browser upload limit); rebuild them
import hashlib, json
man_p = sorted(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True))
if man_p:
    src = Path(man_p[0]).parent
    for rel, m in json.loads(Path(man_p[0]).read_text()).items():
        parts = [src / q for q in m["parts"]]
        if not all(q.exists() for q in parts):
            print("missing parts for", rel, "(Kaggle may have unpacked it); skipped"); continue
        out = BUNDLE / rel; out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fo:
            for q in parts: fo.write(q.read_bytes())
        h = hashlib.sha256(out.read_bytes()).hexdigest()
        assert h == m["sha256"], f"checksum mismatch for {rel}"
    print("rebuilt", sum(1 for _ in BUNDLE.rglob("*") if _.is_file()), "files from parts")
def find(name):
    hits = sorted(glob.glob(f"{BUNDLE}/**/{name}", recursive=True)) + \
           sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    assert hits, name; return Path(hits[0])
code_tar = glob.glob(f"{BUNDLE}/**/pino-code.tar.gz", recursive=True) + glob.glob("/kaggle/input/**/pino-code.tar.gz", recursive=True)
if code_tar:
    with tarfile.open(code_tar[0]) as t: t.extractall(REPO)
else:   # Kaggle may unpack archives on upload
    src = find("improve_t3_regime.py").parent.parent
    for d in ("pino", "experiments", "configs"): shutil.copytree(src / d, REPO / d)
(REPO / "data" / "vdic").mkdir(parents=True)
os.symlink(find("D_pit.h5"), REPO / "data" / "D_pit.h5")
for j in ("filter_fit.json", "noise_model.json", "calibration.json"):
    try: shutil.copy(find(j), REPO / "data" / "vdic")
    except AssertionError: assert j == "calibration.json", j
(WORK / "results").mkdir(exist_ok=True); os.symlink(WORK / "results", REPO / "results")
s7 = WORK / "results/laptop/step7/pit"; s7.mkdir(parents=True, exist_ok=True)
os.symlink(find("data.npz"), s7 / "data.npz")
ck = WORK / "results/laptop/step5/ckpt"; ck.mkdir(parents=True, exist_ok=True)
for f in ("pino_A_N100_s0.pt", "pino_A_N2000_s0.pt", "pino_B_N100_s0.pt", "pino_B_N2000_s0.pt"):
    os.symlink(find(f), ck / f)
for t in glob.glob("/kaggle/input/**/t3_prev.tar.gz", recursive=True):
    with tarfile.open(t) as tf: tf.extractall(WORK)
t3 = WORK / "results/laptop/improve/t3"
for d in glob.glob("/kaggle/input/**/results/laptop/improve/t3", recursive=True):
    shutil.copytree(d, t3, dirs_exist_ok=True)
(t3 / "posteriors").mkdir(parents=True, exist_ok=True)
print("finished posteriors available:", len(list((t3 / "posteriors").glob("*.npz"))))
"""),
    code("""
env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
cfg_args = sum([["--config", c] for c in CFGS], [])
(WORK / "logs").mkdir(exist_ok=True)
procs = []
for i in range(n_gpu):
    log = open(WORK / "logs" / f"t3_shard{i}.log", "a")
    cmd = [sys.executable, "experiments/improve_t3_regime.py", *cfg_args, "--stage", "run", "--surrogate-only",
           "--design", DESIGN, "--shard", f"{i}/{n_gpu}", "--deadline-hours", str(DEADLINE_H)]
    procs.append(subprocess.Popen(cmd, cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=str(i)), stdout=log,
                                  stderr=subprocess.STDOUT))
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(600)
    print(f"{(time.time() - t0) / 3600:5.2f} h  posteriors: {len(list((t3 / 'posteriors').glob('*.npz')))}", flush=True)
print("exit codes:", [p.returncode for p in procs])
for f in sorted((WORK / "logs").glob("t3_*.log")):
    print("====", f.name); print("\\n".join(f.read_text().splitlines()[-10:]))
"""),
    code("""
subprocess.run([sys.executable, "experiments/improve_t3_regime.py", *cfg_args, "--stage", "report"], cwd=REPO, env=env)
shutil.rmtree(WORK / "results/laptop/step7"); shutil.rmtree(WORK / "results/laptop/step5")   # input symlinks only
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).with_name("pino_t3.ipynb"); p.write_text(json.dumps(nb, indent=1)); print("wrote", p)
