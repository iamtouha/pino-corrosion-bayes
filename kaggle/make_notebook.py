"""Writes kaggle/pino_step5.ipynb (run: python3 kaggle/make_notebook.py)."""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)}

cells = [
    md("""
# PINO surrogate — Step 5 (data-efficiency study) on Kaggle

1. Add the dataset built by `scripts/kaggle_bundle.sh` (**Add Input → Your Datasets → pino-step5**).
   To continue an earlier session, also add that notebook version's **output** as an input.
2. Settings: **Accelerator = GPU T4 x2** (one family per GPU), Internet **on** (pip install).
3. **Save Version → Save & Run All (Commit)**. It runs headless (up to 12 h); you can close the
   browser. The run stops starting new trainings ~1 h before the limit (`DEADLINE_H`).
4. Download the version's output and run `./scripts/kaggle_import.sh <downloaded zip>` locally.
"""),
    code("""
CFG = "laptop.yaml"     # must match the config the datasets were built with
DEADLINE_H = 10.75      # stop starting new runs after this (Kaggle limit: 12 h per session)
# first-run time estimates (minutes per run on a T4); replaced by measured times as runs finish
EST_MINUTES = {"deeponet": 15, "fno": 60, "pino": 120}
"""),
    code("""
import glob, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
import torch
print(torch.__version__, "CUDA", torch.version.cuda)
n_gpu = torch.cuda.device_count()
assert n_gpu > 0, "no GPU: set Accelerator to GPU T4 x2"
for i in range(n_gpu):
    cap = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    print(i, name, cap)
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        raise SystemExit(f"This PyTorch build has no kernels for {name} (sm_{cap[0]}{cap[1]}); "
                         "switch the accelerator to GPU T4 x2.")
x = torch.randn(64, 64, device="cuda"); torch.fft.rfft2(x); print("GPU OK")
"""),
    code("""
# neuraloperator does not pin torch; the constraint keeps Kaggle's CUDA build of torch.
import tempfile
cons = Path(tempfile.mkdtemp()) / "constraints.txt"
cons.write_text(f"torch=={torch.__version__}\\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", str(cons),
                "neuraloperator==1.0.2", "torch-harmonics==0.7.3", "tensorly==0.10.0",
                "tensorly-torch==0.5.0"], check=True)
import importlib; import neuralop; print("neuralop", neuralop.__version__, "torch", torch.__version__)
"""),
    code("""
REPO, WORK = Path("/tmp/pino-surrogate"), Path("/kaggle/working")
shutil.rmtree(REPO, ignore_errors=True); REPO.mkdir(parents=True)
# Kaggle unpacks .tar.gz files on upload, so the code is either the archive or a folder
# holding pino/ and experiments/step5_study.py.
code_tar = glob.glob("/kaggle/input/**/pino-code.tar.gz", recursive=True)
code_dir = [Path(p).parent.parent for p in
            glob.glob("/kaggle/input/**/experiments/step5_study.py", recursive=True)
            if (Path(p).parent.parent / "pino" / "__init__.py").exists()]
if code_tar:
    with tarfile.open(code_tar[0]) as t:
        t.extractall(REPO)
elif code_dir:
    for d in ("pino", "experiments", "configs"):
        shutil.copytree(code_dir[0] / d, REPO / d)
else:
    subprocess.run("find /kaggle/input -maxdepth 5 | head -50", shell=True)
    raise SystemExit("code not found: attach the pino-step5 dataset (Add Input); "
                     "the listing above shows what is mounted under /kaggle/input")
print("code from", code_tar[0] if code_tar else code_dir[0])
import yaml
DATA = "data"
for f in ("base.yaml", CFG):
    DATA = ((yaml.safe_load(open(REPO / "configs" / f)) or {}).get("dataset") or {}).get("dir", DATA)
(REPO / DATA / "vdic").mkdir(parents=True)
def find(name):  # web uploads may flatten or nest folders: locate files by name
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    assert hits, f"{name} not found under /kaggle/input"
    return Path(hits[0])
for s in "ABCU":
    os.symlink(find(f"{s}.h5"), REPO / DATA / f"{s}.h5")
for j in ("filter_fit.json", "calibration.json", "noise_model.json"):
    try:
        shutil.copy(find(j), REPO / DATA / "vdic")
    except AssertionError:
        assert j != "filter_fit.json", "filter_fit.json is required"
# results live in /kaggle/working so they are saved with the notebook version
(WORK / "results").mkdir(exist_ok=True); os.symlink(WORK / "results", REPO / "results")
(WORK / "logs").mkdir(exist_ok=True)
tag = CFG.replace(".yaml", "")
step5 = WORK / "results" / tag / "step5"
# resume: earlier results shipped in the dataset and/or attached outputs of earlier versions
for t in glob.glob("/kaggle/input/**/step5_prev.tar.gz", recursive=True):
    with tarfile.open(t) as tf:
        tf.extractall(WORK)          # -> /kaggle/working/results/<tag>/step5
for d in glob.glob(f"/kaggle/input/**/results/{tag}/step5", recursive=True):
    shutil.copytree(d, step5, dirs_exist_ok=True)
n_done = len(list((step5 / "runs").glob("*.json"))) if step5.exists() else 0
print("finished runs available:", n_done)
"""),
    code("""
env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1",
           MPLBACKEND="Agg")
est = [f"{k}={v}" for k, v in EST_MINUTES.items()]
base = [sys.executable, "experiments/step5_study.py", "--config", CFG,
        "--deadline-hours", str(DEADLINE_H), "--est-minutes", *est]
shards = [["A"], ["B"]] if n_gpu >= 2 else [["A", "B"]]
procs = []
for i, fams in enumerate(shards):
    log = open(WORK / "logs" / f"step5_{'_'.join(fams)}.log", "a")
    e = dict(env, CUDA_VISIBLE_DEVICES=str(i % n_gpu))
    procs.append(subprocess.Popen(base + ["--families", *fams], cwd=REPO, env=e,
                                  stdout=log, stderr=subprocess.STDOUT))
    print("started families", fams, "on GPU", i % n_gpu)
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(600)
    done = len(list((step5 / "runs").glob("*.json")))
    print(f"{(time.time() - t0) / 3600:5.2f} h  finished runs: {done}", flush=True)
    for f in sorted((WORK / "logs").glob("step5_*.log")):
        lines = [l for l in f.read_text().splitlines() if "arn" not in l][-2:]
        print("   ", f.name, "|", " || ".join(l[:160] for l in lines))
print("exit codes:", [p.returncode for p in procs])
"""),
    code("""
subprocess.run([sys.executable, "experiments/step5_study.py", "--config", CFG, "--summary-only"],
               cwd=REPO, env=env)
runs = sorted(p.stem for p in (step5 / "runs").glob("*.json"))
print(len(runs), "finished runs:", runs)
for f in sorted((WORK / "logs").glob("step5_*.log")):
    print("====", f.name); print("\\n".join(f.read_text().splitlines()[-15:]))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).with_name("pino_step5.ipynb")
p.write_text(json.dumps(nb, indent=1))
print("wrote", p)
