"""T5 step 4 (follow-up studies, README): FNO surrogates of the 3D face fields (one per observed face).

Data: data/A3.h5 (improve_t5_dataset.py), seeded split train / val / test =
improve.t5.split. Model: the Step-5 FNO (no physics loss), output = in-plane face displacement
perturbation w_face on the operator grid, so the existing observation operator H (strain after
the DIC filter) and the inference code apply unchanged. Gate: test relative L2 of the
perturbation strain after H < 5 %.
Also reported: the error of the 2D membrane surrogate (pino_A_N2000) against the 3D face
fields on the same test set (the model-form error in surrogate terms), and inference time.
Outputs: results/laptop/improve/t5/ckpt/fno_A3_<face>_s0.pt, results/laptop/improve/t5/train.json
"""
import h5py
import numpy as np
import torch

from pino import config
from pino.grid import Grid
from pino.surrogates.models import load_surrogate
from pino.surrogates.train import DEV, evaluate, time_inference, train_run
from pino.vdic.observe import make_obs


def faces_of(cfg):
    f = cfg["forward"]["observed_face"]
    return ["flat", "corroded"] if f == "both" else [f]


def load_face(cfg, name, face):
    with h5py.File(cfg.data_dir / f"{name}.h5", "r") as f:
        done = np.flatnonzero(f["done"][:] == 1)
        tau = torch.from_numpy(f["t_tilde"][done])
        w = torch.from_numpy(f[f"w_{face}"][done])
        nu = torch.from_numpy(f["nu"][done])
    return tau, w, nu


def split(n, sp):
    perm = np.random.default_rng(1234).permutation(n)
    a, b = sp["train"], sp["train"] + sp["val"]
    return perm[:a], perm[a:b], perm[b:b + sp["test"]]


def main():
    cfg, args = config.cli(__doc__)
    tc = cfg["improve"]["t5"]
    grid = Grid.from_cfg(cfg)
    H = make_obs(cfg, grid, "full").to(DEV)
    od = cfg.results_dir("improve", "t5")
    res = {}
    for face in faces_of(cfg):
        tau, w, nu = load_face(cfg, "A3", face)
        itr, iva, ite = split(len(nu), tc["split"])
        sel = lambda i: (tau[i], w[i], nu[i])
        ck = od / "ckpt" / f"fno_A3_{face}_s0.pt"
        if not ck.exists():
            _, info = train_run(cfg, "fno", "A3", len(itr), 0, 0.0, tc["train_steps"], ck,
                                splits={"train": sel(itr), "val": sel(iva)})
            print(f"[{face}] trained: val {info['val_rel_l2']:.4f}, {info['train_seconds'] / 60:.0f} min", flush=True)
        sur = load_surrogate(ck, cfg, grid, DEV)
        te = evaluate(sur, *sel(ite), H)
        mem = load_surrogate(cfg.results_dir("step5") / "ckpt" / "pino_A_N2000_s0.pt", cfg, grid, DEV)
        me = evaluate(mem, *sel(ite), H)
        res[face] = dict(n_train=len(itr), n_val=len(iva), n_test=len(ite), test_rel_l2=te["rel_l2"],
                         test_median_rel_l2=te["median_rel_l2"], gate_lt_5pct=te["rel_l2"] < 0.05,
                         membrane_surrogate_vs_3d_rel_l2=me["rel_l2"],
                         membrane_surrogate_vs_3d_median=me["median_rel_l2"],
                         inference_ms_per_1000=time_inference(sur, tau[ite], nu[ite]))
        print(f"[{face}] test rel L2 after H: {te['rel_l2']:.4f} (gate < 0.05: {te['rel_l2'] < 0.05}); "
              f"2D membrane surrogate vs 3D {face} face: {me['rel_l2']:.3f}", flush=True)
        del sur, mem
        torch.cuda.empty_cache()
    config.save_json(res, od / "train.json", cfg)


if __name__ == "__main__":
    main()
