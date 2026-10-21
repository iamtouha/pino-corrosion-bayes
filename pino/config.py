"""YAML config loading with layered overrides and a stable hash for provenance."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """dict with attribute access; nested dicts become Config too."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    @property
    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self, sort_keys=True).encode()).hexdigest()[:10]

    def path(self, *parts) -> Path:
        return ROOT.joinpath(*parts)

    @property
    def data_dir(self) -> Path:
        p = ROOT / self["dataset"]["dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def results_dir(self, *parts) -> Path:
        p = ROOT.joinpath("results", self.get("tag", "full"), *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load(paths=None) -> Config:
    paths = list(paths or [])
    if not paths or Path(paths[0]).name != "base.yaml":
        paths = [str(ROOT / "configs" / "base.yaml")] + paths
    cfg: dict = {}
    for p in paths:
        p = Path(p)
        if not p.is_absolute() and not p.exists():
            p = ROOT / "configs" / p
        cfg = _merge(cfg, yaml.safe_load(p.read_text()) or {})
    return Config(cfg)


def cli(description: str = "", extra=None):
    """Standard CLI: --config (repeatable) + script-specific args. Returns (cfg, args)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", action="append", default=[],
                    help="YAML overrides on top of configs/base.yaml (repeatable)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return load(args.config), args


def save_json(obj, path, cfg: Config | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is not None and isinstance(obj, dict):
        obj = {"config_hash": cfg.hash, **obj}
    path.write_text(json.dumps(obj, indent=2, default=float))
