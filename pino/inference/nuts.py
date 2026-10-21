"""NUTS with Pyro (Step 6.4): gradients of log p flow through the surrogate by autodiff."""
from __future__ import annotations

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
from pyro.ops.stats import effective_sample_size, gelman_rubin

from .problem import Family, Observer


def run_nuts(fam: Family, obs: Observer, lik, mu, d_obs, num_samples=1000, warmup_steps=500,
             chains=4, target_accept=0.9, seed=0, max_tree_depth=10):
    d = d_obs.to(mu.device, torch.float64)
    lo = torch.tensor(np.where(fam.gauss, 0, fam.lo), dtype=torch.float64)
    hi = torch.tensor(np.where(fam.gauss, 1, fam.hi), dtype=torch.float64)

    def model():
        vals = []
        for i, name in enumerate(fam.names):
            if fam.gauss[i]:
                v = pyro.sample(name, dist.Normal(torch.tensor(0.0, dtype=torch.float64),
                                                  torch.tensor(1.0, dtype=torch.float64)))
            else:
                v = pyro.sample(name, dist.Uniform(lo[i], hi[i]))
            vals.append(v)
        x = torch.stack(vals)[None]
        pred = obs(x).to(d.device)
        pyro.factor("loglik", lik(d[None] - pred - mu[None])[0].cpu())   # sites live on CPU

    samples = []
    for c in range(chains):                                  # one after another on the GPU
        pyro.set_rng_seed(seed + c)
        pyro.clear_param_store()
        mcmc = MCMC(NUTS(model, target_accept_prob=target_accept,
                         max_tree_depth=max_tree_depth), num_samples=num_samples,
                    warmup_steps=warmup_steps, disable_progbar=True)
        mcmc.run()
        s = mcmc.get_samples()
        samples.append(torch.stack([s[n] for n in fam.names], -1))
    S = torch.stack(samples)                                 # (chains, n, d)
    rhat = gelman_rubin(S, chain_dim=0, sample_dim=1)
    ess = effective_sample_size(S, chain_dim=0, sample_dim=1)
    diag = {"rhat_max": float(rhat.max()), "ess_min": float(ess.min()),
            "accepted": bool(rhat.max() < 1.01 and ess.min() > 400)}
    return S.reshape(-1, fam.dim).numpy(), diag
