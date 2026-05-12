"""EGTA-style decoupled-performative-risk tensor (simulator-agnostic core).

For K interventions (e.g. different beta, alpha_0, pi_ref) producing trajectories
(pi_t^{(i)}, sigma_t^{(i)}) for i = 1..K, define at each time t the matrix

    M_t[i, j] = E_{z ~ D(pi_t^{(i)}, sigma_t^{(i)})} [L(pi_t^{(j)}; z)]

Diagonal entries are ordinary performative risk (model j == i, evaluated in its
own induced world). Off-diagonals decouple distribution induction from model
evaluation. Stacking over t gives M in R^{T x K x K}.

There are two ways to populate M:

(A) Bucket-factored (cheap, optional optimization).
    When the data space admits a partition into B buckets such that per-bucket
    loss l_t[j, b] is independent of *which world induced model j*, we get

        M_t[i, j] = sum_b w_t[i, b] * l_t[j, b]

    and off-diagonals come free from per-intervention diagonal data. EPG
    satisfies this because per-state holdouts are exogenous. Use EGTAInputs +
    build_bundle().

(B) Direct (compute M by cross-evaluation, no factorization).
    When buckets don't factor cleanly (e.g. FJ, where agent identities are
    static and the world is the *opinion distribution itself*), compute M
    directly via K^2 forward passes per snapshot time: load policy j,
    evaluate on world i's data sample. Then use EGTAInputs.from_matrix() +
    build_bundle_from_matrix(), or just the latter.

Both paths produce identical bundle schemas downstream; equilibrium.py,
plot_egta_matrix.py, and bifurcation_plot.py treat them the same.

Adapters
--------
  * egta_epg.py -- ACSIncome EPG trajectory CSVs (path A).
  * egta_fj.py  -- Friedkin-Johnsen opinion dynamics (path B).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class EGTAInputs:
    """Simulator-agnostic inputs to the EGTA decoupled-PR tensor.

    Two construction modes:
      * bucket-factored: pass world_mixtures + bucket_losses, leave precomputed_M = None.
      * direct: use the from_matrix() classmethod with a precomputed (T, K, K) M.
    """
    world_mixtures: np.ndarray   # (K, T, B), each (i, t, :) a probability vector on buckets
    bucket_losses: np.ndarray    # (K, T, B), in [0, 1] for accuracy-style losses
    t: np.ndarray                # (T,)
    labels: Sequence[str]        # (K,)
    buckets: Sequence[str]       # (B,)
    metadata: dict = field(default_factory=dict)
    precomputed_M: np.ndarray | None = None  # (T, K, K) -- set when bucket factorization is bypassed

    def __post_init__(self):
        if self.world_mixtures.shape != self.bucket_losses.shape:
            raise ValueError(
                f"world_mixtures {self.world_mixtures.shape} != bucket_losses {self.bucket_losses.shape}"
            )
        K, T, B = self.world_mixtures.shape
        if len(self.labels) != K:
            raise ValueError(f"labels length {len(self.labels)} != K={K}")
        if len(self.buckets) != B:
            raise ValueError(f"buckets length {len(self.buckets)} != B={B}")
        if len(self.t) != T:
            raise ValueError(f"t length {len(self.t)} != T={T}")
        if self.precomputed_M is not None:
            if self.precomputed_M.shape != (T, K, K):
                raise ValueError(
                    f"precomputed_M {self.precomputed_M.shape} != (T, K, K) = ({T}, {K}, {K})"
                )

    @property
    def K(self) -> int: return self.world_mixtures.shape[0]
    @property
    def T(self) -> int: return self.world_mixtures.shape[1]
    @property
    def B(self) -> int: return self.world_mixtures.shape[2]

    @classmethod
    def from_matrix(cls, M: np.ndarray, *, t: np.ndarray, labels: Sequence[str],
                    metadata: dict | None = None) -> "EGTAInputs":
        """Build an EGTAInputs from a precomputed (T, K, K) tensor (path B).

        Bucket factorization is bypassed; world_mixtures and bucket_losses are
        zero-width placeholders (B=0) and `precomputed_M` is set.
        """
        M = np.asarray(M)
        if M.ndim != 3 or M.shape[1] != M.shape[2]:
            raise ValueError(f"M must be (T, K, K), got {M.shape}")
        T, K, _ = M.shape
        if len(labels) != K:
            raise ValueError(f"labels length {len(labels)} != K={K}")
        t_arr = np.asarray(t)
        if len(t_arr) != T:
            raise ValueError(f"t length {len(t_arr)} != T={T}")
        return cls(
            world_mixtures=np.zeros((K, T, 0), dtype=np.float64),
            bucket_losses=np.zeros((K, T, 0), dtype=np.float64),
            t=t_arr,
            labels=list(labels),
            buckets=[],
            metadata=dict(metadata or {}),
            precomputed_M=M.astype(np.float64, copy=False),
        )


def build_tensor(world_mixtures: np.ndarray, bucket_losses: np.ndarray) -> np.ndarray:
    """M[t, i, j] = sum_b world_mixtures[i, t, b] * bucket_losses[j, t, b]."""
    if world_mixtures.shape != bucket_losses.shape:
        raise ValueError(
            f"shape mismatch: world_mixtures {world_mixtures.shape} vs bucket_losses {bucket_losses.shape}"
        )
    return np.einsum("itb,jtb->tij", world_mixtures, bucket_losses)


def drop_nan_t(inputs: EGTAInputs, *, verbose: bool = True) -> EGTAInputs:
    """Return new inputs with any t where any (i, b) is NaN removed."""
    nan_mask = (np.isnan(inputs.world_mixtures).any(axis=(0, 2))
                | np.isnan(inputs.bucket_losses).any(axis=(0, 2)))
    good = ~nan_mask
    dropped = int(nan_mask.sum())
    if dropped and verbose:
        print(f"[egta_matrix] dropping {dropped} t-step(s) with NaN: t={list(inputs.t[nan_mask])}")
    return EGTAInputs(
        world_mixtures=inputs.world_mixtures[:, good, :],
        bucket_losses=inputs.bucket_losses[:, good, :],
        t=inputs.t[good],
        labels=list(inputs.labels),
        buckets=list(inputs.buckets),
        metadata=dict(inputs.metadata),
    )


def build_bundle(inputs: EGTAInputs) -> dict:
    """Return the savez-ready bundle. Uses precomputed_M if present, else einsum."""
    if inputs.precomputed_M is not None:
        M = inputs.precomputed_M
    else:
        M = build_tensor(inputs.world_mixtures, inputs.bucket_losses)
    return {
        "M": M,
        "t": np.asarray(inputs.t),
        "labels": np.asarray(list(inputs.labels)),
        "buckets": np.asarray(list(inputs.buckets)),
        "world_mixtures": inputs.world_mixtures,
        "bucket_losses": inputs.bucket_losses,
    }


def build_bundle_from_matrix(M: np.ndarray, *, t: np.ndarray, labels: Sequence[str],
                             metadata: dict | None = None) -> dict:
    """Direct-path bundle builder for simulators without bucket factorization (path B)."""
    inputs = EGTAInputs.from_matrix(M, t=t, labels=labels, metadata=metadata)
    return build_bundle(inputs)


def save_npz(bundle: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **bundle)


def print_summary(bundle: dict) -> None:
    M = bundle["M"]
    diag = np.einsum("tii->ti", M)
    K = M.shape[1]
    if K > 1:
        off = (M.sum(axis=(1, 2)) - diag.sum(axis=1)) / (K * (K - 1))
        off_str = f"mean off-diag: [{off.min():.4f}, {off.max():.4f}]"
    else:
        off_str = "mean off-diag: n/a (K=1)"
    print(f"[egta_matrix] tensor shape M: {M.shape}, labels: {list(bundle['labels'])}")
    print(f"[egta_matrix] diag range over t: [{diag.min():.4f}, {diag.max():.4f}]; {off_str}")
