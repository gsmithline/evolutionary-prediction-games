"""ACSIncome loader for the K=3 state replicator experiment.

Mirrors S&R §6.3: 2018 ACS PUMS, K=3 = (CA, NY, TX), held-out 5k per state for
group-accuracy evaluation.

Folktexts strips the `ST` (state) column when building the per-task DataFrame
since ACSIncome doesn't use state as a feature, so we recover it from the
underlying full ACS frame via index alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from folktexts.acs import ACSDataset


STATES = ("CA", "NY", "TX")
STATE_FIPS = {"CA": 6, "NY": 36, "TX": 48}
K = len(STATES)
DEFAULT_HELDOUT_PER_STATE = 5000
DEFAULT_YEAR = "2018"

# Fixed seed for the train/test partition so the π_ref cache (which is keyed
# only by model name) is valid across all experiment seeds. Replicator-level
# randomness comes from `run_replicator(seed=...)`; the `seed` arg below is
# kept for API compatibility but is intentionally unused for partitioning.
PARTITION_SEED = 42


@dataclass
class ACSIncome3State:
    """ACSIncome split by state with held-out test pools.

    train[state] / test[state] are pd.DataFrames containing both the task
    features and the binary target column (target column name in `target_col`).
    """
    train: dict
    test: dict
    target_col: str
    feature_cols: list
    folktexts_dataset: object  # folktexts.acs.ACSDataset, kept for LLM scoring API


def load_acs_income_3state(
    data_dir: str | Path,
    n_test_per_state: int = DEFAULT_HELDOUT_PER_STATE,
    year: str = DEFAULT_YEAR,
    seed: int = 0,  # noqa: ARG001  — kept for API compat; partition uses PARTITION_SEED
    max_train_per_state: int | None = None,
) -> ACSIncome3State:
    """Load ACSIncome via folktexts and partition by state.

    folktexts goes through folktables to download/cache ACS PUMS into
    `data_dir`/folktables. We then attach the state code from the full ACS
    frame and partition into per-state train/test pools.

    The partition is deterministic across replicator seeds: we use
    `PARTITION_SEED` (42) for both the folktexts internal RNG and our own
    `rng.permutation(len(sub))` calls. This keeps the π_ref cache valid for
    every experiment seed.
    """
    rng = np.random.default_rng(PARTITION_SEED)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    ds = ACSDataset.make_from_task(
        task="ACSIncome",
        cache_dir=str(data_dir),
        survey_year=year,
        seed=PARTITION_SEED,
    )

    parsed = ds.data
    full = ds._full_acs_data  # noqa: SLF001 — needed to recover ST after preprocessing
    if "ST" not in full.columns:
        raise RuntimeError("ST column missing from full ACS frame.")

    # Align ST onto parsed via index (parsed is a row-subset of full after
    # folktables preprocessing).
    state_codes = full.loc[parsed.index, "ST"].astype(int).values
    target_col = ds.task.get_target()
    feature_cols = list(ds.task.features)

    train: dict = {}
    test: dict = {}
    for state in STATES:
        mask = state_codes == STATE_FIPS[state]
        sub = parsed.loc[mask].reset_index(drop=True)
        if len(sub) <= n_test_per_state:
            raise RuntimeError(f"Not enough rows for {state}: {len(sub)}")
        idx = rng.permutation(len(sub))
        test[state] = sub.iloc[idx[:n_test_per_state]].reset_index(drop=True)
        train_idx = idx[n_test_per_state:]
        if max_train_per_state is not None:
            train_idx = train_idx[:max_train_per_state]
        train[state] = sub.iloc[train_idx].reset_index(drop=True)

    return ACSIncome3State(
        train=train,
        test=test,
        target_col=target_col,
        feature_cols=feature_cols,
        folktexts_dataset=ds,
    )


def sample_from_mixture(
    data: ACSIncome3State,
    p: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Draw n rows from D_p = sum_k p_k D_k.

    Returns (rows_df, group_labels, within_state_idx) where:
      - rows_df : concatenated rows from per-state train pools, columns = task
        features ∪ {target_col, _state, _state_idx}
      - group_labels : (n,) int in [0, K) matching STATES order
      - within_state_idx : (n,) int row index into data.train[state]
        (used so policies / π_ref caches can be keyed by example identity)
    """
    counts = rng.multinomial(n, p)
    parts, labels, idx_back = [], [], []
    for k, state in enumerate(STATES):
        m = int(counts[k])
        if m == 0:
            continue
        pool = data.train[state]
        idx = rng.integers(0, len(pool), size=m)
        rows = pool.iloc[idx].copy()
        rows["_state"] = state
        rows["_state_idx"] = idx
        parts.append(rows)
        labels.append(np.full(m, k, dtype=np.int64))
        idx_back.append(idx)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    g = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)
    wsi = np.concatenate(idx_back) if idx_back else np.zeros(0, dtype=np.int64)
    return df, g, wsi
