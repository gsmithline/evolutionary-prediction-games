"""One-shot caching of base-LLM π_ref scores for ACSIncome rows.

We score every row in the per-state pools once with the base LLM through
folktexts' QA interface. Replicator-time policies (Static and ClosedForm) read
from this cache instead of re-querying the LLM each step.

Cache layout (current, position-indexed):
  cache_dir/
    {model_slug}[__chat]/
      {state}_full_p1.npy   # shape (state_size,), p1 indexed by state-row position
      {state}_full_y.npy    # shape (state_size,), y  indexed by state-row position

The `__chat` suffix marks caches built with `prompt_format="chat"`. Default
`prompt_format="flat"` matches folktexts' default flat completion-style prompt.

Indexing by state-row position (0 .. state_size-1, where state-row position
is the row's location in the unpermuted per-state ACS DataFrame) makes the
cache invariant to N_TEST_PER_STATE / max_train_per_state at run time:
slicing into the cache is done by `data.train_pos[state]` /
`data.test_pos[state]` which encode the partition for the current run.

Legacy layout (kept readable for one-time migration):
  {state}_train_p1.npy / {state}_train_y.npy / {state}_test_p1.npy / {state}_test_y.npy
were ordered by the partition that was active when the cache was built —
specifically the values were laid out as [old_train_rows ++ old_test_rows] in
the order produced by `_state_pool_concat`. Migration reverses that using the
deterministic PARTITION_SEED + per-state size to recover the position-indexed
arrays without re-running the LLM.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from .data import STATES, ACSIncome3State, PARTITION_SEED
from . import _folktexts_compat  # noqa: F401  — patches folktexts vocab-size bug
from folktexts.llm_utils import load_model_tokenizer
from folktexts.classifier import TransformersLLMClassifier
from folktexts.prompting import encode_row_prompt_chat


def _slug(model_name: str) -> str:
    return model_name.replace("/", "--")


def cache_path(cache_dir: str | Path, model_name: str, prompt_format: str = "flat") -> Path:
    suffix = "__chat" if prompt_format == "chat" else ""
    return Path(cache_dir) / (_slug(model_name) + suffix)


def _state_pool_concat(data: ACSIncome3State, state: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (X_concat, y_concat, pos_concat) covering the full state.

    Rows are train rows first (in train DataFrame order) then test rows (in
    test DataFrame order). `pos_concat[i]` gives the state-row position of
    each row, used to write the LLM-scored p1 back into a position-indexed
    full-state array.
    """
    train = data.train[state]
    test = data.test[state]
    df = pd.concat([train[data.feature_cols], test[data.feature_cols]], ignore_index=True)
    y = np.concatenate([
        train[data.target_col].astype(int).values,
        test[data.target_col].astype(int).values,
    ])
    pos = np.concatenate([data.train_pos[state], data.test_pos[state]])
    return df, y, pos


def _migrate_legacy_cache(out_dir: Path) -> bool:
    """If a legacy split cache exists, rewrite it as the position-indexed format.

    Reverses the OLD partition (PARTITION_SEED + len(sub) inferred from cache
    sizes) and lays the cached values into one full-state array per state.
    Writes new files alongside the legacy ones. Returns True on success.

    Assumes the legacy cache was built with `max_train_per_state=None`
    (i.e. n_old_test + n_old_train == state_size). If sizes don't add up
    consistently across states, returns False and the caller falls back to a
    fresh LLM build.
    """
    legacy_kinds = ("train_p1", "train_y", "test_p1", "test_y")
    have_legacy = all(
        (out_dir / f"{state}_{kind}.npy").exists()
        for state in STATES for kind in legacy_kinds
    )
    if not have_legacy:
        return False

    rng = np.random.default_rng(PARTITION_SEED)
    for state in STATES:
        train_p1 = np.load(out_dir / f"{state}_train_p1.npy")
        train_y = np.load(out_dir / f"{state}_train_y.npy")
        test_p1 = np.load(out_dir / f"{state}_test_p1.npy")
        test_y = np.load(out_dir / f"{state}_test_y.npy")
        n_old_test = len(test_p1)
        n_old_train = len(train_p1)
        state_size = n_old_test + n_old_train

        # Reproduce the OLD partition's permutation.
        idx_old = rng.permutation(state_size)

        full_p1 = np.empty(state_size, dtype=np.float64)
        full_y = np.empty(state_size, dtype=np.int64)
        full_p1[idx_old[:n_old_test]] = test_p1
        full_p1[idx_old[n_old_test:n_old_test + n_old_train]] = train_p1
        full_y[idx_old[:n_old_test]] = test_y.astype(np.int64)
        full_y[idx_old[n_old_test:n_old_test + n_old_train]] = train_y.astype(np.int64)

        np.save(out_dir / f"{state}_full_p1.npy", full_p1)
        np.save(out_dir / f"{state}_full_y.npy", full_y)
        print(f"[pi_ref] migrated legacy cache for {state}: state_size={state_size} "
              f"(n_test_old={n_old_test}, n_train_old={n_old_train})", file=sys.stderr)
    return True


def build_or_load_pi_ref(
    data: ACSIncome3State,
    model_name: str,
    cache_dir: str | Path,
    batch_size: int = 16,
    context_size: int | None = None,
    prompt_format: str = "flat",
) -> dict:
    """Compute / load full-state-indexed π_ref(y=1|x) per state, with caching.

    Returns:
      pi_ref[state]["full_p1"] : np.ndarray (state_size,) float64,
                                 indexed by state-row position
      pi_ref[state]["full_y"]  : np.ndarray (state_size,) int64

    Slicing for the current run's train/test partition is done at consumer
    sites via `data.train_pos[state]` and `data.test_pos[state]`.
    """
    if prompt_format not in ("flat", "chat"):
        raise ValueError(f"prompt_format must be 'flat' or 'chat', got {prompt_format!r}")
    out_dir = cache_path(cache_dir, model_name, prompt_format=prompt_format)
    out_dir.mkdir(parents=True, exist_ok=True)

    have_full = all(
        (out_dir / f"{state}_full_p1.npy").exists()
        and (out_dir / f"{state}_full_y.npy").exists()
        for state in STATES
    )
    if not have_full:
        # Try one-time migration from the legacy split cache before invoking the LLM.
        have_full = _migrate_legacy_cache(out_dir)

    pi_ref: dict = {}
    if have_full:
        for state in STATES:
            full_p1 = np.load(out_dir / f"{state}_full_p1.npy")
            full_y = np.load(out_dir / f"{state}_full_y.npy")
            if len(full_p1) != data.state_size[state]:
                raise RuntimeError(
                    f"Cache size mismatch for {state}: cache has {len(full_p1)} entries "
                    f"but data has state_size={data.state_size[state]}. "
                    f"Delete the cache dir to force rebuild."
                )
            pi_ref[state] = {"full_p1": full_p1, "full_y": full_y.astype(np.int64)}
        return pi_ref

    # Fresh build: score the full state pool once.
    model, tok = load_model_tokenizer(model_name)
    encode_row = None
    if prompt_format == "chat":
        encode_row = partial(
            encode_row_prompt_chat,
            task=data.folktexts_dataset.task,
            tokenizer=tok,
        )
    clf = TransformersLLMClassifier(
        model=model,
        tokenizer=tok,
        task=data.folktexts_dataset.task,
        encode_row=encode_row,
        batch_size=batch_size,
        correct_order_bias=False,
    )

    for state in STATES:
        df_concat, y_concat, pos_concat = _state_pool_concat(data, state)
        risk = clf.predict_proba(df_concat)  # (n, 2): [P(neg), P(pos)]
        p1 = risk[:, 1].astype(np.float64)

        full_p1 = np.empty(data.state_size[state], dtype=np.float64)
        full_y = np.empty(data.state_size[state], dtype=np.int64)
        full_p1[pos_concat] = p1
        full_y[pos_concat] = y_concat.astype(np.int64)

        np.save(out_dir / f"{state}_full_p1.npy", full_p1)
        np.save(out_dir / f"{state}_full_y.npy", full_y)
        pi_ref[state] = {"full_p1": full_p1, "full_y": full_y}

    return pi_ref
