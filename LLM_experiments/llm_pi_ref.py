"""One-shot caching of base-LLM π_ref scores for ACSIncome rows.

We score every train+test row in the per-state pools once with the base LLM
through folktexts' QA interface. Replicator-time policies (Static and
ClosedForm) read from this cache instead of re-querying the LLM each step.

Cache layout:
  cache_dir/
    {model_slug}[__chat]/
      {state}_train_p1.npy  # P(y=1 | x) under base LLM, shape (n_train,)
      {state}_train_y.npy   # ground-truth binary label, shape (n_train,)
      {state}_test_p1.npy
      {state}_test_y.npy

The `__chat` suffix marks caches built with `prompt_format="chat"` (uses the
model's chat template via folktexts.encode_row_prompt_chat). The default
`prompt_format="flat"` matches folktexts' default flat completion-style prompt.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd

from .data import STATES, ACSIncome3State
from . import _folktexts_compat  # noqa: F401  — patches folktexts vocab-size bug
from folktexts.llm_utils import load_model_tokenizer
from folktexts.classifier import TransformersLLMClassifier
from folktexts.prompting import encode_row_prompt_chat


def _slug(model_name: str) -> str:
    return model_name.replace("/", "--")


def cache_path(cache_dir: str | Path, model_name: str, prompt_format: str = "flat") -> Path:
    suffix = "__chat" if prompt_format == "chat" else ""
    return Path(cache_dir) / (_slug(model_name) + suffix)


def _state_pool_concat(data: ACSIncome3State, state: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X_concat, y_concat) covering both train and test pool for a state.

    Train rows are scored first (positions 0 .. n_train-1), then test rows. We
    return per-pool numpy arrays separately on the loader side; this helper
    just concatenates for one batched LLM pass.
    """
    train = data.train[state]
    test = data.test[state]
    df = pd.concat([train[data.feature_cols], test[data.feature_cols]], ignore_index=True)
    y = np.concatenate([
        train[data.target_col].astype(int).values,
        test[data.target_col].astype(int).values,
    ])
    return df, y


def build_or_load_pi_ref(
    data: ACSIncome3State,
    model_name: str,
    cache_dir: str | Path,
    batch_size: int = 16,
    context_size: int | None = None,
    prompt_format: str = "flat",
) -> dict:
    """Compute π_ref(y=1|x) for every train+test row in every state, with caching.

    `prompt_format` is "flat" (folktexts default completion-style) or "chat"
    (wraps each row in the model's chat template via encode_row_prompt_chat).
    The cache for each format is kept in a separate dir so both can coexist.

    Returns a nested dict:
      pi_ref[state]["train_p1"]  : np.ndarray (n_train,)
      pi_ref[state]["train_y"]   : np.ndarray (n_train,)
      pi_ref[state]["test_p1"]   : np.ndarray (n_test,)
      pi_ref[state]["test_y"]    : np.ndarray (n_test,)
    """
    if prompt_format not in ("flat", "chat"):
        raise ValueError(f"prompt_format must be 'flat' or 'chat', got {prompt_format!r}")
    out_dir = cache_path(cache_dir, model_name, prompt_format=prompt_format)
    out_dir.mkdir(parents=True, exist_ok=True)

    pi_ref: dict = {}
    needs_model = any(
        not (out_dir / f"{state}_train_p1.npy").exists()
        or not (out_dir / f"{state}_test_p1.npy").exists()
        for state in STATES
    )

    clf = None
    if needs_model:

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
            correct_order_bias=False, #make true to use the order correcting bias
        )

    for state in STATES:
        train_p1_path = out_dir / f"{state}_train_p1.npy"
        train_y_path = out_dir / f"{state}_train_y.npy"
        test_p1_path = out_dir / f"{state}_test_p1.npy"
        test_y_path = out_dir / f"{state}_test_y.npy"

        if all(p.exists() for p in (train_p1_path, train_y_path, test_p1_path, test_y_path)):
            pi_ref[state] = {
                "train_p1": np.load(train_p1_path),
                "train_y": np.load(train_y_path),
                "test_p1": np.load(test_p1_path),
                "test_y": np.load(test_y_path),
            }
            continue

        df_concat, y_concat = _state_pool_concat(data, state)
        n_train = len(data.train[state])

        risk = clf.predict_proba(df_concat)  # (n, 2): [P(neg), P(pos)]
        p1 = risk[:, 1].astype(np.float64)

        np.save(train_p1_path, p1[:n_train])
        np.save(train_y_path, y_concat[:n_train].astype(np.int64))
        np.save(test_p1_path, p1[n_train:])
        np.save(test_y_path, y_concat[n_train:].astype(np.int64))

        pi_ref[state] = {
            "train_p1": p1[:n_train],
            "train_y": y_concat[:n_train].astype(np.int64),
            "test_p1": p1[n_train:],
            "test_y": y_concat[n_train:].astype(np.int64),
        }

    return pi_ref
