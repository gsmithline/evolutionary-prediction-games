"""Entry point for the Fig. 5 LLM redo.

Env-var driven (mirrors Opinion-dynamics-post-training/condor/run_one.sh →
pokec_simulations.py pattern):

    POLICY        = static | closed_form | sft | rl_kl
    BETA          = float, KL anchor strength (closed_form, sft, rl_kl)
    SEED          = int
    T             = int, replicator steps (default 600)
    N_PER_STEP    = int (default 1000)
    BASE_MODEL    = HF model id (default Qwen/Qwen2.5-0.5B-Instruct)
    DATA_DIR      = where folktexts caches ACS PUMS
    PI_REF_DIR    = where the per-model π_ref cache lives
    RESULTS_DIR   = where to write the trajectory CSV
    RUN_TAG       = output filename stem
    SFT_EPOCHS    = int (sft only, default 1)
    BATCH_SIZE    = int (sft only, default 4)
    GRAD_ACCUM    = int (sft only, default 1)
    INFERENCE_BATCH_SIZE = int (sft / rl_kl only, default 64) — batch size for
                  per-round LLM scoring (decoupled from training batch size).
    WANDB_PROJECT = str (default "epg-llm-fig5")
    WANDB_ENTITY  = str (optional, defaults to user's wandb default)
    WANDB_MODE    = "online" | "offline" | "disabled"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from .data import K, STATES, load_acs_income_3state
from .llm_pi_ref import build_or_load_pi_ref
from .policies import SamplePack, StaticThresholdPolicy, ClosedFormPolicy, SFTPolicy, RLKLPolicy
from .replicator import run_replicator


# ---------------------------------------------------------------------------
# TRAIN_FILTER: feature-predicate sliver wrapper.
# Filters the population sample to rows matching the predicate before fit().
# Per-state evaluation is unaffected. Used to test whether feature-targeted
# SFT/closed_form training produces emergent state-asymmetric population
# steering through feature-state correlations in the data.
# ---------------------------------------------------------------------------

def _parse_train_filter(spec: str):
    """Parse a simple feature predicate like 'WKHP>40' or 'AGEP>=50'.

    Returns a callable (df) -> bool ndarray of len(df), or None when spec is empty.
    Supported operators (longest-match first): >=, <=, ==, >, <.
    Also treats 'none', '_', '__none__' (case-insensitive) as None, so condor
    configs can pass an explicit no-filter value when needed.
    """
    spec = (spec or "").strip()
    if not spec or spec.lower() in ("none", "_", "__none__"):
        return None
    for op_str, op_fn in [
        (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b),
        ("==", lambda a, b: a == b),
        (">",  lambda a, b: a >  b),
        ("<",  lambda a, b: a <  b),
    ]:
        if op_str in spec:
            col, val = spec.split(op_str, 1)
            col = col.strip()
            try:
                val = float(val.strip())
            except ValueError:
                raise ValueError(f"TRAIN_FILTER value must be numeric; got spec={spec!r}")

            def _f(df, col=col, val=val, op_fn=op_fn):
                arr = df[col].to_numpy()
                return op_fn(arr, val)

            _f._spec = spec
            return _f
    raise ValueError(
        f"TRAIN_FILTER spec {spec!r} did not match any of >=, <=, ==, >, <."
    )


class FilteredPolicy:
    """Wraps a policy so its fit() sees only rows matching a feature predicate.

    score() delegates unchanged so per-state evaluation is uniform across
    conditions. threshold property is passed through for static-style policies.
    """
    def __init__(self, base, filter_fn):
        self.base = base
        self.filter = filter_fn
        self.name = f"{base.name}_filtered"
        self._round = 0

    def fit(self, sample):
        self._round += 1
        if self.filter is None:
            self.base.fit(sample)
            return
        mask = np.asarray(self.filter(sample.X)).astype(bool)
        n_total = int(len(mask))
        n_in = int(mask.sum())
        spec = getattr(self.filter, "_spec", "<filter>")
        pct = (100.0 * n_in / n_total) if n_total else 0.0
        print(f"[filter] round={self._round} spec={spec!r}  kept={n_in}/{n_total} ({pct:.1f}%)")
        if n_in == 0:
            print(f"[filter] WARNING: empty subset; keeping previous policy state, no fit this round")
            return
        filtered = SamplePack(
            X=sample.X.loc[mask].reset_index(drop=True),
            y=sample.y[mask],
            pi_ref_p1=sample.pi_ref_p1[mask],
        )
        self.base.fit(filtered)

    def score(self, test):
        return self.base.score(test)

    # Forward attribute access (e.g. .threshold for StaticThresholdPolicy) to
    # the wrapped base; raise AttributeError when base doesn't have it so the
    # replicator's `hasattr(policy, "threshold")` check works correctly for
    # bases that lack a threshold (e.g. ClosedFormPolicy).
    def __getattr__(self, name):
        if name.startswith("_") or name in ("base", "filter"):
            raise AttributeError(name)
        return getattr(self.base, name)


def _init_wandb(config: dict, run_tag: str):
    """Init a wandb run if WANDB_API_KEY is set and mode != disabled.

    Returns the wandb module on success, or None if wandb is off / unavailable.
    """
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    if not os.environ.get("WANDB_API_KEY"):
        return None
    if wandb is None:
        print("[run_fig5] wandb not installed; skipping wandb logging.", file=sys.stderr)
        return None

    policy = config.get("policy", "")
    tags = [policy]
    if policy in ("sft", "rl_kl"):
        train_mode = "continual" if config.get("sft_continual") else "fresh"
        tags.append(f"{policy}-{train_mode}")
        display_name = f"{run_tag}-{train_mode}"
    else:
        display_name = run_tag

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "epg-llm-fig5"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=display_name,
        tags=tags,
        config=config,
        reinit=True,
    )
    wandb.define_metric("t")
    for state in STATES:
        wandb.define_metric(f"p/{state}", step_metric="t")
        wandb.define_metric(f"acc/{state}", step_metric="t")
        wandb.define_metric(f"fitness/{state}", step_metric="t")
    wandb.define_metric("p/min", step_metric="t")
    wandb.define_metric("p/entropy", step_metric="t")
    wandb.define_metric("acc/overall", step_metric="t")
    wandb.define_metric("step_seconds", step_metric="t")
    return wandb


def main():
    policy_kind = os.environ.get("POLICY", "static")
    beta = float(os.environ.get("BETA", "0.0"))
    seed = int(os.environ.get("SEED", "0"))
    T = int(os.environ.get("T", "600"))
    n_per_step = int(os.environ.get("N_PER_STEP", "1000"))
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    repo_root = Path(__file__).resolve().parent
    data_dir = Path(os.environ.get("DATA_DIR", repo_root / "_data"))
    pi_ref_dir = Path(os.environ.get("PI_REF_DIR", repo_root / "_pi_ref"))
    results_dir = Path(os.environ.get("RESULTS_DIR", repo_root / "_results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    run_tag = os.environ.get("RUN_TAG", f"{policy_kind}_b{beta}_s{seed}")
    # Default = continual (Mendler-Dünner et al. 2020 framing; matches
    # deployed-system semantics). SFT_FRESH=1 flips to RRM (fresh-each-round)
    # for theory comparison.
    sft_continual = os.environ.get("SFT_FRESH", "0") != "1"
    prompt_format = os.environ.get("PROMPT_FORMAT", "flat")
    config = {
        "policy": policy_kind,
        "beta": beta,
        "seed": seed,
        "T": T,
        "n_per_step": n_per_step,
        "base_model": base_model,
        "states": list(STATES),
        "sft_epochs": int(os.environ.get("SFT_EPOCHS", "1")),
        "batch_size": int(os.environ.get("BATCH_SIZE", "4")),
        "grad_accum": int(os.environ.get("GRAD_ACCUM", "1")),
        "optimizer": os.environ.get("OPTIMIZER", "adamw_torch"),
        "sft_continual": sft_continual,
        "inference_batch_size": int(os.environ.get("INFERENCE_BATCH_SIZE", "64")),
        "prompt_format": prompt_format,
    }
    print(f"[run_fig5] tag={run_tag} policy={policy_kind} beta={beta} seed={seed} T={T} n={n_per_step} prompt_format={prompt_format}")
    print(f"[run_fig5] base_model={base_model} data_dir={data_dir} pi_ref_dir={pi_ref_dir}")

    wandb_mod = _init_wandb(config, run_tag)

    max_train = os.environ.get("MAX_TRAIN_PER_STATE")
    n_test = int(os.environ.get("N_TEST_PER_STATE", "5000"))
    data = load_acs_income_3state(
        data_dir, seed=seed,
        n_test_per_state=n_test,
        max_train_per_state=int(max_train) if max_train else None,
    )
    pi_ref = build_or_load_pi_ref(
        data, model_name=base_model, cache_dir=pi_ref_dir,
        batch_size=int(os.environ.get("PI_REF_BATCH_SIZE", "16")),
        prompt_format=prompt_format,
    )

    # Log per-state base-LLM (frozen-π_ref) baseline accuracy at threshold 0.5 to
    # set up the comparison baseline for everyone.
    if wandb_mod is not None:
        baseline = {}
        for state in STATES:
            test_pos = data.test_pos[state]
            p1 = pi_ref[state]["full_p1"][test_pos]
            y = pi_ref[state]["full_y"][test_pos]
            pred = (p1 >= 0.5).astype(np.int64)
            baseline[f"baseline_pi_ref_acc/{state}"] = float((pred == y).mean())
        wandb_mod.log(baseline)

    if policy_kind == "static":
        policy = StaticThresholdPolicy()
    elif policy_kind == "closed_form":
        policy = ClosedFormPolicy(beta=beta)
    elif policy_kind == "sft":
        policy = SFTPolicy(
            base_model=base_model,
            beta=beta,
            folktexts_dataset=data.folktexts_dataset,
            sft_epochs=config["sft_epochs"],
            batch_size=config["batch_size"],
            grad_accum=config["grad_accum"],
            optimizer=config["optimizer"],
            continual=sft_continual,
            inference_batch_size=config["inference_batch_size"],
        )
    elif policy_kind == "rl_kl":
        policy = RLKLPolicy(
            base_model=base_model,
            beta=beta,
            folktexts_dataset=data.folktexts_dataset,
            sft_epochs=config["sft_epochs"],
            batch_size=config["batch_size"],
            grad_accum=config["grad_accum"],
            optimizer=config["optimizer"],
            continual=sft_continual,
            inference_batch_size=config["inference_batch_size"],
        )
    else:
        print(f"[run_fig5] unknown POLICY={policy_kind}", file=sys.stderr)
        sys.exit(2)

    # Optional TRAIN_FILTER wrapper: feature-predicate sliver.
    train_filter_spec = os.environ.get("TRAIN_FILTER", "")
    train_filter = _parse_train_filter(train_filter_spec)
    if train_filter is not None:
        print(f"[run_fig5] TRAIN_FILTER={train_filter_spec!r}  (policy wrapped in FilteredPolicy)")
        policy = FilteredPolicy(policy, train_filter)
        config["train_filter"] = train_filter_spec

    def wandb_step_log(step_record: dict) -> None:
        if wandb_mod is None:
            return
        wandb_mod.log(step_record)

    p0 = np.full(K, 1.0 / K)
    out_path = results_dir / f"{run_tag}.csv"
    traj = run_replicator(
        policy=policy, data=data, pi_ref=pi_ref,
        p0=p0, T=T, n_per_step=n_per_step, seed=seed,
        on_step=wandb_step_log,
        incremental_csv=out_path,
    )
    # incremental_csv has already flushed the final trajectory; this re-write
    # is harmless and keeps the existing log line stable.
    traj.to_csv(out_path, index=False)
    print(f"[run_fig5] wrote {out_path}")

    if wandb_mod is not None:
        _wandb_finalize(wandb_mod, traj, out_path)


def _wandb_finalize(wandb_mod, traj, out_path):
    """End-of-run summary plots + stats matching the three Fig.5 panels."""
    try:
        wandb_mod.save(str(out_path))
    except Exception as e:
        print(f"[run_fig5] wandb.save failed: {e}", file=sys.stderr)

    pivot_p = traj.pivot(index="t", columns="state", values="p")
    pivot_acc = traj.pivot(index="t", columns="state", values="acc")
    pivot_fit = traj.pivot(index="t", columns="state", values="fitness")

    xs = pivot_p.index.to_list()
    keys = list(STATES)
    try:
        wandb_mod.log({
            "summary/composition": wandb_mod.plot.line_series(
                xs=xs,
                ys=[pivot_p[k].fillna(0).to_list() for k in keys],
                keys=list(keys), title="Population composition over time", xname="t",
            ),
            "summary/group_accuracy": wandb_mod.plot.line_series(
                xs=xs,
                ys=[pivot_acc[k].fillna(np.nan).to_list() for k in keys],
                keys=list(keys), title="Group accuracy over time", xname="t",
            ),
            "summary/group_fitness": wandb_mod.plot.line_series(
                xs=xs,
                ys=[pivot_fit[k].fillna(np.nan).to_list() for k in keys],
                keys=list(keys), title="Group fitness over time", xname="t",
            ),
        })
    except Exception as e:
        print(f"[run_fig5] wandb.plot.line_series failed: {e}", file=sys.stderr)

    final_row = pivot_p.iloc[-1]
    p_min_series = pivot_p.min(axis=1)
    below_1pct = (p_min_series < 0.01)
    t_below_1pct = int(below_1pct.idxmax()) if below_1pct.any() else -1
    summary = {
        "summary/p_min_final": float(final_row.min()),
        "summary/p_min_state": str(final_row.idxmin()),
        "summary/t_first_p_below_1pct": t_below_1pct,
        "summary/acc_overall_final": float((pivot_p.iloc[-2] * pivot_acc.iloc[-2]).sum())
        if len(pivot_acc) >= 2 else float("nan"),
    }
    for k in keys:
        summary[f"summary/p_final/{k}"] = float(final_row[k])
    wandb_mod.summary.update(summary)
    wandb_mod.log(summary)
    wandb_mod.finish()


if __name__ == "__main__":
    main()
