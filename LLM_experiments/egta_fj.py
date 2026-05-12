"""FJ (Friedkin-Johnsen opinion dynamics) adapter for the agnostic EGTA core.

Path B (direct cross-evaluation). Buckets are not used in FJ -- the "world" is
the population-level opinion vector x^{i,t} in R^n_labeled, paired with the
static per-agent prompts. M is computed directly:

    M_{t_s}[i, j] = (1/n) sum_b NLL_{pi_j^{t_s}}( f"{x_b^{i,t_s}:.2f}"  |  prompt_b )

Cost: K^2 * S * n forward passes. With batch 64 on Qwen-0.5B + LoRA, K=5, S=5,
n=2163: a few thousand forward passes, ~5-10 GPU-minutes total.

Required FJ run artifacts (produced by llm_predictor.py with the env vars set):
    SAVE_ADAPTER=1
    SAVE_ADAPTER_PER_ROUND=1
    SAVE_ADAPTER_EVERY_N_ROUNDS=<your snapshot stride>
    SAVE_PROMPTS=1

Per run, you should then have:
    ./adapters/{tag}/round_{t}/                 (LoRA adapters at chosen t)
    ./adapters/{tag}/prompts_labeled.json       (n_labeled prompts)
    ./adapters/{tag}/prompts_unlabeled.json     (n_unlabeled prompts; written when
                                                 predicting_llm forwards them to
                                                 sft_on_round)
    {whole_record_path}                          (n_total, T+1) opinion vectors

Choose the evaluation scope with --prompts-scope:
    labeled    -> SFT-train subset only (n_labeled agents)         -- always available
    unlabeled  -> held-out subset (n_unlabeled agents)             -- needs prompts_unlabeled.json
    all        -> both concatenated (n_labeled + n_unlabeled)      -- needs prompts_unlabeled.json

Loss choice: pure CE (NLL on the opinion target). This decouples evaluation
from the beta-KL anchor we are varying. To switch to training-loss
(CE + beta*KL), extend _per_pair_nll_batch.

CLI:
    python -m LLM_experiments.egta_fj \
        --run RECORD.pk ADAPTER_DIR LABEL  \
        --run RECORD.pk ADAPTER_DIR LABEL  \
        ...
        [--static-run RECORD.pk BASE_MODEL LABEL]   (untrained-model anchor; optional)
        --prompts-scope {labeled,unlabeled,all}   (default: labeled)
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --snapshot-times 0 5 10 15 20 25 30 \
        --out /path/egta_fj_sweep.npz

The --static-run row pairs a no-training opinion trajectory with the base model
as the evaluated policy at every snapshot. Use this to anchor "what does no
training look like" in the EGTA tensor.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from LLM_experiments.egta_matrix import (
    build_bundle,
    EGTAInputs,
    print_summary,
    save_npz,
)
from LLM_experiments.egta_wandb import WandbSession, add_wandb_args


@dataclass
class FJRun:
    """Per-intervention artifacts loaded from disk.

    If static_model_path is set, the intervention's *model* is the same at every
    snapshot (the untrained baseline) -- only its *world* (whole_record) was
    induced by running FJ dynamics with that fixed model. adapter_at(t) ignores
    t in that case and returns static_model_path.
    """
    label: str
    whole_record: np.ndarray   # (n, T+1) opinion trajectory restricted to labeled agents
    prompts: list[str]         # (n,) static prompts
    adapter_dir: Path          # directory holding round_{t}/ subdirs (unused for static runs)
    static_model_path: str | None = None  # HF id or local path; if set, model is t-invariant

    @property
    def is_static(self) -> bool:
        return self.static_model_path is not None

    def opinion_vector(self, t: int) -> np.ndarray:
        if t < 0 or t >= self.whole_record.shape[1]:
            raise IndexError(f"{self.label}: t={t} out of bounds (T+1={self.whole_record.shape[1]})")
        return self.whole_record[:, t]

    def adapter_at(self, t: int) -> Path:
        if self.static_model_path is not None:
            return Path(self.static_model_path)
        p = self.adapter_dir / f"round_{t}"
        if not p.exists():
            raise FileNotFoundError(f"{self.label}: missing adapter at t={t}: {p}")
        return p


def load_fj_run(record_path: Path, adapter_dir: Path, label: str, *,
                scope: str = "labeled") -> FJRun:
    """Load whole_record and the requested prompts subset for one intervention.

    scope in {"labeled", "unlabeled", "all"}:
        labeled    -> use prompts_labeled.json, slice whole_record[:n_labeled]
        unlabeled  -> use prompts_unlabeled.json, slice whole_record[n_labeled:n_labeled+n_unlab]
        all        -> concatenate labeled + unlabeled, slice whole_record accordingly

    The convention `whole_record[:n_labeled] = labeled agents, then unlabeled` comes
    from pokec_simulations: `x_labeled_prior = x_temp[:n]; x_unlabeled_prior = x_temp[n:]`.
    """
    if scope not in ("labeled", "unlabeled", "all"):
        raise ValueError(f"scope must be 'labeled', 'unlabeled', or 'all'; got {scope!r}")
    record_path = Path(record_path); adapter_dir = Path(adapter_dir)
    if record_path.suffix in (".pk", ".pkl"):
        with open(record_path, "rb") as f:
            whole_record = pickle.load(f)
    elif record_path.suffix == ".npy":
        whole_record = np.load(record_path)
    else:
        raise ValueError(f"{record_path}: unsupported extension; need .pk/.pkl/.npy")
    whole_record = np.asarray(whole_record)
    if whole_record.ndim != 2:
        raise ValueError(f"{record_path}: whole_record must be 2D (N, T+1), got {whole_record.shape}")

    prompts_labeled_path = adapter_dir / "prompts_labeled.json"
    prompts_unlabeled_path = adapter_dir / "prompts_unlabeled.json"
    with open(prompts_labeled_path) as f:
        prompts_labeled = json.load(f)
    n_labeled = len(prompts_labeled)
    if scope == "labeled":
        prompts = list(prompts_labeled)
        if whole_record.shape[0] < n_labeled:
            raise ValueError(f"{record_path}: only {whole_record.shape[0]} agents < {n_labeled} labeled prompts")
        whole_record_sub = whole_record[:n_labeled, :]
    else:
        if not prompts_unlabeled_path.exists():
            raise FileNotFoundError(
                f"{prompts_unlabeled_path}: scope={scope!r} requires prompts_unlabeled.json. "
                f"Re-run training with SAVE_PROMPTS=1 (predicting_llm now passes them through)."
            )
        with open(prompts_unlabeled_path) as f:
            prompts_unlabeled = json.load(f)
        n_unlab = len(prompts_unlabeled)
        if whole_record.shape[0] < n_labeled + n_unlab:
            raise ValueError(
                f"{record_path}: {whole_record.shape[0]} agents < {n_labeled}+{n_unlab} prompts"
            )
        if scope == "unlabeled":
            prompts = list(prompts_unlabeled)
            whole_record_sub = whole_record[n_labeled:n_labeled + n_unlab, :]
        else:  # all
            prompts = list(prompts_labeled) + list(prompts_unlabeled)
            whole_record_sub = whole_record[:n_labeled + n_unlab, :]

    return FJRun(label=label, whole_record=whole_record_sub, prompts=prompts,
                 adapter_dir=adapter_dir)


def load_fj_static_run(record_path: Path, base_model: str, label: str,
                       prompts_source: FJRun) -> FJRun:
    """Build an FJ run whose model is the untrained base at every snapshot.

    The record is the opinion trajectory from an untuned FJ run (e.g. run_untuned_llm.py).
    Prompts and the labeled-vs-unlabeled scope are inherited from `prompts_source`,
    so the same agent subset is used as for the trained interventions.
    """
    record_path = Path(record_path)
    if record_path.suffix in (".pk", ".pkl"):
        with open(record_path, "rb") as f:
            whole_record = pickle.load(f)
    elif record_path.suffix == ".npy":
        whole_record = np.load(record_path)
    else:
        raise ValueError(f"{record_path}: unsupported extension; need .pk/.pkl/.npy")
    whole_record = np.asarray(whole_record)
    n = len(prompts_source.prompts)
    if whole_record.shape[0] < n:
        raise ValueError(
            f"{record_path}: only {whole_record.shape[0]} agents < {n} required by prompts_source"
        )
    return FJRun(
        label=label,
        whole_record=whole_record[:n, :],
        prompts=list(prompts_source.prompts),
        adapter_dir=Path("."),  # unused
        static_model_path=base_model,
    )


def _check_runs_compatible(runs: Sequence[FJRun]) -> None:
    n_ref = runs[0].whole_record.shape[0]
    T1_ref = runs[0].whole_record.shape[1]
    prompts_ref = runs[0].prompts
    for r in runs[1:]:
        if r.whole_record.shape[0] != n_ref:
            raise ValueError(f"{r.label}: n_labeled {r.whole_record.shape[0]} != {n_ref}")
        if r.whole_record.shape[1] != T1_ref:
            raise ValueError(f"{r.label}: T+1 {r.whole_record.shape[1]} != {T1_ref}")
        if r.prompts != prompts_ref:
            raise ValueError(f"{r.label}: prompts differ from reference run -- labeled subsets must match across interventions")


def _per_pair_nll_batch(model, tok, prompts: list[str], completions: list[str],
                        batch_size: int, max_length: int, device) -> np.ndarray:
    """Mean NLL per (prompt, completion) pair, completion-only loss masking.

    Mirrors trl SFTConfig(completion_only_loss=True, max_length=...) tokenization
    exactly: tokenize prompt and completion SEPARATELY with add_special_tokens=False,
    concatenate token IDs, then mask prompt positions to -100. This matches
    trl.sft_trainer._collate_prompt_completion (TRL 1.2). Tokenizing tok(p + c) is
    wrong when BPE merges across the boundary.
    """
    n = len(prompts)
    out = np.zeros(n, dtype=np.float64)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        encs = []
        for p, c in zip(prompts[start:end], completions[start:end]):
            prompt_ids = tok(p, add_special_tokens=False)["input_ids"]
            completion_ids = tok(c, add_special_tokens=False)["input_ids"]
            full = list(prompt_ids) + list(completion_ids)
            labels_seq = [-100] * len(prompt_ids) + list(completion_ids)
            if len(full) > max_length:
                full = full[:max_length]
                labels_seq = labels_seq[:max_length]
            encs.append({"input_ids": full, "labels": labels_seq})
        max_len = max(len(e["input_ids"]) for e in encs)
        B = len(encs)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        for k, e in enumerate(encs):
            L = len(e["input_ids"])
            input_ids[k, :L] = torch.tensor(e["input_ids"], dtype=torch.long)
            attention_mask[k, :L] = 1
            labels[k, :L] = torch.tensor(e["labels"], dtype=torch.long)
        with torch.no_grad():
            o = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            logits = o.logits  # (B, L, V)
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].to(device)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        safe = shift_labels.clamp(min=0)
        gathered = log_probs.gather(2, safe.unsqueeze(-1)).squeeze(-1)
        mask = (shift_labels != -100).float()
        per_item_sum = -(gathered * mask).sum(dim=1)
        per_item_n = mask.sum(dim=1).clamp(min=1)
        per_item_nll = (per_item_sum / per_item_n).cpu().numpy()
        out[start:end] = per_item_nll
    return out


def _auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_lora_dir(p: Path) -> bool:
    """A directory is a LoRA adapter iff it has adapter_config.json."""
    return (p / "adapter_config.json").exists()


def cross_evaluate_fj(runs: Sequence[FJRun], snapshot_times: Sequence[int],
                      base_model_name: str, *, batch_size: int = 64, max_length: int = 200,
                      device_name: str | None = None,
                      wb: WandbSession | None = None) -> np.ndarray:
    """Return M of shape (S, K, K) by direct cross-evaluation. K=len(runs), S=len(snapshot_times).

    Auto-detects per-snapshot whether the save dir is a LoRA adapter or a full
    fine-tuned model (by presence of adapter_config.json). Mixing the two across
    runs is allowed (e.g. some interventions used USE_LORA=1, others USE_LORA=0).
    """
    _check_runs_compatible(runs)
    K = len(runs); S = len(snapshot_times)
    device = torch.device(device_name) if device_name else _auto_device()
    print(f"[egta_fj] device={device} base={base_model_name} K={K} S={S}")

    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Pre-detect which (s, j) pairs need LoRA so we only pay the base-model load once.
    any_lora = any(_is_lora_dir(run.adapter_at(t)) for t in snapshot_times for run in runs)
    base = None
    if any_lora:
        base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype).to(device)
        base.eval()
        print(f"[egta_fj] loaded base model for LoRA path")

    M = np.zeros((S, K, K), dtype=np.float64)
    prompts = runs[0].prompts  # invariant across runs by _check_runs_compatible
    # Pre-load static-run models once; their weights don't change across snapshots.
    static_models = {}  # j -> model
    for j, run_j in enumerate(runs):
        if run_j.is_static:
            path = run_j.adapter_at(snapshot_times[0])
            m = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype).to(device)
            m.eval()
            static_models[j] = m
            print(f"[egta_fj] pre-loaded static model for j={run_j.label} from {path}")
    for s, t_s in enumerate(snapshot_times):
        for j, run_j in enumerate(runs):
            if j in static_models:
                model = static_models[j]
                opened_here = False
            else:
                ckpt_path = run_j.adapter_at(t_s)
                if _is_lora_dir(ckpt_path):
                    model = PeftModel.from_pretrained(base, str(ckpt_path))
                else:
                    model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), torch_dtype=dtype).to(device)
                model.eval()
                opened_here = True
            for i, run_i in enumerate(runs):
                completions = [f"{float(x):.2f}" for x in run_i.opinion_vector(t_s)]
                nll = _per_pair_nll_batch(model, tok, prompts, completions,
                                          batch_size=batch_size, max_length=max_length,
                                          device=device)
                M[s, i, j] = float(nll.mean())
            if opened_here:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print(f"[egta_fj]  done t_s={t_s} j={run_j.label}")
        if wb is not None:
            slice_s = M[s]
            diag_s = np.einsum("ii->i", slice_s)
            off_s = (slice_s.sum() - diag_s.sum()) / max(K * (K - 1), 1)
            wb.log_metrics({
                "egta/t_s": int(t_s),
                "egta/diag_min": float(diag_s.min()),
                "egta/diag_max": float(diag_s.max()),
                "egta/diag_mean": float(diag_s.mean()),
                "egta/offdiag_mean": float(off_s),
            }, step=s)
    return M


def _parse_run_triples(run_args, scope):
    """Each --run takes 3 positional args: record_path adapter_dir label.
    prompts_labeled.json (and optionally prompts_unlabeled.json) are read from adapter_dir.
    """
    runs = []
    for triple in run_args:
        if len(triple) != 3:
            raise SystemExit(f"--run expects 3 args (record, adapter_dir, label); got {triple}")
        record_path, adapter_dir, label = triple
        runs.append(load_fj_run(Path(record_path), Path(adapter_dir), label, scope=scope))
    return runs


def main():
    ap = argparse.ArgumentParser(description="FJ adapter -> EGTA decoupled-PR tensor via direct cross-evaluation.")
    ap.add_argument("--run", nargs=3, action="append", metavar=("RECORD", "ADAPTER_DIR", "LABEL"),
                    required=True,
                    help="One per intervention. Repeat for K interventions. "
                         "RECORD = .pk/.pkl/.npy of whole_record (N, T+1). "
                         "ADAPTER_DIR = directory containing round_{t}/ subdirs and "
                         "prompts_labeled.json (and optionally prompts_unlabeled.json).")
    ap.add_argument("--static-run", nargs=3, action="append", default=[],
                    metavar=("RECORD", "BASE_MODEL", "LABEL"),
                    help="Optional 'no-training' intervention. RECORD = whole_record from an "
                         "untuned FJ run (run_untuned_llm.py). BASE_MODEL = HF id or local path "
                         "for the untrained model (e.g. Qwen/Qwen2.5-0.5B-Instruct). "
                         "Prompts and labeled-vs-unlabeled scope are inherited from the first --run. "
                         "Repeatable if you have multiple static seeds.")
    ap.add_argument("--prompts-scope", choices=["labeled", "unlabeled", "all"], default="labeled",
                    help="Which subset of agents to cross-evaluate over. "
                         "'labeled' = agents SFT trained on (always available). "
                         "'unlabeled' = held-out agents (requires prompts_unlabeled.json). "
                         "'all' = both concatenated.")
    ap.add_argument("--base-model", type=str, required=True,
                    help="HuggingFace model id used to train the LoRA adapters (e.g. Qwen/Qwen2.5-0.5B-Instruct).")
    ap.add_argument("--snapshot-times", type=int, nargs="+", required=True,
                    help="Round indices to evaluate at (must match round_{t}/ subdirs).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=200,
                    help="Token budget per (prompt+completion) pair. Must match training SFTConfig.max_length.")
    ap.add_argument("--device", type=str, default=None,
                    help="Override device (cuda/mps/cpu). Default: auto.")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    add_wandb_args(ap, default_project="egta-analysis")
    args = ap.parse_args()

    runs = _parse_run_triples(args.run, scope=args.prompts_scope)
    # Static runs inherit prompts from the first regular run; they must therefore
    # be added AFTER regular runs are loaded.
    for triple in args.static_run:
        record_path, base_model, label = triple
        runs.append(load_fj_static_run(Path(record_path), base_model, label,
                                       prompts_source=runs[0]))
    labels = [r.label for r in runs]
    training_run_tags = [Path(r.adapter_dir).name for r in runs if not r.is_static]
    print(f"[egta_fj] prompts_scope={args.prompts_scope}, n_agents={len(runs[0].prompts)}, "
          f"K={len(runs)} (static: {sum(r.is_static for r in runs)})")

    wb = WandbSession(
        enabled=args.wandb,
        project=args.wandb_project,
        name=args.wandb_name or args.out.stem,
        config={
            "adapter": "fj",
            "labels": labels,
            "K": len(runs),
            "S": len(args.snapshot_times),
            "snapshot_times": list(args.snapshot_times),
            "prompts_scope": args.prompts_scope,
            "n_agents": len(runs[0].prompts),
            "base_model": args.base_model,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "training_run_tags": training_run_tags,
            "static_labels": [r.label for r in runs if r.is_static],
        },
    )

    M = cross_evaluate_fj(runs, args.snapshot_times, args.base_model,
                          batch_size=args.batch_size, max_length=args.max_length,
                          device_name=args.device, wb=wb)
    inputs = EGTAInputs.from_matrix(
        M,
        t=np.asarray(args.snapshot_times, dtype=np.int64),
        labels=labels,
        metadata={"adapter": "fj", "base_model": args.base_model,
                  "max_length": args.max_length, "batch_size": args.batch_size,
                  "prompts_scope": args.prompts_scope},
    )
    bundle = build_bundle(inputs)
    save_npz(bundle, args.out)
    print(f"[egta_fj] wrote {args.out}")
    print_summary(bundle)

    if wb.active:
        M_final = bundle["M"]
        diag_T = np.einsum("ii->i", M_final[-1])
        offT = (M_final[-1].sum() - diag_T.sum()) / max(len(runs) * (len(runs) - 1), 1)
        wb.log_metrics({
            "egta/terminal_diag_min": float(diag_T.min()),
            "egta/terminal_diag_max": float(diag_T.max()),
            "egta/terminal_diag_mean": float(diag_T.mean()),
            "egta/terminal_offdiag_mean": float(offT),
        })
        wb.log_artifact(args.out, artifact_type="egta-fj-tensor")
        wb.finish()


if __name__ == "__main__":
    main()
