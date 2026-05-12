"""Sanity check: verify that egta_fj._per_pair_nll_batch matches the loss the
FJ SFT-KL trainer would compute on the same (prompt, completion) pairs.

Run on the cluster, in the FJ env, after at least one FJ run has been completed
with SAVE_ADAPTER=1, SAVE_ADAPTER_PER_ROUND=1, SAVE_PROMPTS=1.

Method:
  1. Load base model + a saved LoRA adapter from a real FJ run.
  2. Build a small dataset of (prompt, completion) pairs from the saved prompts +
     a corresponding column of whole_record (any time index works).
  3. Compute per-pair NLL via _per_pair_nll_batch (our function).
  4. Compute per-pair NLL via trl SFTTrainer with the exact same SFTConfig as
     llm_predictor.sft_on_round used. Mean over the batch should match (1) within
     numerical noise.

The trl path computes a single scalar batch-mean loss (mean over all unmasked
positions), not per-item. We compare:
    mean(per_pair_nll_from_our_fn)  vs  trl_batch_mean_loss
adjusted for per-item vs per-token averaging (see code).

Usage:
    python -m LLM_experiments.egta_fj_sanity \
        --record /path/whole_record_b0_s0.pk \
        --adapter-dir /path/adapters/fj_b0_s0 \
        --round 0 \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --max-length 200 \
        --n-pairs 16
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from LLM_experiments.egta_fj import _auto_device, _is_lora_dir, _per_pair_nll_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=Path, required=True,
                    help="whole_record (.pk/.pkl/.npy) from one FJ run.")
    ap.add_argument("--adapter-dir", type=Path, required=True,
                    help="adapters/{tag}/ root (must contain prompts_labeled.json and round_X/).")
    ap.add_argument("--round", type=int, required=True, help="Adapter round subdir to evaluate.")
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--max-length", type=int, default=200)
    ap.add_argument("--n-pairs", type=int, default=16, help="Number of (prompt, completion) pairs to compare on.")
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    # Load artifacts.
    if args.record.suffix in (".pk", ".pkl"):
        with open(args.record, "rb") as f:
            whole_record = pickle.load(f)
    else:
        whole_record = np.load(args.record)
    whole_record = np.asarray(whole_record)
    with open(args.adapter_dir / "prompts_labeled.json") as f:
        prompts = json.load(f)
    n = min(args.n_pairs, len(prompts), whole_record.shape[0])
    prompts = list(prompts[:n])
    # Use opinion vector at the same round we are loading the adapter for.
    y = whole_record[:n, args.round]
    completions = [f"{float(v):.2f}" for v in y]

    # Load model + adapter.
    device = _auto_device()
    print(f"[sanity] device={device}, n_pairs={n}, round={args.round}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    ckpt_path = args.adapter_dir / f"round_{args.round}"
    if _is_lora_dir(ckpt_path):
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device)
        base.eval()
        model = PeftModel.from_pretrained(base, str(ckpt_path))
        print(f"[sanity] loaded LoRA adapter from {ckpt_path} onto base {args.base_model}")
    else:
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), torch_dtype=dtype).to(device)
        print(f"[sanity] loaded full-FT model from {ckpt_path} (no adapter_config.json -> not LoRA)")
    model.eval()

    # (1) Our function: per-pair NLL.
    nll_ours = _per_pair_nll_batch(model, tok, prompts, completions,
                                   batch_size=n, max_length=args.max_length, device=device)
    mean_ours = float(nll_ours.mean())

    # (2) TRL reference: run SFTTrainer with the same SFTConfig and read the batch loss.
    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})
    cfg = SFTConfig(
        output_dir="./_sanity_tmp",
        per_device_train_batch_size=n,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=0.0,  # do not actually update
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        max_length=args.max_length,
        bf16=False,
        optim="adamw_torch",
    )
    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=ds)
    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    # Move batch to device.
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    logits = out.logits
    labels = batch["labels"]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logp = torch.log_softmax(shift_logits, dim=-1)
    safe = shift_labels.clamp(min=0)
    gathered = logp.gather(2, safe.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    # Per-item NLL from TRL-collated tensors:
    per_item_sum = -(gathered * mask).sum(dim=1)
    per_item_n = mask.sum(dim=1).clamp(min=1)
    nll_trl = (per_item_sum / per_item_n).cpu().numpy()
    mean_trl = float(nll_trl.mean())

    # Per-item comparison.
    abs_diff = np.abs(nll_ours - nll_trl)
    print(f"[sanity] per-pair NLL stats:")
    print(f"  ours mean = {mean_ours:.6f}")
    print(f"  trl  mean = {mean_trl:.6f}")
    print(f"  per-pair max abs diff = {abs_diff.max():.6e}")
    print(f"  per-pair mean abs diff = {abs_diff.mean():.6e}")
    if abs_diff.max() < args.rtol:
        print(f"[sanity] PASS (max diff < {args.rtol})")
    else:
        # Print up to 5 worst pairs for inspection.
        order = np.argsort(-abs_diff)
        print(f"[sanity] FAIL: max diff {abs_diff.max():.4e} >= rtol {args.rtol}")
        for k in order[:5]:
            print(f"  pair {k}: ours={nll_ours[k]:.4f}, trl={nll_trl[k]:.4f}, diff={abs_diff[k]:.4e}")


if __name__ == "__main__":
    main()
