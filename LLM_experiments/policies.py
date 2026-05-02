"""Classifier conditions for the Fig. 5 redo.

Each policy exposes:
    fit(sample) -> None        # called once per replicator step on the population sample
    score(test_pack) -> float  # called per state on its held-out test pool, returns acc

A `SamplePack` carries the population-weighted sample's features, labels, and
cached π_ref(y=1|x) values for those rows. `TestPack` is the per-state held-out
analog. Cached π_ref values come from `llm_pi_ref.build_or_load_pi_ref`.

The four conditions correspond to four KL-anchored learning regimes:

* `StaticThresholdPolicy` — frozen LLM, threshold tuned on the population sample.
  No training; baseline for "what does the unmodified prior do."

* `ClosedFormPolicy` — analytical optimum of Korbak-Williams form (II) KL-RL:
  π_θ*(y|x) ∝ π_ref(y|x) · h_p(y|x)^(1/β), where h_p is a logreg head fit on
  the population. No gradient descent; closed form via geometric mixture.

* `SFTPolicy` — gradient descent on form (I), MLE + KL: minimizes
  −log p_θ(y_obs|x) + β·KL(p_θ||π_ref). Standard "SFT-with-KL" practitioner loss.
  Note: NOT the same objective as form (II); use this as a baseline rather than
  as a gradient-descent realization of the closed_form geometric mixture.

* `RLKLPolicy` — gradient descent on form (II), Korbak-Williams KL-RL: minimizes
  −E_{y~q'_θ(·|x)}[log h_p(y|x)] + β·KL_2(q'_θ||π_ref') on the 2-class
  restricted softmax q' over {pos_token, neg_token}. Same objective as the
  closed_form analytical optimum; agreement between the two pins down
  optimizer / capacity gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from datasets import Dataset
from folktexts.prompting import encode_row_prompt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from trl import SFTConfig
from .sft_kl_trainer import KLSFTTrainer, RLKLSFTTrainer

from . import _folktexts_compat  # noqa: F401  — must precede folktexts imports
from folktexts.classifier import TransformersLLMClassifier
from .sft_kl_trainer import ManualMaskCollator


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------

@dataclass
class SamplePack:
    X: pd.DataFrame      # task features only, n rows
    y: np.ndarray        # (n,) binary labels
    pi_ref_p1: np.ndarray  # (n,) cached P(y=1|x) under base LLM


@dataclass
class TestPack:
    X: pd.DataFrame
    y: np.ndarray
    pi_ref_p1: np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy_at_threshold(p1: np.ndarray, y: np.ndarray, tau: float) -> float:
    pred = (p1 >= tau).astype(np.int64)
    return float((pred == y).mean())


def _best_threshold(p1: np.ndarray, y: np.ndarray) -> float:
    """Brute-force over candidate thresholds (sorted unique scores).

    Sample sizes here are small (~1000) so the O(n^2) version is fine. We
    return the threshold maximizing accuracy; ties broken toward 0.5.
    """
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], np.sort(p1)]))
    best_tau, best_acc = 0.5, -1.0
    for tau in candidates:
        acc = _accuracy_at_threshold(p1, y, tau)
        if acc > best_acc or (acc == best_acc and abs(tau - 0.5) < abs(best_tau - 0.5)):
            best_tau, best_acc = float(tau), acc
    return best_tau


# ---------------------------------------------------------------------------
# Condition A — frozen LLM, threshold tuned per population
# ---------------------------------------------------------------------------

class StaticThresholdPolicy:
    name = "static"

    def __init__(self):
        self.threshold: float = 0.5

    def fit(self, sample: SamplePack) -> None:
        self.threshold = _best_threshold(sample.pi_ref_p1, sample.y)

    def score(self, test: TestPack) -> float:
        return _accuracy_at_threshold(test.pi_ref_p1, test.y, self.threshold)


# ---------------------------------------------------------------------------
# Closed-form geometric mixture: π_β,p*(y|x) ∝ π_ref(y|x) · p_target(y|x;p)^(1/β)
# p_target is fit per replicator step on population-weighted sample.
# ---------------------------------------------------------------------------

class ClosedFormPolicy:
    name = "closed_form"

    def __init__(self, beta: float, head_kind: str = "logreg"):
        if beta <= 0:
            raise ValueError("beta must be > 0; β=0 limit is degenerate (use head-only baseline).")
        self.beta = float(beta)
        self.head_kind = head_kind
        self.head = None
        self._feature_cols: Optional[list] = None

    def _build_head(self):
        

        if self.head_kind == "logreg":
            return Pipeline([
                ("scale", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=200, n_jobs=1)),
            ])
        raise NotImplementedError(self.head_kind)

    def fit(self, sample: SamplePack) -> None:
        if self._feature_cols is None:
            self._feature_cols = list(sample.X.columns)
        self.head = self._build_head()
        # Single class in the sample => degenerate head; skip and let the LLM
        # alone govern this round.
        if len(np.unique(sample.y)) < 2:
            self.head = None
            return
        self.head.fit(sample.X.values, sample.y)

    def _mix_p1(self, X: pd.DataFrame, pi_ref_p1: np.ndarray) -> np.ndarray:
        if self.head is None:
            return pi_ref_p1
        p_tgt = self.head.predict_proba(X.values)[:, 1]
        # log π* ∝ log π_ref + (1/β) log p_target
        eps = 1e-9
        log_ref = np.stack([np.log(1.0 - pi_ref_p1 + eps), np.log(pi_ref_p1 + eps)], axis=1)
        log_tgt = np.stack([np.log(1.0 - p_tgt + eps), np.log(p_tgt + eps)], axis=1)
        log_unnorm = log_ref + (1.0 / self.beta) * log_tgt
        log_unnorm -= log_unnorm.max(axis=1, keepdims=True)
        exp = np.exp(log_unnorm)
        return exp[:, 1] / exp.sum(axis=1)

    def score(self, test: TestPack) -> float:
        p1 = self._mix_p1(test.X, test.pi_ref_p1)
        pred = (p1 >= 0.5).astype(np.int64)
        return float((pred == test.y).mean())


# ---------------------------------------------------------------------------
# Fast scorer: pre-tokenize the test pool once, batched forward pass per round.
# folktexts' TransformersLLMClassifier.predict_proba re-encodes and re-tokenizes
# the test pool on every call (Python-loop), costing ~10s per state per round.
# This scorer caches per-test-pack tokenized tensors on device and reads logits
# at the answer position for the two answer tokens directly.
# ---------------------------------------------------------------------------

class _FastTestScorer:
    """Cached fast path for per-state test scoring during the replicator loop.

    Keyed by `id(test_pack.X)`, which is stable when run_replicator builds
    test_packs once and reuses them every round.
    """

    def __init__(self, tokenizer, task, max_length: int = 512):
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = int(max_length)
        self._cache: dict[int, dict] = {}

    def _build(self, test_pack: "TestPack", device) -> dict:
        prompts = [
            encode_row_prompt(row, self.task)
            for _, row in test_pack.X.iterrows()
        ]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        last_idx = enc.attention_mask.sum(dim=-1) - 1
        return {
            "input_ids": enc.input_ids.to(device),
            "attention_mask": enc.attention_mask.to(device),
            "last_idx": last_idx.to(device),
        }

    def score(self, model, test_pack: "TestPack", pos_tok_id: int, neg_tok_id: int) -> float:
        device = next(model.parameters()).device
        key = id(test_pack.X)
        if key not in self._cache:
            self._cache[key] = self._build(test_pack, device)
        cache = self._cache[key]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = model(
                    input_ids=cache["input_ids"],
                    attention_mask=cache["attention_mask"],
                ).logits
            last_logits = logits[
                torch.arange(logits.shape[0], device=logits.device),
                cache["last_idx"],
            ]
            pos = last_logits[:, pos_tok_id]
            neg = last_logits[:, neg_tok_id]
            pred = (pos > neg).cpu().numpy().astype(np.int64)
        finally:
            if was_training:
                model.train()
        return float((pred == test_pack.y).mean())


def _compute_single_answer_tokens(tokenizer, task) -> dict[int, int]:
    """Return {0: neg_tok_id, 1: pos_tok_id}; both must be single-token."""
    qa = task.multiple_choice_qa
    if qa is None:
        raise RuntimeError("Task lacks a multiple-choice QA interface.")
    out: dict[int, int] = {}
    for cls in (0, 1):
        key = qa.get_answer_key_from_value(int(cls))
        ids = tokenizer.encode(" " + key, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Fast scorer requires single-token answers; got len={len(ids)} "
                f"for class={cls} key={key!r} ids={ids}."
            )
        out[cls] = ids[0]
    return out


# ---------------------------------------------------------------------------
# Condition C — full FT base LLM with β·KL anchor toward π_ref
# ---------------------------------------------------------------------------

class SFTPolicy:
    """Full FT Qwen with β·KL toward initial π_ref. Persistent across rounds.

    Mirrors the KLSFTTrainer pattern in
    Opinion-dynamics-post-training/llm_predictor.py — TRL SFTTrainer subclass
    that adds β · KL(π_θ || π_ref) on next-token distributions over labeled
    positions, with π_ref the frozen base model.
    """
    name = "sft"

    def __init__(
        self,
        base_model: str,
        beta: float,
        folktexts_dataset,
        sft_epochs: int = 1,
        batch_size: int = 4,
        grad_accum: int = 1,
        learning_rate: float = 2e-5,
        optimizer: str = "adamw_torch",
        continual: bool = True,
        inference_batch_size: int = 64,
    ):
        """Default is **continual** SFT — weights persist across replicator
        rounds, β·KL anchor pulls toward π_ref each step. Matches the
        Mendler-Dünner et al. 2020 framing of performative prediction
        (stochastic optimization with persistent state) and the realistic
        deployed-system regime where models are not redeployed from scratch.

        Pass `continual=False` (or set env `SFT_FRESH=1` upstream) to reset
        weights to π_ref at the start of each `fit()` — the RRM / S&R
        Markov-in-p ablation, useful for theory comparison.
        """
        self.base_model = base_model
        self.beta = float(beta)
        self.folktexts_dataset = folktexts_dataset
        self.sft_epochs = sft_epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.continual = bool(continual)
        self.inference_batch_size = int(inference_batch_size)
        self._max_length = 512
        self._init_model_lazy()

    def _init_model_lazy(self):


        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Avoid `device_map="auto"` on CPU — accelerate's dispatcher can hang
        # on Apple Silicon. Explicit CPU placement when no CUDA.
        common = dict(torch_dtype=dtype)
        if cuda:
            common["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **common)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.base_model, **common)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self._fast_scorer = _FastTestScorer(
            self.tokenizer, self.folktexts_dataset.task, max_length=self._max_length,
        )
        self._answer_tok_id = _compute_single_answer_tokens(
            self.tokenizer, self.folktexts_dataset.task,
        )

    def _build_clf(self):

        return TransformersLLMClassifier(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.folktexts_dataset.task,
            batch_size=self.inference_batch_size,
            correct_order_bias=False, #set to trye to change the order correcting bias.
        )

    def _make_dataset(self, sample: SamplePack):
        """Build a pre-tokenized HF Dataset with manual prompt/answer masking.

        For each example we tokenize the folktexts prompt and the
        leading-space-prefixed answer key separately, concat, and emit:

            input_ids = prompt_ids + answer_ids
            labels    = [-100] * len(prompt_ids) + answer_ids
            attention_mask = all ones

        This makes the gradient-bearing token id identical to the answer token
        id folktexts reads at scoring time (see check_completion_tokenization.py),
        and removes any reliance on TRL's `completion_only_loss` heuristic.
        """

        task = self.folktexts_dataset.task
        qa = task.multiple_choice_qa
        if qa is None:
            raise RuntimeError("Task lacks a multiple-choice QA interface.")

        max_len = self._max_length
        examples = []
        boundary_failures = 0
        for (_, row), yi in zip(sample.X.iterrows(), sample.y):
            prompt = encode_row_prompt(row, task)
            answer_key = qa.get_answer_key_from_value(int(yi))

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(" " + answer_key, add_special_tokens=False)

            # Sanity: separate-then-concat tokenization should equal joint
            # tokenization at the boundary. If a BPE merge spans the join we
            # warn and fall back to the joint tokenization (still anchored to
            # the answer ids by length).
            joint_ids = self.tokenizer.encode(prompt + " " + answer_key, add_special_tokens=False)
            if joint_ids != prompt_ids + answer_ids:
                boundary_failures += 1
                # Use the joint tokenization, place answer_ids at the end:
                # this is correct as long as the trailing len(answer_ids)
                # tokens of joint_ids equal answer_ids.
                if joint_ids[-len(answer_ids):] == answer_ids:
                    prompt_ids = joint_ids[:-len(answer_ids)]
                else:
                    # Hard fail — boundary is ambiguous.
                    raise RuntimeError(
                        f"Tokenization boundary ambiguous for answer={answer_key!r}; "
                        f"separate tokenization does not align with joint."
                    )

            input_ids = prompt_ids + answer_ids
            if len(input_ids) > max_len:
                # Truncate the head of the prompt; never truncate the answer.
                cut = len(input_ids) - max_len
                prompt_ids = prompt_ids[cut:]
                input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + list(answer_ids)
            examples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })

        if boundary_failures and not getattr(self, "_warned_boundary", False):
            print(f"[SFTPolicy] tokenization boundary recovered for {boundary_failures} examples "
                  f"(BPE merge across prompt/answer join).")
            self._warned_boundary = True

        return Dataset.from_list(examples)

    def fit(self, sample: SamplePack) -> None:
        

        ds = self._make_dataset(sample)
        report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
        # We supply pre-tokenized input_ids / attention_mask / labels with
        # explicit -100 masking on the prompt span, so completion_only_loss is
        # turned OFF and we provide our own collator that pads labels with -100
        # rather than the input pad id.
        cfg = SFTConfig(
            output_dir="./_sft_round_tmp",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            num_train_epochs=self.sft_epochs,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to=report_to,
            completion_only_loss=False,
            max_length=self._max_length,
            bf16=True,
            optim=self.optimizer,
            remove_unused_columns=False,
        )
        # Fresh-each-round semantics: reset trainable weights to π_ref before
        # this round's optimization. Optimizer state is already fresh because
        # the trainer is re-created below. We reuse `ref_model.state_dict()`
        # as the source of truth — it holds an unmodified copy of the base
        # weights since we froze it in `_init_model_lazy`.
        if not self.continual:
            self.model.load_state_dict(self.ref_model.state_dict())

        
        collator = ManualMaskCollator(pad_token_id=self.tokenizer.pad_token_id)
        trainer = KLSFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=cfg,
            train_dataset=ds,
            data_collator=collator,
            kl_beta=self.beta,
            ref_model=self.ref_model,
        )

        # Round-1 sanity: confirm prompt tokens are masked to -100 and only the
        # completion tokens contribute gradient. Same check as opdyn's
        # pokec_simulations.py:420-431. Set SFT_SANITY=1 to enable.
        if os.environ.get("SFT_SANITY", "0") == "1" and not getattr(self, "_sanity_done", False):
            try:
                batch = next(iter(trainer.get_train_dataloader()))
                lb = batch["labels"][0]
                ids = batch["input_ids"][0]
                n_masked = int((lb == -100).sum())
                n_total = int(lb.numel())
                # Identify the unmasked (gradient-bearing) positions.
                kept_pos = (lb != -100).nonzero(as_tuple=True)[0].tolist()
                kept_ids = lb[lb != -100].tolist()
                kept_decoded = self.tokenizer.decode(kept_ids)
                print(f"[sft-sanity] labels shape={tuple(lb.shape)} masked={n_masked}/{n_total} "
                      f"({100*n_masked/n_total:.0f}%)  kept_positions={kept_pos}")
                print(f"[sft-sanity] kept_token_ids={kept_ids}  decoded={kept_decoded!r}")
                print(f"[sft-sanity] last 12 input_ids: {ids[-12:].tolist()}")
                print(f"[sft-sanity] last 12 labels   : {lb[-12:].tolist()}")
            except Exception as e:
                print(f"[sft-sanity] check failed: {e}")
            self._sanity_done = True

        trainer.train()

    def score(self, test: TestPack) -> float:
        return self._fast_scorer.score(
            self.model, test,
            pos_tok_id=self._answer_tok_id[1],
            neg_tok_id=self._answer_tok_id[0],
        )


# ---------------------------------------------------------------------------
# Condition D — full FT base LLM, KL-RL form (II) gradient descent.
# Reward model is a logreg head fit per replicator step (matches ClosedForm).
# ---------------------------------------------------------------------------

class RLKLPolicy:
    """Form (II) gradient-descent realization of the Korbak-Williams KL-RL optimum.

    Per replicator step:
        1. Fit a reward model h_p (logreg) on the population sample.
        2. Build a SFT-style dataset; each example carries
              log_reward_pos = log h_p(y=1|x)
              log_reward_neg = log h_p(y=0|x).
        3. Train the LLM with `RLKLSFTTrainer`, minimizing
              -E_{y~q'_θ(·|x)}[log h_p(y|x)] + β · KL_2(q'_θ || π_ref')
           on the 2-class restricted softmax q' over the two answer tokens.

    The trainer's optimum is the same geometric mixture computed analytically
    by `ClosedFormPolicy`, so the two should agree at convergence. Defaults
    mirror `SFTPolicy` (continual=True, 1 epoch per round, full FT).
    """

    name = "rl_kl"

    def __init__(
        self,
        base_model: str,
        beta: float,
        folktexts_dataset,
        sft_epochs: int = 1,
        batch_size: int = 4,
        grad_accum: int = 1,
        learning_rate: float = 2e-5,
        optimizer: str = "adamw_torch",
        continual: bool = True,
        head_kind: str = "logreg",
        eps: float = 1e-9,
        inference_batch_size: int = 64,
    ):
        if beta <= 0:
            raise ValueError("RLKLPolicy requires beta > 0; β=0 limit is unregularized.")
        self.base_model = base_model
        self.beta = float(beta)
        self.folktexts_dataset = folktexts_dataset
        self.sft_epochs = sft_epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.continual = bool(continual)
        self.head_kind = head_kind
        self.eps = float(eps)
        self.inference_batch_size = int(inference_batch_size)
        self._max_length = 512
        self.head: Optional[Pipeline] = None
        self._init_model_lazy()
        self._answer_tokens = self._compute_answer_tokens()
        self._fast_scorer = _FastTestScorer(
            self.tokenizer, self.folktexts_dataset.task, max_length=self._max_length,
        )

    def _init_model_lazy(self):
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        common = dict(torch_dtype=dtype)
        if cuda:
            common["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **common)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.base_model, **common)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    def _compute_answer_tokens(self) -> dict[int, list[int]]:
        """Encode the leading-space answer keys for both classes once."""
        task = self.folktexts_dataset.task
        qa = task.multiple_choice_qa
        if qa is None:
            raise RuntimeError("Task lacks a multiple-choice QA interface.")
        out: dict[int, list[int]] = {}
        for cls in (0, 1):
            key = qa.get_answer_key_from_value(int(cls))
            ids = self.tokenizer.encode(" " + key, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"RLKLPolicy currently supports single-token answers; got "
                    f"len={len(ids)} for class={cls} key={key!r} ids={ids}."
                )
            out[cls] = ids
        return out

    def _build_head(self):
        if self.head_kind == "logreg":
            return Pipeline([
                ("scale", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=200, n_jobs=1)),
            ])
        raise NotImplementedError(self.head_kind)

    def _build_clf(self):
        return TransformersLLMClassifier(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.folktexts_dataset.task,
            batch_size=self.inference_batch_size,
            correct_order_bias=False,
        )

    def _make_dataset(self, sample: SamplePack, h_pos: np.ndarray):
        """Build a HF Dataset with masked prompt + per-row log-reward fields.

        h_pos[i] = h_p(y=1 | x_i), the reward model's positive-class probability.
        Each example carries:
            input_ids, attention_mask, labels (same as SFTPolicy)
            log_reward_pos = log h_pos[i]
            log_reward_neg = log (1 - h_pos[i])
        """
        task = self.folktexts_dataset.task
        qa = task.multiple_choice_qa
        if qa is None:
            raise RuntimeError("Task lacks a multiple-choice QA interface.")

        max_len = self._max_length
        examples = []
        boundary_failures = 0
        for ((_, row), yi, hp) in zip(sample.X.iterrows(), sample.y, h_pos):
            prompt = encode_row_prompt(row, task)
            answer_key = qa.get_answer_key_from_value(int(yi))

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(" " + answer_key, add_special_tokens=False)

            joint_ids = self.tokenizer.encode(prompt + " " + answer_key, add_special_tokens=False)
            if joint_ids != prompt_ids + answer_ids:
                boundary_failures += 1
                if joint_ids[-len(answer_ids):] == answer_ids:
                    prompt_ids = joint_ids[:-len(answer_ids)]
                else:
                    raise RuntimeError(
                        f"Tokenization boundary ambiguous for answer={answer_key!r}; "
                        f"separate tokenization does not align with joint."
                    )

            input_ids = prompt_ids + answer_ids
            if len(input_ids) > max_len:
                cut = len(input_ids) - max_len
                prompt_ids = prompt_ids[cut:]
                input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + list(answer_ids)

            hp_clip = float(np.clip(hp, self.eps, 1.0 - self.eps))
            examples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "log_reward_pos": float(np.log(hp_clip)),
                "log_reward_neg": float(np.log(1.0 - hp_clip)),
            })

        if boundary_failures and not getattr(self, "_warned_boundary", False):
            print(f"[RLKLPolicy] tokenization boundary recovered for {boundary_failures} examples "
                  f"(BPE merge across prompt/answer join).")
            self._warned_boundary = True

        return Dataset.from_list(examples)

    def fit(self, sample: SamplePack) -> None:
        # 1. Fit the reward model on the population sample.
        if len(np.unique(sample.y)) < 2:
            # Degenerate single-class round: fall back to a "dummy head" that
            # predicts the observed class. The KL term still pulls the LM
            # toward π_ref so this is still a valid update.
            const = float(sample.y.mean())
            h_pos = np.full(len(sample.y), const, dtype=np.float64)
        else:
            self.head = self._build_head()
            self.head.fit(sample.X.values, sample.y)
            h_pos = self.head.predict_proba(sample.X.values)[:, 1]

        # 2. Build the dataset with per-row reward log-probs.
        ds = self._make_dataset(sample, h_pos)

        # 3. Continual vs. fresh-each-round semantics, identical to SFTPolicy.
        if not self.continual:
            self.model.load_state_dict(self.ref_model.state_dict())

        report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
        cfg = SFTConfig(
            output_dir="./_rlkl_round_tmp",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            num_train_epochs=self.sft_epochs,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to=report_to,
            completion_only_loss=False,
            max_length=self._max_length,
            bf16=True,
            optim=self.optimizer,
            remove_unused_columns=False,
        )

        collator = ManualMaskCollator(pad_token_id=self.tokenizer.pad_token_id)
        trainer = RLKLSFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=cfg,
            train_dataset=ds,
            data_collator=collator,
            kl_beta=self.beta,
            ref_model=self.ref_model,
            pos_token_id=self._answer_tokens[1][0],
            neg_token_id=self._answer_tokens[0][0],
        )

        if os.environ.get("SFT_SANITY", "0") == "1" and not getattr(self, "_sanity_done", False):
            try:
                batch = next(iter(trainer.get_train_dataloader()))
                lb = batch["labels"][0]
                lr_pos = batch["log_reward_pos"][:4].tolist()
                lr_neg = batch["log_reward_neg"][:4].tolist()
                kept_pos = (lb != -100).nonzero(as_tuple=True)[0].tolist()
                kept_ids = lb[lb != -100].tolist()
                print(f"[rlkl-sanity] kept_positions={kept_pos} kept_ids={kept_ids}")
                print(f"[rlkl-sanity] log_reward_pos[:4]={lr_pos}")
                print(f"[rlkl-sanity] log_reward_neg[:4]={lr_neg}")
                print(f"[rlkl-sanity] pos_token_id={self._answer_tokens[1][0]} "
                      f"neg_token_id={self._answer_tokens[0][0]}")
            except Exception as e:
                print(f"[rlkl-sanity] check failed: {e}")
            self._sanity_done = True

        trainer.train()

    def score(self, test: TestPack) -> float:
        return self._fast_scorer.score(
            self.model, test,
            pos_tok_id=self._answer_tokens[1][0],
            neg_tok_id=self._answer_tokens[0][0],
        )
