"""SFT trainer with explicit β · KL(π_θ || π_ref) on next-token distributions.

Adapted from Opinion-dynamics-post-training/llm_predictor.py KLSFTTrainer.
We pass the frozen reference model in directly (full-FT case) rather than
relying on PEFT's adapter-disable trick.

Also exposes `ManualMaskCollator` for pre-tokenized datasets where labels are
already constructed with -100 on prompt tokens (so the gradient bears strictly
on the answer span, by construction). This lets us bypass TRL's
`completion_only_loss` heuristic.
"""
from __future__ import annotations

import torch
from trl import SFTTrainer


class ManualMaskCollator:
    """Pad pre-tokenized input_ids / attention_mask / labels.

    `input_ids` and `attention_mask` pad with `pad_token_id` and 0; `labels`
    pads with -100 so padded positions never contribute to CE/KL loss.
    """

    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("pad_token_id is required (set tokenizer.pad_token first).")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples):
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attn, labels = [], [], []
        for ex in examples:
            n = len(ex["input_ids"])
            pad = max_len - n
            input_ids.append(list(ex["input_ids"]) + [self.pad_token_id] * pad)
            attn.append(list(ex["attention_mask"]) + [0] * pad)
            labels.append(list(ex["labels"]) + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class KLSFTTrainer(SFTTrainer):
    def __init__(self, *args, kl_beta: float = 0.0, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_beta = float(kl_beta)
        self.ref_model = ref_model
        if self.kl_beta > 0 and self.ref_model is None:
            raise ValueError("KLSFTTrainer with kl_beta > 0 needs ref_model.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce_loss = outputs.loss
        if self.kl_beta <= 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        with torch.no_grad():
            ref_logits = self.ref_model(**inputs).logits

        policy_logits = outputs.logits
        labels = inputs.get("labels")
        mask = (labels != -100).float() if labels is not None else torch.ones_like(policy_logits[..., 0])

        # Shift for next-token prediction.
        policy_logp = torch.log_softmax(policy_logits[:, :-1, :], dim=-1)
        ref_logp = torch.log_softmax(ref_logits[:, :-1, :], dim=-1)
        mask_shift = mask[:, 1:]

        policy_p = policy_logp.exp()
        kl = (policy_p * (policy_logp - ref_logp)).sum(dim=-1)
        kl = (kl * mask_shift).sum() / mask_shift.sum().clamp_min(1.0)

        loss = ce_loss + self.kl_beta * kl
        self.log({"ce_loss": ce_loss.detach().item(), "kl": kl.detach().item()})
        return (loss, outputs) if return_outputs else loss
