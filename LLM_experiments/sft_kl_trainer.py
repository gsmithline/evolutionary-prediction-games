"""SFT trainers for two distinct KL-regularized objectives.

Both inherit from TRL's SFTTrainer and add an explicit β·KL(π_θ || π_ref)
toward a frozen reference. They differ in the data-fit term:

* `KLSFTTrainer` — form (I), MLE + KL. The data term is the standard
  cross-entropy on observed labels: -log p_θ(y_obs | x). This is the
  "SFT-with-KL" loss most practitioners write.

* `RLKLSFTTrainer` — form (II), Korbak-Williams KL-regularized RL with reward
  = log h_p(y|x). The data term is the *expectation under the policy* of the
  reward model's log-density: -E_{y~π_θ(·|x)}[log h_p(y|x)]. For binary y the
  expectation is a closed-form 2-class sum and no sampling is needed. The
  optimum of this objective is exactly the geometric mixture
  π_θ*(y|x) ∝ π_ref(y|x) · h_p(y|x)^(1/β) implemented analytically by
  `policies.ClosedFormPolicy`.

Also exposes `ManualMaskCollator` for pre-tokenized datasets where labels are
already constructed with -100 on prompt tokens (so the gradient bears strictly
on the answer span, by construction). For form (II) we extend the collator to
pass through per-row reward-model log-probabilities as additional batch
tensors.
"""
from __future__ import annotations

import torch
from trl import SFTTrainer


class ManualMaskCollator:
    """Pad pre-tokenized input_ids / attention_mask / labels.

    `input_ids` and `attention_mask` pad with `pad_token_id` and 0; `labels`
    pads with -100 so padded positions never contribute to CE/KL loss. Any
    additional scalar fields on the example dicts (e.g. `log_reward_pos`,
    `log_reward_neg` for the form-II trainer) are stacked into 1-D tensors
    keyed by name.
    """

    _LIST_FIELDS = ("input_ids", "attention_mask", "labels")

    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("pad_token_id is required (set tokenizer.pad_token first).")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples):
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attn, labels = [], [], []
        scalar_fields: dict[str, list] = {}
        for ex in examples:
            n = len(ex["input_ids"])
            pad = max_len - n
            input_ids.append(list(ex["input_ids"]) + [self.pad_token_id] * pad)
            attn.append(list(ex["attention_mask"]) + [0] * pad)
            labels.append(list(ex["labels"]) + [-100] * pad)
            for k, v in ex.items():
                if k in self._LIST_FIELDS:
                    continue
                scalar_fields.setdefault(k, []).append(v)
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        for k, vals in scalar_fields.items():
            out[k] = torch.tensor(vals, dtype=torch.float32)
        return out


class KLSFTTrainer(SFTTrainer):
    """Form (I): MLE + β·KL on next-token distributions.

    Loss per example (in expectation over (x, y_obs) ~ D_p):
        L = -log p_θ(y_obs | x) + β · KL(p_θ(·|x) || π_ref(·|x))

    This is the standard "SFT-with-KL anchor" loss. Note: this is NOT the same
    objective as the Korbak-Williams KL-regularized RL form; for that, use
    `RLKLSFTTrainer`.
    """

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


class RLKLSFTTrainer(SFTTrainer):
    """Form (II): KL-regularized RL with reward = log h_p(y|x).

    Korbak & Williams 2022 framing. The objective on the 2-class restricted
    softmax q' over {pos_token_id, neg_token_id} at each answer position is:

        max_θ  E_{y~q'_θ(·|x)}[log h_p(y|x)]  -  β · KL(q'_θ || π_ref')

    Optimum is q'* ∝ π_ref' · h_p^(1/β), which `policies.ClosedFormPolicy`
    computes analytically. This trainer realizes the same optimum via gradient
    descent on the LLM weights.

    For binary y the policy expectation is a closed-form 2-class sum:

        L = -[ q'(pos|x) · log h_p(pos|x) + q'(neg|x) · log h_p(neg|x) ]
            + β · KL_2( q'_θ(·|x) || π_ref'(·|x) )

    where KL_2 is the 2-class KL on the renormalized {pos, neg} distribution.
    The dataset must carry `log_reward_pos` and `log_reward_neg` per example
    (computed from the reward model `h_p` upstream); the collator passes these
    through. The trainer reads them from `inputs` at the answer position.
    """

    def __init__(
        self,
        *args,
        kl_beta: float = 0.0,
        ref_model=None,
        pos_token_id: int,
        neg_token_id: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_beta = float(kl_beta)
        self.ref_model = ref_model
        self.pos_token_id = int(pos_token_id)
        self.neg_token_id = int(neg_token_id)
        if self.pos_token_id == self.neg_token_id:
            raise ValueError("pos_token_id and neg_token_id must differ.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        log_reward_pos = inputs.pop("log_reward_pos")
        log_reward_neg = inputs.pop("log_reward_neg")
        outputs = model(**inputs)
        policy_logits = outputs.logits

        with torch.no_grad():
            ref_logits = self.ref_model(**inputs).logits if self.ref_model is not None else None

        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("RLKLSFTTrainer requires labels with -100 masking on prompt tokens.")

        # Shift to next-token prediction. policy_logits[:, t] predicts token at t+1.
        policy_logits_shift = policy_logits[:, :-1, :]
        labels_shift = labels[:, 1:]
        mask = (labels_shift != -100)
        ref_logits_shift = ref_logits[:, :-1, :] if ref_logits is not None else None

        # 2-class restricted log-softmax over {pos_token_id, neg_token_id} at each
        # masked answer position.
        z_pos = policy_logits_shift[..., self.pos_token_id]
        z_neg = policy_logits_shift[..., self.neg_token_id]
        z_max = torch.maximum(z_pos, z_neg)
        denom = (z_pos - z_max).exp() + (z_neg - z_max).exp()
        log_q_pos = (z_pos - z_max) - denom.log()
        log_q_neg = (z_neg - z_max) - denom.log()
        q_pos = log_q_pos.exp()
        q_neg = log_q_neg.exp()

        # Broadcast per-example rewards to per-position. Shape: (B, T-1).
        # log_reward_pos / log_reward_neg are (B,).
        lr_pos = log_reward_pos.to(q_pos.dtype).unsqueeze(1).expand_as(q_pos)
        lr_neg = log_reward_neg.to(q_neg.dtype).unsqueeze(1).expand_as(q_neg)

        # -E_{y~q'}[log h_p(y|x)] at each masked position.
        neg_reward = -(q_pos * lr_pos + q_neg * lr_neg)

        # KL_2(q' || π_ref') on the 2-class restricted distribution.
        if self.kl_beta > 0 and ref_logits_shift is not None:
            r_pos = ref_logits_shift[..., self.pos_token_id]
            r_neg = ref_logits_shift[..., self.neg_token_id]
            r_max = torch.maximum(r_pos, r_neg)
            r_denom = (r_pos - r_max).exp() + (r_neg - r_max).exp()
            log_pref_pos = (r_pos - r_max) - r_denom.log()
            log_pref_neg = (r_neg - r_max) - r_denom.log()
            kl_per_pos = q_pos * (log_q_pos - log_pref_pos) + q_neg * (log_q_neg - log_pref_neg)
        else:
            kl_per_pos = torch.zeros_like(neg_reward)

        # Average over masked positions (treat each answer position as an
        # independent action). For single-token answers this is one position
        # per example.
        denom = mask.float().sum().clamp_min(1.0)
        neg_reward_mean = (neg_reward * mask.float()).sum() / denom
        kl_mean = (kl_per_pos * mask.float()).sum() / denom

        loss = neg_reward_mean + self.kl_beta * kl_mean
        self.log({
            "neg_reward": neg_reward_mean.detach().item(),
            "kl_2class": kl_mean.detach().item(),
        })
        return (loss, outputs) if return_outputs else loss
