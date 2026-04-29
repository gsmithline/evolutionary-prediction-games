"""Compatibility shims for folktexts on newer Qwen-family tokenizers.

Qwen2.5 (and other modern HF models) expose `model.config.vocab_size` larger
than `len(tokenizer.vocab)` because of added special tokens that live in the
embedding matrix but aren't in the tokenizer's `vocab` dict. Folktexts'
`query_model_batch_multiple_passes` builds an "allowed tokens" boolean mask
sized to `len(tokenizer.vocab)` and then indexes the model logits with it,
causing:

    IndexError: boolean index did not match indexed array along axis 1;
                size of axis is 151936 but size of corresponding boolean axis is 151665

Importing this module patches both the original location in `folktexts.llm_utils`
and the already-bound copy in `folktexts.classifier.transformers_classifier`.

The patched function truncates `current_probs` to `len(tokenizer.vocab)` before
applying the answer-token filter; the dropped tail entries correspond to
special tokens whose probability mass we don't want surfacing as answers
anyway.
"""
from __future__ import annotations

import numpy as np

import folktexts.llm_utils as _flu
import folktexts.classifier.transformers_classifier as _tc


def _patched_query_model_batch_multiple_passes(
    text_inputs,
    model,
    tokenizer,
    context_size,
    n_passes,
    digits_only: bool = False,
) -> np.ndarray:
    vocab_n = len(tokenizer.vocab)
    allowed_tokens_filter = np.ones(vocab_n, dtype=bool)
    if digits_only:
        allowed_token_ids = np.array([
            tok_id
            for token, tok_id in tokenizer.vocab.items() if token.isdecimal()
        ])
        allowed_tokens_filter = np.zeros(vocab_n, dtype=bool)
        allowed_tokens_filter[allowed_token_ids] = True

    current_batch = text_inputs
    last_token_probs = []
    for _ in range(n_passes):
        current_probs = _flu.query_model_batch(current_batch, model, tokenizer, context_size)
        # Drop the (special-token) tail so the filter and probs match.
        if current_probs.shape[1] > vocab_n:
            current_probs = current_probs[:, :vocab_n]
        current_probs[:, ~allowed_tokens_filter] = 0
        next_tokens = [tokenizer.decode([int(np.argmax(probs))]) for probs in current_probs]
        current_batch = [t + nt for t, nt in zip(current_batch, next_tokens)]
        last_token_probs.append(current_probs)

    arr = np.array(last_token_probs)
    arr = np.moveaxis(arr, 0, 1)
    assert arr.shape == (len(text_inputs), n_passes, vocab_n)
    return arr


# Patch both the canonical location and the already-bound import.
_flu.query_model_batch_multiple_passes = _patched_query_model_batch_multiple_passes
_tc.query_model_batch_multiple_passes = _patched_query_model_batch_multiple_passes
