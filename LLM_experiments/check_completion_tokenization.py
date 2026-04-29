"""Verify which token the SFT gradient lands on, without running training.

Loads only the Qwen tokenizer (no model, no GPU). Builds a folktexts ACSIncome
prompt for a synthetic row, simulates TRL's completion_only_loss masking, and
compares the gradient-bearing token id with the answer-key token id folktexts
uses at inference.

If the two ids differ, training pushes a different token than scoring reads,
and we need to prepend a leading space to the completion in
`SFTPolicy._make_dataset`.

Usage:
    python -m LLM_experiments.check_completion_tokenization
    BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m LLM_experiments.check_completion_tokenization
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from transformers import AutoTokenizer

from . import _folktexts_compat  # noqa: F401  — must precede folktexts imports
from folktexts.acs.acs_tasks import ACSTaskMetadata
from folktexts.prompting import encode_row_prompt


def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"[check] tokenizer = {base_model}")

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    task = ACSTaskMetadata.get_task("ACSIncome")
    qa = task.multiple_choice_qa
    if qa is None:
        print("[check] task has no multiple_choice_qa", file=sys.stderr)
        sys.exit(1)

    # Build a synthetic row covering ACSIncome's required features.
    feature_defaults = {
        "AGEP": 35, "COW": 1, "SCHL": 16, "MAR": 1, "OCCP": 5700,
        "POBP": 6, "RELP": 0, "WKHP": 40, "SEX": 1, "RAC1P": 1,
    }
    row = pd.Series({k: feature_defaults.get(k, 0) for k in task.features})

    # Folktexts' encode_row for a row -> Q&A prompt string lives in
    # folktexts.prompting, not on TaskMetadata.
    prompt = encode_row_prompt(row, task)
    print("\n[check] === PROMPT ===")
    print(prompt)

    # Answer keys for the two labels.
    keys_by_label = {y: qa.get_answer_key_from_value(y) for y in (0, 1)}
    print("\n[check] === COMPLETION CANDIDATES ===")
    for y, k in keys_by_label.items():
        ids_no_space = tok.encode(k, add_special_tokens=False)
        ids_with_space = tok.encode(" " + k, add_special_tokens=False)
        print(f"  label={y}  key={k!r}")
        print(f"    encode({k!r})           -> ids={ids_no_space}  decoded={tok.decode(ids_no_space)!r}")
        print(f"    encode({' '+k!r})       -> ids={ids_with_space}  decoded={tok.decode(ids_with_space)!r}")

    # What folktexts looks up at inference: see qa_interface.py:_get_choice_token_id,
    # default prefix=" ".
    print("\n[check] === FOLKTEXTS INFERENCE TARGETS (default prefix=' ') ===")
    for y, k in keys_by_label.items():
        ids = tok.encode(" " + k, add_special_tokens=False)
        print(f"  label={y}  scoring_token_ids={ids}  decoded={tok.decode(ids)!r}")

    # Simulate TRL's completion_only_loss mask: tokenize prompt, then
    # prompt+completion, the labels for the prompt span are -100, the rest
    # carry gradient.
    print("\n[check] === TRL completion_only_loss SIMULATION ===")
    for y, k in keys_by_label.items():
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        # Two cases: completion = "A" vs completion = " A"
        for completion in (k, " " + k):
            full_ids = tok.encode(prompt + completion, add_special_tokens=False)
            n_prompt = len(prompt_ids)
            kept_ids = full_ids[n_prompt:]
            kept_decoded = tok.decode(kept_ids)
            inference_ids = tok.encode(" " + k, add_special_tokens=False)
            match = kept_ids == inference_ids
            verdict = "MATCH" if match else "MISMATCH"
            print(f"  label={y}  completion={completion!r}")
            print(f"    grad_target_ids={kept_ids}  decoded={kept_decoded!r}")
            print(f"    scoring_target_ids={inference_ids}  decoded={tok.decode(inference_ids)!r}")
            print(f"    -> {verdict}")

    print("\n[check] If MATCH is reported only when completion has a leading space,")
    print("[check] update SFTPolicy._make_dataset to use ' '+answer_key as completion.")


if __name__ == "__main__":
    main()
