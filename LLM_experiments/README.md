# LLM redo of Saig & Rosenfeld (2025) Fig. 5

Re-runs the ACSIncome 3-state replicator experiment with an LLM in place of the
linear SVM, plus two cheaper baselines for comparison.

## Conditions

| name        | classifier at each replicator step                                                          |
| ----------- | ------------------------------------------------------------------------------------------- |
| `static`    | frozen base LLM, threshold τ tuned on the population-weighted sample                        |
| `closed_form` | π_β,p\*(y\|x) ∝ π_ref(y\|x) · p_target(y\|x;p)^(1/β); p_target = small head fit on the sample |
| `sft` (default = continual) | full fine-tune with β·KL anchor toward π_ref; weights persist across rounds. Mendler-Dünner et al. 2020 framing, deployed-system semantics |
| `sft` + `SFT_FRESH=1` | reset to π_ref at the top of each `fit()` (RRM ablation; matches S&R Markov-in-p theorem setup) |

`static` is the deployed-model baseline (no retraining feedback loop on the model
itself). `closed_form` is the analytic optimum under oracle SFT — the form of
π\_β,p\* that the planned theorems are about. `sft` is what you actually get when
the inner loop is real KL-regularized supervised fine-tuning with finite data.

## Data

ACSIncome from folktexts (which wraps folktables + ACS PUMS). K=3 groups by
state: California, New York, Texas, 2018 PUMS. Same n=1000 per replicator
step and 5000 held-out per state as S&R §6.3.

## Layout

```
LLM_experiments/
├── data.py            # ACSIncome loader + mixture sampler
├── llm_pi_ref.py      # one-shot cache of base-LLM π_ref logprobs
├── policies.py        # StaticThresholdPolicy / ClosedFormPolicy / SFTPolicy
├── replicator.py      # discrete-replicator loop, K=3
├── run_fig5.py        # env-var-driven entry point
└── condor/
    ├── run_one.sh
    ├── fig5.sub
    ├── configs_fig5_static.txt
    ├── configs_fig5_closedform.txt
    └── configs_fig5_sft.txt
```

## Running

Local smoke from the EPG repo root (CPU OK for `static`; closed-form needs the π_ref cache; SFT needs a GPU):

```
POLICY=static T=50 SEED=0 python -m LLM_experiments.run_fig5
POLICY=closed_form BETA=1.0 T=50 SEED=0 python -m LLM_experiments.run_fig5
POLICY=sft BETA=1.0 T=20 SEED=0 BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m LLM_experiments.run_fig5
```

Cluster (HTCondor, MPI-IS Tübingen):

```
condor_submit_bid 15 LLM_experiments/condor/fig5.sub  # picks policy/beta/seed from configs file
```

## Logging

All runs log to Weights & Biases when `WANDB_API_KEY` is set (the Condor sub
file picks it up from `~/.wandb_key`, mirroring the Opinion-dynamics-post-training
sweep). Project defaults to `epg-llm-fig5`; override with `WANDB_PROJECT` /
`WANDB_ENTITY`. To disable, set `WANDB_MODE=disabled`.

Per replicator step (`step_metric=t`):
- `p/{CA,NY,TX}`, `p/min`, `p/entropy` — population trajectory.
- `acc/{CA,NY,TX}`, `acc/overall` — group + population-weighted accuracy.
- `fitness/{CA,NY,TX}` — replicator fitness (= accuracy here).
- `sample_count/{CA,NY,TX}`, `sample_y_mean/{CA,NY,TX}` — per-state draw size and pos-rate.
- `step_seconds` — wall time per step.
- `threshold` — for `static`, the per-step τ tuned on the sample.

For `sft`, the TRL trainer also logs `ce_loss`, `kl`, `loss`, `learning_rate`
per-batch. Set the run name to `RUN_TAG` so β-sweeps are easy to filter.

## Figures

`make_fig5.py` reproduces the S&R Fig.5 three-panel layout (ternary trajectory +
composition + fitness) from one trajectory CSV. Mirrors the EPG repo's
`notebook_env` + `evoml.analysis` + `mpltern` styling.

```
# one figure per (policy, β, seed) trajectory:
python -m LLM_experiments.make_fig5 LLM_experiments/_results/fig5_sft_b1_s0.csv
# batch:
for f in LLM_experiments/_results/fig5_sft_*.csv; do
    python -m LLM_experiments.make_fig5 "$f"
done
```

Output is a PDF next to the input CSV by default. β-overlay across multiple
trajectories is a separate (later) script.

## Compute budget (0.5B Qwen, A100 24GB)

| condition    | per (β, seed) trajectory | sweep total                           |
| ------------ | ------------------------ | ------------------------------------- |
| static       | minutes                  | < 1 GPU-hour total                    |
| closed_form  | minutes                  | < 2 GPU-hours total                   |
| sft          | ~5 GPU-hours at T=600    | ~120 GPU-hours for full β-grid × 3 reps |

7B is a follow-up; same pipeline, FSDP/8bit-Adam, request_gpus=1 with 75GB.
