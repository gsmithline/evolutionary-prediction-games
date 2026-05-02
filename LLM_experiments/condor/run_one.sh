#!/usr/bin/env bash
# Single-config runner for the Fig.5 LLM redo on HTCondor.
# Args: RUN_TAG POLICY BETA SEED
set -eo pipefail

RUN_TAG="$1"
POLICY="$2"
BETA="$3"
SEED="$4"

REPO="${REPO:-/home/gsmithline/evolutionary-prediction-games}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

T="${T:-600}"
N_PER_STEP="${N_PER_STEP:-1000}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-/is/cluster/gsmithline/folktexts_data}"
PI_REF_DIR="${PI_REF_DIR:-/is/cluster/gsmithline/folktexts_pi_ref}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/LLM_experiments/_results}"

echo "[run_one] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one] tag=$RUN_TAG policy=$POLICY beta=$BETA seed=$SEED T=$T n=$N_PER_STEP model=$BASE_MODEL"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi
export HF_HOME="${HF_HOME:-/is/cluster/gsmithline/.cache/huggingface}"

cd "$REPO"

env \
    RUN_TAG="$RUN_TAG" \
    POLICY="$POLICY" \
    BETA="$BETA" \
    SEED="$SEED" \
    T="$T" \
    N_PER_STEP="$N_PER_STEP" \
    N_TEST_PER_STATE="${N_TEST_PER_STATE:-5000}" \
    BASE_MODEL="$BASE_MODEL" \
    DATA_DIR="$DATA_DIR" \
    PI_REF_DIR="$PI_REF_DIR" \
    RESULTS_DIR="$RESULTS_DIR" \
    SFT_EPOCHS="${SFT_EPOCHS:-1}" \
    BATCH_SIZE="${BATCH_SIZE:-4}" \
    GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    OPTIMIZER="${OPTIMIZER:-adamw_torch}" \
    INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}" \
    PROMPT_FORMAT="${PROMPT_FORMAT:-flat}" \
    WANDB_PROJECT="${WANDB_PROJECT:-epg-llm-fig5}" \
    WANDB_DIR="${WANDB_DIR:-$REPO/wandb}" \
    python -m LLM_experiments.run_fig5
