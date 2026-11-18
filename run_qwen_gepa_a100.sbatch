#!/bin/bash
#SBATCH --job-name=qwen36-gepa
#SBATCH --partition=batch
#SBATCH --nodelist=cloud-206
#SBATCH --time=02:00:00
#SBATCH --gpus=a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/gepa-%j.out
#SBATCH --error=logs/gepa-%j.err

set -euo pipefail

cd "$HOME/llm-exploration-thesis"

export PATH="$HOME/micromamba/envs/vllm-server/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHON_DOTENV_DISABLED=1

unset AGENT_REASONING_EFFORT
unset ENV_REASONING_EFFORT
unset JUDGE_REASONING_EFFORT
unset REFLECTION_REASONING_EFFORT

MODEL_PATH="$HOME/models/Qwen3.6-35B-A3B"
MODEL_NAME="Qwen/Qwen3.6-35B-A3B"

VLLM_BIN="$HOME/micromamba/envs/vllm-server/bin/vllm"
RESEARCH_PYTHON="$HOME/micromamba/envs/llm-exploration/bin/python"

PORT=$((18000 + SLURM_JOB_ID % 10000))

echo "===================================="
echo "DSPy GEPA Optimization"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "===================================="

nvidia-smi -L

echo "Starting vLLM..."

"$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --language-model-only \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "logs/vllm-gepa-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "Waiting for vLLM..."

READY=0

for i in $(seq 1 180); do

    if curl -fsS \
        "http://127.0.0.1:$PORT/v1/models" \
        >/dev/null 2>&1; then

        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM terminated."
        tail -80 "logs/vllm-gepa-${SLURM_JOB_ID}.log"
        exit 1
    fi

    sleep 10
done

if (( READY == 0 )); then
    echo "vLLM startup timeout."
    exit 1
fi

echo "vLLM ready."

API_URL="http://127.0.0.1:$PORT/v1"

# Agent
export AGENT_MODEL="$MODEL_NAME"
export AGENT_BASE_URL="$API_URL"
export AGENT_API_KEY="EMPTY"

# Environment
export ENV_MODEL="$MODEL_NAME"
export ENV_BASE_URL="$API_URL"
export ENV_API_KEY="EMPTY"

# Judge
export JUDGE_MODEL="$MODEL_NAME"
export JUDGE_BASE_URL="$API_URL"
export JUDGE_API_KEY="EMPTY"

# Reflection model for GEPA
export REFLECTION_MODEL="$MODEL_NAME"
export REFLECTION_BASE_URL="$API_URL"
export REFLECTION_API_KEY="EMPTY"

echo "Starting GEPA optimization..."

"$RESEARCH_PYTHON" -m llm_exploration.gepa \
    --game twenty_questions \
    --prompt-mode paprika \
    --train-size 12 \
    --val-size 6 \
    --max-turns 20 \
    --max-metric-calls 120 \
    --seed 42 \
    --output-dir "results/gepa/twenty_questions/test_${SLURM_JOB_ID}"

echo "GEPA optimization finished."