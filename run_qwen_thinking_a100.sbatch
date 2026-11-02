#!/bin/bash
#SBATCH --job-name=qwen36-thinking
#SBATCH --partition=batch
#SBATCH --nodelist=cloud-206
#SBATCH --time=02:00:00
#SBATCH --gpus=a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/qwen-%j.out
#SBATCH --error=logs/qwen-%j.err

set -euo pipefail

# Make executables from vllm-server available to worker processes.
export PATH="$HOME/micromamba/envs/vllm-server/bin:$PATH"

cd "$HOME/llm-exploration-thesis"

# Check build tools before starting vLLM.
echo "Checking Ninja..."
command -v ninja
ninja --version

MODEL_PATH="$HOME/models/Qwen3.6-35B-A3B"
MODEL_NAME="Qwen/Qwen3.6-35B-A3B"

PORT=18000

echo "===================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "===================================="

nvidia-smi -L

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Model files not found: $MODEL_PATH"
    exit 1
fi

echo "Starting vLLM..."

VLLM_BIN="$HOME/micromamba/envs/vllm-server/bin/vllm"

if [ ! -x "$VLLM_BIN" ]; then
    echo "ERROR: vLLM executable not found: $VLLM_BIN"
    exit 1
fi

"$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --language-model-only \
  --reasoning-parser qwen3 \
  > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &

SERVER_PID=$!

trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "Waiting for API server..."

READY=0

for i in $(seq 1 180); do

    if curl -fsS \
      "http://127.0.0.1:$PORT/v1/models" \
      > /dev/null 2>&1; then

        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM process terminated."
        tail -80 "logs/vllm-$SLURM_JOB_ID.log"
        exit 1
    fi

    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "vLLM server did not become ready."
    tail -80 "logs/vllm-$SLURM_JOB_ID.log"
    exit 1
fi

echo "vLLM API is ready."

export AGENT_MODEL="$MODEL_NAME"
export AGENT_BASE_URL="http://127.0.0.1:$PORT/v1"
export AGENT_API_KEY="EMPTY"

export ENV_MODEL="$MODEL_NAME"
export ENV_BASE_URL="$AGENT_BASE_URL"
export ENV_API_KEY="EMPTY"

export JUDGE_MODEL="$MODEL_NAME"
export JUDGE_BASE_URL="$AGENT_BASE_URL"
export JUDGE_API_KEY="EMPTY"

# Experiment B: Agent Thinking ON
export AGENT_REASONING_EFFORT="high"

# Keep Environment and Judge unchanged
export ENV_REASONING_EFFORT="none"
export JUDGE_REASONING_EFFORT="none"

echo "===================================="
echo "Experiment B: PAPRIKA Original"
echo "Agent Thinking: ON"
echo "Environment Thinking: OFF"
echo "Judge Thinking: OFF"
echo "===================================="

echo "Testing Qwen response..."

curl -fsS \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one sentence."
      }
    ],
    "max_tokens": 64,
    "temperature": 0
  }'

echo
echo "Starting Twenty Questions..."

RESEARCH_PYTHON="$HOME/micromamba/envs/llm-exploration/bin/python"

if [ ! -x "$RESEARCH_PYTHON" ]; then
    echo "ERROR: Research Python not found: $RESEARCH_PYTHON"
    exit 1
fi

echo "Starting Experiment B..."

"$RESEARCH_PYTHON" -m llm_exploration.cli \
  --game twenty_questions \
  --split train \
  --index 0 \
  --runs 1 \
  --prompt-mode paprika \
  --agent-max-tokens 2048 \
  --output-dir "results/twenty_questions/qwen36_thinking_on_${SLURM_JOB_ID}"

echo "Test finished."
