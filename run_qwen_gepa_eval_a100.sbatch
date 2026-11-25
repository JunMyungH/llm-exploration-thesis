#!/bin/bash
#SBATCH --job-name=dspy-eval
#SBATCH --partition=batch
#SBATCH --nodelist=cloud-206
#SBATCH --time=02:00:00
#SBATCH --gpus=a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-1%1
#SBATCH --output=logs/dspy-eval-%A_%a.out
#SBATCH --error=logs/dspy-eval-%A_%a.err

set -euo pipefail

cd "$HOME/llm-exploration-thesis"

export PATH="$HOME/micromamba/envs/vllm-server/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
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

# ----------------------------------------
# Evaluation configuration
# ----------------------------------------

GAME="twenty_questions"

OPTIMIZED_PROGRAM="results/gepa/twenty_questions/test_39873/optimized_program.json"

PROFILE="${1:-smoke}"

CHUNK_SIZE=10

case "$PROFILE" in

    smoke)
        TOTAL=20
        REPEATS=2
        ;;

    full)
        TOTAL=367
        REPEATS=4
        ;;

    *)
        echo "Unknown evaluation profile: $PROFILE"
        exit 1
        ;;
esac

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

if (( START >= TOTAL )); then
    echo "No scenarios assigned to this task."
    exit 0
fi

if (( END > TOTAL )); then
    END="$TOTAL"
fi

NUM_SCENARIOS=$((END - START))

OUTPUT_DIR="results/dspy_eval/$GAME/pilot_39873/temp_07/$PROFILE"

export VLLM_MAX_MODEL_LEN=8192

if [[ ! -f "$OPTIMIZED_PROGRAM" ]]; then
    echo "Optimized program not found:"
    echo "$OPTIMIZED_PROGRAM"
    exit 1
fi

echo "========================================"
echo "DSPy Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Profile: $PROFILE"
echo "Game: $GAME"
echo "Scenarios: $START to $((END - 1))"
echo "Repeats: $REPEATS"
echo "Output: $OUTPUT_DIR"
echo "========================================"

nvidia-smi -L

# ----------------------------------------
# Start vLLM
# ----------------------------------------

"$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 \
    --language-model-only \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "logs/vllm-dspy-eval-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "Waiting for vLLM..."

READY=0

for i in $(seq 1 180); do

    if curl -fsS --max-time 3 \
        "http://127.0.0.1:$PORT/v1/models" \
        >/dev/null 2>&1; then

        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then

        echo "vLLM terminated unexpectedly."

        tail -80 \
            "logs/vllm-dspy-eval-${SLURM_JOB_ID}.log"

        exit 1
    fi

    sleep 10
done

if (( READY == 0 )); then

    echo "vLLM startup timeout."

    exit 1
fi

echo "vLLM ready."

# ----------------------------------------
# Model settings
# ----------------------------------------

API_URL="http://127.0.0.1:$PORT/v1"

export AGENT_MODEL="$MODEL_NAME"
export AGENT_BASE_URL="$API_URL"
export AGENT_API_KEY="EMPTY"

export ENV_MODEL="$MODEL_NAME"
export ENV_BASE_URL="$API_URL"
export ENV_API_KEY="EMPTY"

export JUDGE_MODEL="$MODEL_NAME"
export JUDGE_BASE_URL="$API_URL"
export JUDGE_API_KEY="EMPTY"

# ----------------------------------------
# Evaluate both DSPy programs
# ----------------------------------------

"$RESEARCH_PYTHON" -m llm_exploration.eval_dspy \
    --game "$GAME" \
    --program "$OPTIMIZED_PROGRAM" \
    --start-index "$START" \
    --num-scenarios "$NUM_SCENARIOS" \
    --repeats "$REPEATS" \
    --output-dir "$OUTPUT_DIR"

echo "Evaluation chunk completed."