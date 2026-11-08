#!/bin/bash
#SBATCH --job-name=paprika-eval
#SBATCH --partition=batch
#SBATCH --nodelist=cloud-206
#SBATCH --time=02:00:00
#SBATCH --gpus=a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-110%1
#SBATCH --output=logs/paprika-eval-%A_%a.out
#SBATCH --error=logs/paprika-eval-%A_%a.err

set -euo pipefail

cd "$HOME/llm-exploration-thesis"

export PATH="$HOME/micromamba/envs/vllm-server/bin:$PATH"
export PYTHONUNBUFFERED=1

# All model/API settings are provided by this script.
export PYTHON_DOTENV_DISABLED=1
unset AGENT_REASONING_EFFORT
unset ENV_REASONING_EFFORT
unset JUDGE_REASONING_EFFORT

MODEL_PATH="$HOME/models/Qwen3.6-35B-A3B"
MODEL_NAME="Qwen/Qwen3.6-35B-A3B"

VLLM_BIN="$HOME/micromamba/envs/vllm-server/bin/vllm"
RESEARCH_PYTHON="$HOME/micromamba/envs/llm-exploration/bin/python"

PORT=$((18000 + SLURM_JOB_ID % 10000))

# Evaluation settings
CHUNK_SIZE=5
REPEATS=4

TWENTY_TOTAL=367
CITY_TOTAL=185

TWENTY_CHUNKS=$(( (TWENTY_TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE ))

if (( SLURM_ARRAY_TASK_ID < TWENTY_CHUNKS )); then
    GAME="twenty_questions"
    TOTAL="$TWENTY_TOTAL"
    CHUNK_INDEX="$SLURM_ARRAY_TASK_ID"
else
    GAME="guess_my_city"
    TOTAL="$CITY_TOTAL"
    CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID - TWENTY_CHUNKS))
fi

START=$((CHUNK_INDEX * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

if (( END > TOTAL )); then
    END="$TOTAL"
fi

echo "========================================"
echo "PAPRIKA Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Game: $GAME"
echo "Scenarios: $START to $((END - 1))"
echo "Repeats per scenario: $REPEATS"
echo "========================================"

nvidia-smi -L

# Start vLLM
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
    > "logs/vllm-eval-${SLURM_JOB_ID}.log" 2>&1 &

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
        echo "vLLM exited unexpectedly."
        tail -80 "logs/vllm-eval-${SLURM_JOB_ID}.log"
        exit 1
    fi

    sleep 10
done

if (( READY == 0 )); then
    echo "vLLM startup timed out."
    exit 1
fi

echo "vLLM ready."

export AGENT_MODEL="$MODEL_NAME"
export AGENT_BASE_URL="http://127.0.0.1:$PORT/v1"
export AGENT_API_KEY="EMPTY"

export ENV_MODEL="$MODEL_NAME"
export ENV_BASE_URL="$AGENT_BASE_URL"
export ENV_API_KEY="EMPTY"

export JUDGE_MODEL="$MODEL_NAME"
export JUDGE_BASE_URL="$AGENT_BASE_URL"
export JUDGE_API_KEY="EMPTY"

# Run every assigned scenario four times.
for (( INDEX=START; INDEX<END; INDEX++ )); do

    for (( ATTEMPT=1; ATTEMPT<=REPEATS; ATTEMPT++ )); do

        OUTPUT_DIR=$(printf \
            "results/paprika_eval/%s/scenario_%04d/attempt_%02d" \
            "$GAME" "$INDEX" "$ATTEMPT")

        # Preserve successfully completed attempts.
        if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
            echo "Skipping completed: $OUTPUT_DIR"
            continue
        fi

        echo "----------------------------------------"
        echo "Game: $GAME"
        echo "Scenario: $INDEX"
        echo "Attempt: $ATTEMPT/$REPEATS"
        echo "----------------------------------------"

        "$RESEARCH_PYTHON" -m llm_exploration.cli \
            --game "$GAME" \
            --split eval \
            --index "$INDEX" \
            --runs 1 \
            --prompt-mode paprika \
            --agent-max-tokens 128 \
            --output-dir "$OUTPUT_DIR"

    done

done

echo "Evaluation chunk completed."
