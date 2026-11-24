from __future__ import annotations

import argparse
import hashlib
import json
import os

from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import dspy
from dotenv import load_dotenv

from llm_exploration.cli import build_backend
from llm_exploration.paprika import load_game_config
from llm_exploration.runner import GuessingGameRunner
from llm_exploration.gepa import (
    DspyAgentBackend,
    GepaEpisode,
    EpisodeFeedback,
    normalize,
    make_lm,
)


MODES = ("baseline", "gepa")

AGENT_TEMPERATURE = 0.0
AGENT_MAX_TOKENS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and GEPA-optimized DSPy agents"
    )

    parser.add_argument(
        "--game",
        choices=("twenty_questions", "guess_my_city"),
        default="twenty_questions",
    )

    parser.add_argument(
        "--program",
        type=Path,
        default=None,
        help="Path to optimized_program.json",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Aggregate existing evaluation results without running an LLM",
    )

    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def prompt_hash(instructions: str) -> str:
    return hashlib.sha256(
        instructions.encode("utf-8")
    ).hexdigest()[:12]


def episode_path(
    output_dir: Path,
    mode: str,
    scenario_index: int,
    attempt: int,
) -> Path:
    return (
        output_dir
        / mode
        / f"scenario_{scenario_index:04d}"
        / f"attempt_{attempt:02d}"
        / "episode.json"
    )


def calculate_metrics(
    result,
    metric: EpisodeFeedback,
) -> dict:

    questions = [
        message["content"]
        for message in result.conversation
        if message["role"] == "assistant"
    ]

    raw_answers = [
        message["content"]
        for message in result.env_conversation
        if message["role"] == "assistant"
    ]

    extracted_answers = [
        message["content"]
        for message in result.conversation[1:]
        if message["role"] == "user"
    ]

    invalid_turns = []
    repeated_pairs = []
    first_seen = {}

    for turn, question in enumerate(
        questions,
        start=1,
    ):
        valid = (
            question.count("?") == 1
            and question.strip().endswith("?")
        )

        if not valid:
            invalid_turns.append(turn)

        key = normalize(question)

        if key in first_seen:
            repeated_pairs.append({
                "current_turn": turn,
                "original_turn": first_seen[key],
            })
        else:
            first_seen[key] = turn

    # Environment's own goal signal, separated
    # from the combined Runner goal decision.
    environment_goal = any(
        "goal reached" in answer.lower()
        for answer in extracted_answers
    )

    final_judge_label = (
        result.judge_label
        if result.judge_conversation is not None
        else None
    )

    # Auxiliary GEPA-style reward.
    # The main benchmark metric remains result.success.
    prediction = dspy.Prediction(
        questions=questions,
        environment_answers=raw_answers,
        environment_goal=environment_goal,
        judge_label=final_judge_label,
        num_turns=result.num_turns,
    )

    gold = dspy.Example(
        scenario_index=result.scenario_index,
    ).with_inputs("scenario_index")

    feedback = metric(
        gold,
        prediction,
    )

    return {
        "success": bool(result.success),

        "goal_reached": bool(result.goal_reached),
        "environment_goal": environment_goal,
        "final_judge_label": final_judge_label,

        "num_turns": result.num_turns,
        "max_turns": result.max_turns,
        "num_questions": len(questions),

        "invalid_count": len(invalid_turns),
        "invalid_turns": invalid_turns,

        "repeat_count": len(repeated_pairs),
        "repeated_pairs": repeated_pairs,

        "gepa_style_reward": float(feedback.score),
        "feedback": feedback.feedback,
    }


def summarize_results(
    args: argparse.Namespace,
    indices: list[int],
) -> None:

    summary = {
        "game": args.game,
        "split": "eval",
        "num_scenarios": len(indices),
        "repeats": args.repeats,
        "expected_episodes_per_mode": (
            len(indices) * args.repeats
        ),
        "modes": {},
    }

    reference_settings = None

    for mode in MODES:

        records = []
        complete_scenarios = 0
        passed_scenarios = 0

        for index in indices:

            scenario_records = []

            for attempt in range(
                1,
                args.repeats + 1,
            ):
                path = episode_path(
                    args.output_dir,
                    mode,
                    index,
                    attempt,
                )

                if not path.exists():
                    continue

                record = read_json(path)

                if (
                    record["mode"] != mode
                    or record["scenario_index"] != index
                    or record["attempt"] != attempt
                    or record["settings"]["game"] != args.game
                ):
                    raise RuntimeError(
                        f"Unexpected episode metadata: {path}"
                    )

                if reference_settings is None:
                    reference_settings = record["settings"]

                elif record["settings"] != reference_settings:
                    raise RuntimeError(
                        f"Inconsistent experiment settings: {path}"
                    )

                scenario_records.append(record)
                records.append(record)

            # Pass@k is calculated only for scenarios
            # with all required attempts available.
            if len(scenario_records) == args.repeats:

                complete_scenarios += 1

                if any(
                    record["metrics"]["success"]
                    for record in scenario_records
                ):
                    passed_scenarios += 1

        metrics = [
            record["metrics"]
            for record in records
        ]

        successful = [
            item
            for item in metrics
            if item["success"]
        ]

        total = len(metrics)
        successes = len(successful)

        total_questions = sum(
            item["num_questions"]
            for item in metrics
        )

        total_repeats = sum(
            item["repeat_count"]
            for item in metrics
        )

        total_invalid = sum(
            item["invalid_count"]
            for item in metrics
        )

        summary["modes"][mode] = {
            "completed_episodes": total,

            "missing_episodes": (
                len(indices) * args.repeats - total
            ),

            "successes": successes,
            "failures": total - successes,

            "success_rate": (
                successes / total
                if total else None
            ),

            "avg_turns_all": (
                sum(
                    item["num_turns"]
                    for item in metrics
                ) / total
                if total else None
            ),

            "avg_turns_success": (
                sum(
                    item["num_turns"]
                    for item in successful
                ) / successes
                if successes else None
            ),

            "complete_scenarios": complete_scenarios,

            f"pass_at_{args.repeats}": (
                passed_scenarios / complete_scenarios
                if complete_scenarios else None
            ),

            "total_questions": total_questions,
            "total_repeated_questions": total_repeats,
            "total_invalid_questions": total_invalid,

            "repeat_rate": (
                total_repeats / total_questions
                if total_questions else None
            ),

            "invalid_rate": (
                total_invalid / total_questions
                if total_questions else None
            ),

            "avg_gepa_style_reward": (
                sum(
                    item["gepa_style_reward"]
                    for item in metrics
                ) / total
                if total else None
            ),
        }

    summary["settings"] = reference_settings

    summary["complete"] = all(
        item["missing_episodes"] == 0
        for item in summary["modes"].values()
    )

    path = args.output_dir / "summary.json"

    save_json(path, summary)

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"Summary saved: {path}")


def evaluate(
    args: argparse.Namespace,
    config,
    scenarios,
    indices: list[int],
) -> None:

    if args.program is None:
        raise ValueError(
            "--program is required for evaluation"
        )

    if not args.program.is_file():
        raise FileNotFoundError(args.program)

    optimization_config = read_json(
        args.program.parent / "run_config.json"
    )

    if optimization_config["game"] != args.game:
        raise ValueError(
            "The optimized program belongs to another game"
        )

    if optimization_config["prompt_mode"] != config.prompt_mode:
        raise ValueError("Prompt mode mismatch")

    if optimization_config["max_turns"] != config.max_turns:
        raise ValueError("Maximum turns mismatch")

    if "{env}" in config.data["agent"]:
        raise ValueError(
            "The agent prompt must not contain the hidden target"
        )

    dspy.configure(
        lm=make_lm(
            "AGENT",
            max_tokens=AGENT_MAX_TOKENS,
            temperature=AGENT_TEMPERATURE,
        )
    )

    metric = EpisodeFeedback(
        args.game,
        scenarios,
        config.max_turns,
    )

    with ExitStack() as stack:

        agent_backend = build_backend("AGENT")
        stack.callback(agent_backend.client.close)

        environment = build_backend(
            "ENV",
            fallback=agent_backend,
        )
        stack.callback(environment.client.close)

        judge = build_backend(
            "JUDGE",
            fallback=environment,
        )
        stack.callback(judge.client.close)

        if optimization_config["agent_model"] != agent_backend.model:
            raise ValueError("Agent model mismatch")

        if optimization_config["environment_model"] != environment.model:
            raise ValueError("Environment model mismatch")

        programs = {}

        for mode in MODES:

            program = GepaEpisode(
                config,
                scenarios,
                environment,
                judge,
                use_judge=True,
                episode_dir=args.output_dir / "_unused",
                val_indices=[],
            )

            if mode == "gepa":
                program.load(str(args.program))

            # Both programs use the same globally configured LM.
            program.ask.lm = None

            programs[mode] = program

        instructions = {
            mode: program.ask.signature.instructions
            for mode, program in programs.items()
        }

        settings = {
            "game": args.game,
            "split": "eval",
            "prompt_mode": config.prompt_mode,

            "agent_model": agent_backend.model,
            "environment_model": environment.model,
            "judge_model": judge.model,

            "agent_temperature": AGENT_TEMPERATURE,
            "agent_max_tokens": AGENT_MAX_TOKENS,

            "environment_temperature": 0.0,
            "environment_max_tokens": 1024,

            "judge_temperature": 0.0,
            "judge_max_tokens": 128,

            "judge_used": True,
            "max_turns": config.max_turns,

            "vllm_max_model_len": os.getenv(
                "VLLM_MAX_MODEL_LEN"
            ),

            "dspy_version": dspy.__version__,

            "optimized_program_sha256": hashlib.sha256(
                args.program.read_bytes()
            ).hexdigest(),

            "candidate_hashes": {
                mode: prompt_hash(text)
                for mode, text in instructions.items()
            },
        }

        print("=" * 60)
        print(f"Game: {args.game}")
        print("Split: eval")
        print(f"Scenarios: {indices}")
        print(f"Repeats: {args.repeats}")

        for mode in MODES:
            print()
            print(f"Mode: {mode}")
            print(
                f"Candidate hash: "
                f"{settings['candidate_hashes'][mode]}"
            )
            print(instructions[mode])

        if instructions["baseline"] == instructions["gepa"]:
            raise RuntimeError(
                "Baseline and optimized instructions are identical"
            )

        print("=" * 60)

        for index in indices:

            scenario = scenarios[index]

            for attempt in range(
                1,
                args.repeats + 1,
            ):

                # Alternate execution order to reduce
                # systematic ordering effects.
                modes = (
                    MODES
                    if (index + attempt) % 2 == 0
                    else MODES[::-1]
                )

                for mode in modes:

                    path = episode_path(
                        args.output_dir,
                        mode,
                        index,
                        attempt,
                    )

                    if path.exists():

                        existing = read_json(path)

                        if (
                            existing["settings"] != settings
                            or existing["mode"] != mode
                            or existing["scenario_index"] != index
                            or existing["attempt"] != attempt
                        ):
                            raise RuntimeError(
                                f"Existing result has incompatible settings: {path}"
                            )

                        print(
                            f"Skipping completed: {path}"
                        )
                        continue

                    agent = DspyAgentBackend(
                        programs[mode].ask,
                        config.agent_prompt(scenario),
                    )

                    runner = GuessingGameRunner(
                        agent=agent,
                        environment=environment,
                        judge=judge,

                        agent_temperature=AGENT_TEMPERATURE,
                        agent_max_tokens=AGENT_MAX_TOKENS,

                        use_judge=True,
                    )

                    print()
                    print("-" * 60)
                    print(f"Mode: {mode}")
                    print(f"Scenario: {index}")
                    print(f"Target: {scenario.env}")
                    print(f"Attempt: {attempt}/{args.repeats}")

                    result = runner.run(
                        game_config=config,
                        scenario=scenario,
                        split="eval",
                        scenario_index=index,
                    )

                    metrics = calculate_metrics(
                        result,
                        metric,
                    )

                    episode_data = result.to_dict()

                    # The Runner uses a default True value
                    # when no final Judge verification occurs.
                    episode_data["judge_label"] = (
                        metrics["final_judge_label"]
                    )

                    record = {
                        "timestamp_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),

                        "mode": mode,
                        "scenario_index": index,
                        "attempt": attempt,

                        "target": scenario.env,
                        "category": scenario.agent,

                        "candidate_hash": (
                            settings["candidate_hashes"][mode]
                        ),

                        "candidate_instructions": instructions[mode],

                        "settings": settings,

                        "episode": episode_data,
                        "metrics": metrics,
                    }

                    save_json(path, record)

                    print(
                        f"Success: {metrics['success']}"
                    )

                    print(
                        f"Turns: "
                        f"{metrics['num_turns']}/"
                        f"{metrics['max_turns']}"
                    )

                    print(
                        f"Reward: "
                        f"{metrics['gepa_style_reward']:.4f}"
                    )

                    print(f"Saved: {path}")

    print("DSPy evaluation chunk completed.")


def main() -> None:

    load_dotenv()

    args = parse_args()

    if (
        args.start_index < 0
        or args.num_scenarios <= 0
        or args.repeats <= 0
    ):
        raise ValueError("Invalid evaluation arguments")

    config = load_game_config(
        args.game,
        prompt_mode="paprika",
    )

    scenarios = config.scenarios("eval")

    end_index = (
        args.start_index + args.num_scenarios
    )

    if end_index > len(scenarios):
        raise IndexError(
            f"Requested {end_index} scenarios, "
            f"but eval split contains {len(scenarios)}"
        )

    indices = list(
        range(args.start_index, end_index)
    )

    if args.summary_only:

        summarize_results(
            args,
            indices,
        )

    else:

        evaluate(
            args,
            config,
            scenarios,
            indices,
        )


if __name__ == "__main__":
    main()