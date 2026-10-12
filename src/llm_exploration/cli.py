from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_exploration.backend import (
    OpenAICompatibleBackend,
)
from llm_exploration.paprika import (
    SUPPORTED_GAMES,
    load_game_config,
)
from llm_exploration.runner import (
    GuessingGameRunner,
)

GAME_MENU = {
    "1": "twenty_questions",
    "2": "guess_my_city",
}

def build_backend(
    prefix: str,
    fallback: OpenAICompatibleBackend | None = None,
) -> OpenAICompatibleBackend:
    model = os.getenv(
        f"{prefix}_MODEL"
    )
    api_key = os.getenv(
        f"{prefix}_API_KEY"
    )
    base_url = os.getenv(
        f"{prefix}_BASE_URL"
    )
    reasoning_effort = os.getenv(
        f"{prefix}_REASONING_EFFORT"
    )

    if (
        model is None
        and fallback is not None
    ):
        model = fallback.model

    if (
        api_key is None
        and fallback is not None
    ):
        api_key = fallback.api_key

    if (
        base_url is None
        and fallback is not None
    ):
        base_url = fallback.base_url

    if (
        reasoning_effort is None
        and fallback is not None
    ):
        reasoning_effort = (
            fallback.reasoning_effort
        )

    if not model:
        raise RuntimeError(
            f"{prefix}_MODEL is not configured."
        )

    if not api_key:
        raise RuntimeError(
            f"{prefix}_API_KEY is not configured."
        )

    return OpenAICompatibleBackend(
        model=model,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )

def choose_game() -> str:
    print()
    print("Choose a game:")
    print("  1) Twenty Questions")
    print("  2) Guess My City")
    print()

    while True:
        choice = input(
            "Selection [1]: "
        ).strip()

        if choice == "":
            choice = "1"

        game = GAME_MENU.get(choice)

        if game is not None:
            return game

        print(
            "Please enter 1 or 2."
        )

def choose_runs(
    default: int = 20,
) -> int:
    while True:
        raw = input(
            f"Number of episodes [{default}]: "
        ).strip()

        if raw == "":
            return default

        try:
            runs = int(raw)
        except ValueError:
            print(
                "Please enter a whole number."
            )
            continue

        if runs <= 0:
            print(
                "Number of episodes must "
                "be greater than zero."
            )
            continue

        return runs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PAPRIKA guessing-game "
            "trajectories."
        )
    )

    parser.add_argument(
        "--game",
        choices=sorted(
            SUPPORTED_GAMES
        ),
        default=None,
        help=(
            "Game to run. If omitted, "
            "an interactive menu is shown."
        ),
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help=(
            "Starting scenario index."
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=(
            "Number of episodes to run."
        ),
    )

    parser.add_argument(
        "--repeat-same-scenario",
        action="store_true",
        help=(
            "Repeat the scenario at --index "
            "instead of moving through "
            "successive scenarios."
        ),
    )

    parser.add_argument(
        "--paprika-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser.parse_args()

def save_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    load_dotenv()

    args = parse_args()

    interactive = (
        args.game is None
    )

    if interactive:
        game = choose_game()
    else:
        game = args.game

    if args.runs is not None:
        runs = args.runs
    elif interactive:
        runs = choose_runs(
            default=20
        )
    else:
        runs = 1

    if runs <= 0:
        raise ValueError(
            "--runs must be greater than zero."
        )

    game_config = load_game_config(
        game,
        paprika_root=args.paprika_root,
    )

    scenarios = game_config.scenarios(
        args.split
    )

    if args.index < 0:
        raise IndexError(
            "Scenario index cannot be negative."
        )

    if args.repeat_same_scenario:
        required_last_index = args.index
    else:
        required_last_index = (
            args.index + runs - 1
        )

    if required_last_index >= len(
        scenarios
    ):
        raise IndexError(
            f"Requested scenario index "
            f"{required_last_index}, but "
            f"split {args.split!r} contains "
            f"only {len(scenarios)} scenarios."
        )

    agent = build_backend(
        "AGENT"
    )
    environment = build_backend(
        "ENV",
        fallback=agent,
    )
    judge = build_backend(
        "JUDGE",
        fallback=environment,
    )

    runner = GuessingGameRunner(
        agent=agent,
        environment=environment,
        judge=judge,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    if args.output_dir is None:
        output_dir = (
            Path("results")
            / game
            / (
                f"{timestamp}_"
                f"{args.split}_"
                f"n{runs}"
            )
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print()
    print("=" * 60)
    print(f"Game: {game}")
    print(f"Split: {args.split}")
    print(f"Episodes: {runs}")
    print(
        "Mode: "
        + (
            "repeat same scenario"
            if args.repeat_same_scenario
            else "successive scenarios"
        )
    )
    print(f"Output: {output_dir}")
    print("=" * 60)

    for run_number in range(
        1,
        runs + 1,
    ):
        if args.repeat_same_scenario:
            scenario_index = args.index
        else:
            scenario_index = (
                args.index
                + run_number
                - 1
            )

        scenario = scenarios[
            scenario_index
        ]

        print()
        print(
            f"[{run_number}/{runs}] "
            f"Scenario {scenario_index}"
        )
        print(
            f"Category: {scenario.agent}"
        )
        print(
            f"Hidden target: {scenario.env}"
        )

        result = runner.run(
            game_config=game_config,
            scenario=scenario,
            split=args.split,
            scenario_index=scenario_index,
        )

        episode_path = (
            output_dir
            / (
                f"episode_"
                f"{run_number:03d}_"
                f"scenario_"
                f"{scenario_index:03d}.json"
            )
        )

        save_json(
            episode_path,
            result.to_dict(),
        )

        results.append(result)

        print(
            f"Success: {result.success}"
        )
        print(
            f"Turns: "
            f"{result.num_turns}/"
            f"{result.max_turns}"
        )

    successful_results = [
        result
        for result in results
        if result.success
    ]

    success_count = len(
        successful_results
    )

    success_rate = (
        success_count
        / len(results)
    )

    avg_turns_all = (
        sum(
            result.num_turns
            for result in results
        )
        / len(results)
    )

    if successful_results:
        avg_turns_success = (
            sum(
                result.num_turns
                for result
                in successful_results
            )
            / len(
                successful_results
            )
        )
    else:
        avg_turns_success = None

    summary = {
        "game": game,
        "split": args.split,
        "runs": runs,
        "start_index": args.index,
        "repeat_same_scenario": (
            args.repeat_same_scenario
        ),
        "successes": success_count,
        "failures": (
            len(results)
            - success_count
        ),
        "success_rate": success_rate,
        "avg_turns_all": avg_turns_all,
        "avg_turns_success": (
            avg_turns_success
        ),
        "episodes": [
            {
                "run": index,
                "scenario_index": (
                    result.scenario_index
                ),
                "target": (
                    result.env_game_scenario
                ),
                "success": (
                    result.success
                ),
                "turns": (
                    result.num_turns
                ),
            }
            for index, result in enumerate(
                results,
                start=1,
            )
        ],
    }

    summary_path = (
        output_dir
        / "summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    print()
    print("=" * 60)
    print("Batch finished")
    print(
        f"Success: "
        f"{success_count}/{len(results)} "
        f"({success_rate:.1%})"
    )
    print(
        f"Average turns "
        f"(all episodes): "
        f"{avg_turns_all:.2f}"
    )

    if avg_turns_success is None:
        print(
            "Average turns "
            "(successful episodes): N/A"
        )
    else:
        print(
            f"Average turns "
            f"(successful episodes): "
            f"{avg_turns_success:.2f}"
        )
    print(
        f"Summary: {summary_path}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()