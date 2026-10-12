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


def build_backend(
    prefix: str,
    fallback: OpenAICompatibleBackend | None = None,
) -> OpenAICompatibleBackend:
    model = os.getenv(f"{prefix}_MODEL")
    api_key = os.getenv(f"{prefix}_API_KEY")
    base_url = os.getenv(f"{prefix}_BASE_URL")
    reasoning_effort = os.getenv(
        f"{prefix}_REASONING_EFFORT"
    )

    if model is None and fallback is not None:
        model = fallback.model

    if api_key is None and fallback is not None:
        api_key = fallback.api_key

    if base_url is None and fallback is not None:
        base_url = fallback.base_url

    if reasoning_effort is None and fallback is not None:
        reasoning_effort = fallback.reasoning_effort

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one PAPRIKA guessing-game "
            "trajectory."
        )
    )

    parser.add_argument(
        "--game",
        choices=sorted(SUPPORTED_GAMES),
        default="twenty_questions",
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--paprika-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()

    game_config = load_game_config(
        args.game,
        paprika_root=args.paprika_root,
    )

    scenarios = game_config.scenarios(
        args.split
    )

    if args.index < 0 or args.index >= len(
        scenarios
    ):
        raise IndexError(
            f"Scenario index {args.index} is "
            f"outside [0, {len(scenarios) - 1}]."
        )

    scenario = scenarios[args.index]

    agent = build_backend("AGENT")
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

    print(
        f"Game: {args.game}"
    )
    print(
        f"Split: {args.split}"
    )
    print(
        f"Scenario index: {args.index}"
    )
    print(
        f"Agent category: {scenario.agent}"
    )
    print(
        f"Hidden target: {scenario.env}"
    )
    print()

    result = runner.run(
        game_config=game_config,
        scenario=scenario,
        split=args.split,
        scenario_index=args.index,
    )

    if args.output is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output = (
            Path("results")
            / args.game
            / (
                f"{timestamp}_"
                f"{args.split}_"
                f"{args.index}.json"
            )
        )
    else:
        output = args.output

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.to_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Episode finished")
    print(
        f"Success: {result.success}"
    )
    print(
        f"Turns: "
        f"{result.num_turns}/"
        f"{result.max_turns}"
    )
    print(
        f"Result: {output}"
    )


if __name__ == "__main__":
    main()
