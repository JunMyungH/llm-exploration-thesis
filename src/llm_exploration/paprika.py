from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_GAMES = {
    "twenty_questions": "twenty_questions.json",
    "guess_my_city": "guess_my_city.json",
}

TWENTY_QUESTIONS_AGENT_PROTOCOL = """
IMPORTANT INTERACTION RULE:
- Output exactly ONE yes-or-no question per turn.
- Do not answer your own question.
- Do not simulate the environment or future turns.
- Do not output multiple questions.
- Do not use labels such as "Question:" or "Answer:".
- Stop immediately after the single question.
- If making a final guess, phrase it as one yes-or-no question.
""".strip()

@dataclass(frozen=True)
class Scenario:
    env: str
    agent: str

class PaprikaGameConfig:
    def __init__(self, name: str, data: dict[str, Any]) -> None:
        self.name = name
        self.data = data

    @property
    def max_turns(self) -> int:
        return int(self.data.get("max_turns", 20))

    def scenarios(self, split: str) -> list[Scenario]:
        if split not in self.data:
            available = [
                key
                for key, value in self.data.items()
                if isinstance(value, list)
            ]
            raise ValueError(
                f"Splict {split!r} does not exist. "
                f"Available dataset splits: {available}"
            )
        raw_scenarios = self.data[split]

        if not isinstance(raw_scenarios, list):
            raise ValueError(
                f"Split {split!r} is not a flat list of scenarios."
            )
        
        return [
            Scenario(
                env=item["env"],
                agent=item["agent"]
            )
            for item in raw_scenarios
        ]

    def agent_prompt(self, scenario: Scenario) -> str:
        prompt = self.data["agent"].format(
            agent=scenario.agent,
            env=scenario.env,
        ).strip()

        if self.name == "twenty_questions":
            prompt += (
                "\n\n"
                + TWENTY_QUESTIONS_AGENT_PROTOCOL
            )

        return prompt

    def environment_prompt(self, scenario: Scenario) -> str:
        return self.data["env"].format(
            agent=scenario.agent,
            env=scenario.env
        ).strip()

    def environment_optional_message(
            self,
            scenario: Scenario
    ) -> str | None:
        template = self.data.get("env_optional_message")

        if template is None:
            return None

        return template.format(
            agent=scenario.agent,
            env=scenario.env
        ).strip()

    @property
    def environment_default_response(self) -> str | None:
        response = self.data.get("environment_default_response")

        if response is None:
            return None

        return response.strip()

    def judge_prompt(self, scenario: Scenario) -> str | None:
        template = self.data.get("judge_prompt_agent")
        
        if template is None:
            return None
        
        return template.format(
            agent=scenario.agent,
            env=scenario.env
        ).strip()

    def judge_suffix(self, scenario: Scenario) -> str:
        return self.data["judge_prompt_suffix"].format(
            agent=scenario.agent,
            env=scenario.env
        ).strip()

def default_paprika_root() -> Path:
    override = os.getenv("PAPRIKA_ROOT")

    if override:
        return Path(override).expanduser().resolve()

    repository_root = Path(__file__).resolve().parents[2]

    return repository_root / "third_party" / "paprika"

def load_game_config(
    game: str,
    paprika_root: Path | None = None
) -> PaprikaGameConfig:
    if game not in SUPPORTED_GAMES:
        raise ValueError(
            f"Unsupported game: {game!r}."
            f"Supported games are: {sorted(SUPPORTED_GAMES)}."
        )
        
    root = paprika_root or default_paprika_root()

    config_path = (
        root
        / "llm_exploration"
        / "game"
        / "game_configs"
        / SUPPORTED_GAMES[game]
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"PAPARIKA config not found: \n{config_path}\n\n"
            "Did you initialize the PAPRIKA git submodule?"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return PaprikaGameConfig(
        name=game,
        data=data
    )

