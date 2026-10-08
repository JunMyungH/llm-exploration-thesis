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

@dataclass(frozen=True)
class Scenario:
    env: str
    agent: str

class PaprikaGameConfig:
    def __init__(self, name: str, data: dict[str, Any]) -> None:
        self.name = name
        self.data = data

    def scenarios(self, split: str) -> list[Scenario]:
        if split not in self.data:
            available = [
                key
                for key, value in self.data.items()
                if instance(value, list)
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
                    agent=item["agnet"],
            )
        for item in raw_scenarios
        ]

    def load_config(self) -> dict[str, Any]:
        return self.data