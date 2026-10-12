from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llm_exploration.backend import OpenAICompatibleBackend
from llm_exploration.paprika import (
    PaprikaGameConfig,
    Scenario,
)


Message = dict[str, str]


@dataclass
class EpisodeResult:
    game: str
    split: str
    scenario_index: int

    agent_game_scenario: str
    env_game_scenario: str

    goal_reached: bool
    judge_label: bool
    success: bool

    num_turns: int
    max_turns: int

    conversation: list[Message]
    env_conversation: list[Message]
    judge_conversation: list[Message] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GuessingGameRunner:
    def __init__(
        self,
        *,
        agent: OpenAICompatibleBackend,
        environment: OpenAICompatibleBackend,
        judge: OpenAICompatibleBackend,
        agent_temperature: float = 0.7,
        agent_top_p: float = 1.0,
        agent_max_tokens: int = 128,
        environment_temperature: float = 0.0,
        environment_top_p: float = 1.0,
        environment_max_tokens: int = 1024,
        judge_temperature: float = 0.0,
        judge_top_p: float = 1.0,
        judge_max_tokens: int = 128,
        max_environment_attempts: int = 5,
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.judge = judge

        self.agent_temperature = agent_temperature
        self.agent_top_p = agent_top_p
        self.agent_max_tokens = agent_max_tokens

        self.environment_temperature = environment_temperature
        self.environment_top_p = environment_top_p
        self.environment_max_tokens = environment_max_tokens

        self.judge_temperature = judge_temperature
        self.judge_top_p = judge_top_p
        self.judge_max_tokens = judge_max_tokens

        self.max_environment_attempts = max_environment_attempts

    @staticmethod
    def _extract_twenty_questions_environment_response(
        response: str,
    ) -> str:
        normalized = response.strip().lower()

        if normalized in {"yes", "yes."}:
            return "Yes"

        if normalized in {"no", "no."}:
            return "No"

        if normalized in {
            "goal reached",
            "goal reached.",
        }:
            return "Goal reached"

        raise ValueError(
            "Twenty Questions environment response "
            f"is not valid: {response!r}"
        )

    def _generate_environment_response(
        self,
        game_config: PaprikaGameConfig,
        messages: list[Message],
    ) -> tuple[str, str]:
        if game_config.name != "twenty_questions":
            raw = self.environment.generate(
                messages,
                temperature=self.environment_temperature,
                top_p=self.environment_top_p,
                max_tokens=self.environment_max_tokens,
            )

            return raw, raw

        last_raw_response = ""

        for _ in range(self.max_environment_attempts):
            last_raw_response = self.environment.generate(
                messages,
                temperature=self.environment_temperature,
                top_p=self.environment_top_p,
                max_tokens=self.environment_max_tokens,
            )

            try:
                extracted = (
                    self
                    ._extract_twenty_questions_environment_response(
                        last_raw_response
                    )
                )

                return last_raw_response, extracted

            except ValueError:
                continue

        default_response = (
            game_config.environment_default_response
        )

        if default_response is None:
            raise RuntimeError(
                "Environment produced no valid response "
                "and the game has no default response."
            )

        return last_raw_response, default_response

    @staticmethod
    def _judge_user_message(
        game_config: PaprikaGameConfig,
        scenario: Scenario,
        agent_response: str,
    ) -> str:
        suffix = game_config.judge_suffix(scenario)

        return (
            "The conversation begins here: \n\n"
            f"Agent: {agent_response}\n\n"
            " (End of Agent Turn)\n\n"
            f"{suffix}"
        )

    def _run_judge(
        self,
        game_config: PaprikaGameConfig,
        scenario: Scenario,
        agent_response: str,
    ) -> tuple[bool, list[Message] | None]:
        system_prompt = game_config.judge_prompt(
            scenario
        )

        if system_prompt is None:
            return True, None

        messages: list[Message] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": self._judge_user_message(
                    game_config,
                    scenario,
                    agent_response,
                ),
            },
        ]

        response = self.judge.generate(
            messages,
            temperature=self.judge_temperature,
            top_p=self.judge_top_p,
            max_tokens=self.judge_max_tokens,
        ).strip()

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        return response == "<VALID>", messages

    def run(
        self,
        *,
        game_config: PaprikaGameConfig,
        scenario: Scenario,
        split: str,
        scenario_index: int,
    ) -> EpisodeResult:
        agent_messages: list[Message] = [
            {
                "role": "user",
                "content": game_config.agent_prompt(
                    scenario
                ),
            }
        ]

        environment_messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    game_config.environment_prompt(
                        scenario
                    )
                ),
            }
        ]

        goal_reached = False
        judge_label = True
        final_judge_messages = None

        turn = 0

        while (
            turn < game_config.max_turns
            and not goal_reached
        ):
            agent_response = self.agent.generate(
                agent_messages,
                temperature=self.agent_temperature,
                top_p=self.agent_top_p,
                max_tokens=self.agent_max_tokens,
            )

            agent_messages.append(
                {
                    "role": "assistant",
                    "content": agent_response,
                }
            )

            env_input = agent_response

            optional_message = (
                game_config
                .environment_optional_message(
                    scenario
                )
            )

            if optional_message:
                env_input += (
                    "\n" + optional_message
                )

            environment_messages.append(
                {
                    "role": "user",
                    "content": env_input,
                }
            )

            (
                raw_environment_response,
                extracted_environment_response,
            ) = self._generate_environment_response(
                game_config,
                environment_messages,
            )

            environment_messages.append(
                {
                    "role": "assistant",
                    "content": raw_environment_response,
                }
            )

            agent_messages.append(
                {
                    "role": "user",
                    "content": (
                        extracted_environment_response
                    ),
                }
            )

            turn += 1

            env_says_goal_reached = (
                "goal reached"
                in extracted_environment_response.lower()
            )

            first_judge_label, _ = (
                self._run_judge(
                    game_config,
                    scenario,
                    agent_response,
                )
            )

            goal_reached = (
                env_says_goal_reached
                or first_judge_label
            )

            # PAPRIKA performs an additional judge
            # verification when the game appears solved.
            if goal_reached:
                (
                    judge_label,
                    final_judge_messages,
                ) = self._run_judge(
                    game_config,
                    scenario,
                    agent_response,
                )

                break

        success = (
            goal_reached
            and judge_label
        )

        return EpisodeResult(
            game=game_config.name,
            split=split,
            scenario_index=scenario_index,
            agent_game_scenario=scenario.agent,
            env_game_scenario=scenario.env,
            goal_reached=goal_reached,
            judge_label=judge_label,
            success=success,
            num_turns=turn,
            max_turns=game_config.max_turns,
            conversation=agent_messages,
            env_conversation=environment_messages,
            judge_conversation=final_judge_messages,
        )
