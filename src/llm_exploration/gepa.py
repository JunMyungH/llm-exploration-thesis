from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import dspy
from dotenv import load_dotenv

from llm_exploration.cli import build_backend
from llm_exploration.paprika import PaprikaGameConfig, Scenario, load_game_config
from llm_exploration.runner import GuessingGameRunner


class AskQuestion(dspy.Signature):
    """Choose the next question in a multi-turn guessing game. Produce one question and then wait for a reply."""

    game_rules: str = dspy.InputField(desc="Fixed original game rules and interaction protocol")
    history: str = dspy.InputField(desc="Previous questions and environment replies, in order")
    question: str = dspy.OutputField(desc="Exactly one legal question, with no simulated answer")


class DspyAgentBackend:
    """Adapt a DSPy predictor to the existing runner's agent.generate interface."""

    def __init__(self, predictor: dspy.Predict, rules: str) -> None:
        self.predictor = predictor
        self.rules = rules

    def generate(self, messages, *, temperature, top_p, max_tokens) -> str:
        history_lines = []
        for message in messages[1:]:
            role = "Agent" if message["role"] == "assistant" else "Environment"
            history_lines.append(f"{role}: {message['content']}")
        history = "\n".join(history_lines) or "(No questions asked yet.)"
        answer = self.predictor(game_rules=self.rules, history=history)
        return str(answer.question).strip()


class GepaEpisode(dspy.Module):
    """Run a full PAPRIKA-configured episode; GEPA optimizes only self.ask."""

    def __init__(
        self,
        game_config: PaprikaGameConfig,
        scenarios: list[Scenario],
        environment,
        judge,
        use_judge: bool,
    ) -> None:
        super().__init__()
        self.ask = dspy.Predict(AskQuestion)
        self.game_config = game_config
        self.scenarios = scenarios
        self.environment = environment
        self.judge = judge
        self.use_judge = use_judge

    def forward(self, scenario_index: int) -> dspy.Prediction:
        index = int(scenario_index)
        scenario = self.scenarios[index]
        # PAPRIKA's two supported agent templates only use {agent}. Never include the answer.
        if "{env}" in self.game_config.data["agent"]:
            raise ValueError("The agent prompt must never contain the hidden target")
        rules = self.game_config.agent_prompt(scenario)
        agent = DspyAgentBackend(self.ask, rules)
        runner = GuessingGameRunner(
            agent=agent,
            environment=self.environment,
            judge=self.judge,
            use_judge=self.use_judge,
        )
        result = runner.run(
            game_config=self.game_config,
            scenario=scenario,
            split="train",
            scenario_index=index,
        )
        return dspy.Prediction(
            questions=[m["content"] for m in result.conversation if m["role"] == "assistant"],
            environment_answers=[m["content"] for m in result.env_conversation if m["role"] == "assistant"],
            environment_goal=result.goal_reached,
            judge_label=result.judge_label if self.use_judge else None,
            num_turns=result.num_turns,
        )


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", value))


def mentions_phrase(text: str, phrase: str) -> bool:
    return f" {normalize(phrase)} " in f" {normalize(text)} "


def explicitly_guessed(game: str, target: str, question: str) -> bool:
    # Conservative guard against a false positive from a weak environment.
    if question.count("?") != 1 or not question.strip().endswith("?"):
        return False
    if game == "guess_my_city":
        parts = [part.strip() for part in target.rsplit(",", 1)]
        if len(parts) != 2:
            return False
        city, country = parts
        aliases = [city.split("(")[0].strip()]
        if "(" in city and ")" in city:
            aliases.append(city.split("(", 1)[1].split(")", 1)[0].strip())
        return any(mentions_phrase(question, alias) for alias in aliases) and mentions_phrase(question, country)
    return mentions_phrase(question, target)


def city_name_leaked(target: str, question: str, answer: str) -> bool:
    if normalize(answer) == "goal reached":
        return False
    city, country = (part.strip() for part in target.rsplit(",", 1))
    primary_city = city.split("(", 1)[0].strip()
    # Flags both city and country; a guess-confirmation answer should still avoid naming either.
    return mentions_phrase(answer, primary_city) or mentions_phrase(answer, country)


class EpisodeFeedback:
    """A train/dev-only metric. Targets are used here, never in the agent input."""

    def __init__(self, game: str, scenarios: list[Scenario], max_turns: int) -> None:
        self.game = game
        self.scenarios = scenarios
        self.max_turns = max_turns

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        target = self.scenarios[int(gold.scenario_index)].env
        questions = list(pred.questions)
        answers = list(pred.environment_answers)
        turns = int(pred.num_turns)
        final_question = questions[-1] if questions else ""
        correct_guess = explicitly_guessed(self.game, target, final_question)
        success = bool(pred.environment_goal and correct_guess)

        invalid = sum(q.count("?") != 1 or not q.strip().endswith("?") for q in questions)
        normalized_questions = [normalize(q) for q in questions]
        repeats = len(normalized_questions) - len(set(normalized_questions))
        leaked = (
            self.game == "guess_my_city"
            and any(city_name_leaked(target, q, a) for q, a in zip(questions, answers))
        )

        score = (0.7 + 0.3 * (self.max_turns - turns) / self.max_turns) if success and not leaked else 0.0
        score = max(0.0, score - 0.04 * invalid - 0.02 * repeats)

        recent = [f"{i+1}. {q} -> {a}" for i, (q, a) in enumerate(zip(questions, answers))]
        compact_trace = "\n".join(recent[-min(len(recent), 12):])
        feedback = (
            f"Game: {self.game}; success: {success}; turns: {turns}/{self.max_turns}; "
            f"invalid questions: {invalid}; exact repeats: {repeats}; environment leakage: {leaked}.\n"
            f"Recent interactions:\n{compact_trace}\n"
            "Improve the strategy of selecting the next informative question while respecting all fixed game rules. "
            "Do not assume the target is known to the agent."
        )
        if leaked:
            feedback += "\nThis trajectory is invalid because the environment revealed the answer."
        return dspy.Prediction(score=score, feedback=feedback)


def make_lm(prefix: str, *, max_tokens: int, temperature: float) -> dspy.LM:
    model = os.environ[f"{prefix}_MODEL"]
    api_key = os.getenv(f"{prefix}_API_KEY", "ollama")
    base_url = os.getenv(f"{prefix}_BASE_URL", "http://127.0.0.1:11434/v1")
    kwargs = dict(api_base=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens, cache=False)
    effort = os.getenv(f"{prefix}_REASONING_EFFORT")
    if effort:
        kwargs["reasoning_effort"] = effort
    providers = ("openai/", "anthropic/", "ollama/", "ollama_chat/", "azure/", "gemini/")
    model_id = model if model.startswith(providers) else f"openai/{model}"
    return dspy.LM(model_id, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize an agent question prompt using DSPy GEPA")
    parser.add_argument("--game", choices=("twenty_questions", "guess_my_city"), required=True)
    parser.add_argument("--train-size", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=2)
    parser.add_argument("--max-metric-calls", type=int, default=24)
    parser.add_argument("--max-turns", type=int, default=20, help="6 for local smoke tests; 20 for real runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--use-judge", action="store_true", help="Use a trustworthy model for judging")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if min(args.train_size, args.val_size, args.max_metric_calls, args.max_turns) <= 0:
        raise ValueError("All sizes, metric calls, and turns must be positive")
    config = load_game_config(args.game)
    if args.max_turns > config.max_turns:
        raise ValueError("Cannot exceed the game's maximum turns")
    if args.max_turns != config.max_turns:
        config = PaprikaGameConfig(config.name, {**config.data, "max_turns": args.max_turns})
    scenarios = config.scenarios("train")
    if args.train_size + args.val_size > len(scenarios):
        raise ValueError("Not enough training scenarios to create disjoint splits")

    indices = list(range(len(scenarios)))
    random.Random(args.seed).shuffle(indices)
    train_indices = indices[:args.train_size]
    val_indices = indices[args.train_size:args.train_size + args.val_size]
    trainset = [dspy.Example(scenario_index=i).with_inputs("scenario_index") for i in train_indices]
    valset = [dspy.Example(scenario_index=i).with_inputs("scenario_index") for i in val_indices]

    dspy.configure(lm=make_lm("AGENT", max_tokens=256, temperature=0.7))
    
    # Fall back to same Ollama model if a separate reflection model is not configured.
    for suffix in ("MODEL", "API_KEY", "BASE_URL", "REASONING_EFFORT"):
        os.environ.setdefault(f"REFLECTION_{suffix}", os.getenv(f"AGENT_{suffix}", ""))
    reflection_lm = make_lm("REFLECTION", max_tokens=4096, temperature=0.7)
    environment = build_backend("ENV", fallback=build_backend("AGENT"))
    judge = build_backend("JUDGE", fallback=environment) if args.use_judge else None

    student = GepaEpisode(config, scenarios, environment, judge, use_judge=args.use_judge)
    metric = EpisodeFeedback(args.game, scenarios, config.max_turns)
    output_dir = args.output_dir or (Path("results") / "gepa" / args.game / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "game": args.game,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "max_turns": config.max_turns,
        "max_metric_calls": args.max_metric_calls,
        "seed": args.seed,
        "agent_model": os.getenv("AGENT_MODEL"),
        "reflection_model": os.getenv("REFLECTION_MODEL"),
        "environment_model": environment.model,
        "judge_used": args.use_judge,
        "note": "Development run; no final PAPRIKA eval split was used.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=2,
        num_threads=1,
        use_merge=False,
        seed=args.seed,
        track_stats=True,
        log_dir=str(output_dir / "logs"),
    )
    print(f"GEPA start: {args.game} train={train_indices} val={val_indices}")
    optimized = optimizer.compile(student, trainset=trainset, valset=valset)
    optimized.save(str(output_dir / "optimized_program.json"))
    (output_dir / "optimized_instructions.txt").write_text(
        optimized.ask.signature.instructions, encoding="utf-8"
    )
    print(f"Saved: {output_dir / 'optimized_program.json'}")
    print(f"Optimized instructions: {output_dir / 'optimized_instructions.txt'}")
    print("Held-out PAPRIKA eval split was NOT used.")


if __name__ == "__main__":
    main()
