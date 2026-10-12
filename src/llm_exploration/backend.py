from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


Message = dict[str, str]


@dataclass
class OpenAICompatibleBackend:
    model: str
    api_key: str
    base_url: str | None = None
    timeout: float = 180.0

    def __post_init__(self) -> None:
        client_args = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }

        if self.base_url:
            client_args["base_url"] = self.base_url

        self.client = OpenAI(**client_args)

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError(
                f"Model {self.model!r} returned an empty response."
            )

        return content.strip()