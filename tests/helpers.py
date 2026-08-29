from __future__ import annotations

from cot_faithfulness.config import DemoConfig, ExperimentConfig, PromptConfig


class CharacterTokenizer:
    """Tiny deterministic tokenizer sufficient for offline prompt/unit tests."""

    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = -1
    _special = {"</think>": 2, "<think>": 3}

    def __init__(self, multi_char_tokens: dict[str, list[int]] | None = None):
        self.multi_char_tokens = multi_char_tokens or {}

    def convert_tokens_to_ids(self, token: str):
        return self._special.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._special:
            return [self._special[text]]
        if text in self.multi_char_tokens:
            return list(self.multi_char_tokens[text])
        result: list[int] = []
        for character in text:
            override = self.multi_char_tokens.get(character)
            if override is not None:
                result.extend(override)
            else:
                result.append(ord(character) + 100)
        return result

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ):
        reverse = {value: key for key, value in self._special.items()}
        pieces = []
        for token_id in token_ids:
            if token_id in reverse:
                if not skip_special_tokens:
                    pieces.append(reverse[token_id])
            elif token_id >= 100:
                pieces.append(chr(token_id - 100))
        return "".join(pieces)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert tokenize and add_generation_prompt and enable_thinking
        content = messages[0]["content"]
        return [10, *self.encode(content), 11, self._special["<think>"]]


def prompt_config() -> PromptConfig:
    demos = (
        DemoConfig("d1", "Demo one?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A"),
        DemoConfig("d2", "Demo two?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "B"),
        DemoConfig("d3", "Demo three?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "C"),
        DemoConfig("d4", "Demo four?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "D"),
    )
    return PromptConfig(black_square_demos=demos)


def experiment_config() -> ExperimentConfig:
    return ExperimentConfig(prompts=prompt_config())
