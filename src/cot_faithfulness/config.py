"""Typed experiment configuration and command-line overrides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when configuration values are missing or inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "Qwen/Qwen3-14B"
    revision: str | None = None
    dtype: str = "bfloat16"
    quantization: str = "none"
    device: str = "cuda:0"
    attention_implementation: str = "sdpa"
    low_cpu_mem_usage: bool = True
    layers: tuple[int, ...] = (9, 19, 29)


@dataclass(frozen=True)
class RobustnessSamplingConfig:
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


@dataclass(frozen=True)
class DecodingConfig:
    do_sample: bool = False
    max_reasoning_tokens: int = 1000
    answer_cue: str = "\n\nAnswer:"
    answer_label_prefix: str = " "
    answer_scoring_batch_size: int = 1
    repetition_min_repeats: int = 4
    repetition_min_cycle_tokens: int = 2
    repetition_max_cycle_tokens: int = 64
    robustness_sampling: RobustnessSamplingConfig = field(default_factory=RobustnessSamplingConfig)


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str = "allenai/openbookqa"
    subset: str = "main"
    split: str = "validation"
    revision: str | None = None
    screen_all_examples: bool = True
    target_paired_examples: int = 20


@dataclass(frozen=True)
class DemoConfig:
    id: str
    question_stem: str
    choices: Mapping[str, str]
    answer_key: str


@dataclass(frozen=True)
class PromptConfig:
    labels: tuple[str, ...] = ("A", "B", "C", "D")
    metadata_neutral_candidates: tuple[str, ...] = ("X", "Z", "N", "?")
    black_square_symbol_pairs: tuple[tuple[str, str], ...] = (
        ("□", "■"),
        ("○", "●"),
        ("◇", "◆"),
        ("▽", "▼"),
    )
    black_square_demos: tuple[DemoConfig, ...] = ()


@dataclass(frozen=True)
class ScreeningConfig:
    output_path: str = "results/screening/openbookqa_validation.jsonl"
    resume: bool = True
    limit: int | None = None


@dataclass(frozen=True)
class TextMediationConfig:
    input_path: str = "results/screening/openbookqa_validation.jsonl"
    output_path: str = "results/text_mediation/openbookqa_validation.jsonl"
    summary_json_path: str = "results/text_mediation/summary.json"
    summary_csv_path: str = "results/text_mediation/summary.csv"
    epsilon: float = 1e-8
    minimum_effect_magnitude: float = 1e-6
    eligible_only: bool = True
    resume: bool = True
    limit: int | None = None


@dataclass(frozen=True)
class MechanisticConfig:
    max_reasoning_blocks: int = 10
    batch_size_directions: int = 5
    num_structured_directions: int = 1
    num_random_directions: int = 4
    alpha_primary: float = 1.0
    alpha_sensitivity: float = 0.5
    activation_storage_dtype: str = "bfloat16"
    random_distribution: str = "rademacher"
    direction_norm_epsilon: float = 1e-6
    gram_schmidt_tolerance: float = 1e-7
    gram_schmidt_max_attempts: int = 32


@dataclass(frozen=True)
class MechanisticSmokeConfig:
    input_path: str = "results/screening/openbookqa_validation.jsonl"
    output_path: str = "results/mechanistic/phase4_smoke.json"
    question_id: str | None = None
    hint_condition: str = "metadata"
    layer_idx: int | None = None
    block_idx: int = 0
    null_batch_size: int = 2
    null_absolute_tolerance: float = 1e-4
    null_relative_tolerance: float = 1e-4
    additivity_tolerance: float = 1e-8


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 314159
    model: ModelConfig = field(default_factory=ModelConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    text_mediation: TextMediationConfig = field(default_factory=TextMediationConfig)
    mechanistic: MechanisticConfig = field(default_factory=MechanisticConfig)
    mechanistic_smoke: MechanisticSmokeConfig = field(default_factory=MechanisticSmokeConfig)
    statistics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.prompts.labels != ("A", "B", "C", "D"):
            raise ConfigError("Phase 1 expects exactly the OpenBookQA labels A, B, C, D")
        if self.decoding.do_sample:
            raise ConfigError("Primary screening must use deterministic decoding (do_sample=false)")
        if self.model.quantization != "none":
            raise ConfigError("The primary experiment must not use model quantization")
        if not self.model.layers:
            raise ConfigError("model.layers cannot be empty")
        if any(layer < 0 for layer in self.model.layers):
            raise ConfigError("model.layers must contain non-negative zero-based indices")
        if len(set(self.model.layers)) != len(self.model.layers):
            raise ConfigError("model.layers must not contain duplicates")
        if self.decoding.max_reasoning_tokens <= 0:
            raise ConfigError("max_reasoning_tokens must be positive")
        if self.decoding.answer_scoring_batch_size <= 0:
            raise ConfigError("answer_scoring_batch_size must be positive")
        if self.dataset.target_paired_examples <= 0:
            raise ConfigError("target_paired_examples must be positive")
        if self.text_mediation.epsilon <= 0:
            raise ConfigError("text_mediation.epsilon must be positive")
        if self.text_mediation.minimum_effect_magnitude < 0:
            raise ConfigError("text_mediation.minimum_effect_magnitude cannot be negative")
        if self.text_mediation.limit is not None and self.text_mediation.limit <= 0:
            raise ConfigError("text_mediation.limit must be positive when provided")
        if self.mechanistic.max_reasoning_blocks <= 0:
            raise ConfigError("mechanistic.max_reasoning_blocks must be positive")
        if self.mechanistic.batch_size_directions <= 0:
            raise ConfigError("mechanistic.batch_size_directions must be positive")
        if self.mechanistic.num_structured_directions != 1:
            raise ConfigError("The experiment requires exactly one structured direction")
        if self.mechanistic.num_random_directions < 0:
            raise ConfigError("mechanistic.num_random_directions cannot be negative")
        if self.mechanistic.alpha_primary < 0 or self.mechanistic.alpha_sensitivity < 0:
            raise ConfigError("mechanistic alpha values cannot be negative")
        if self.mechanistic.activation_storage_dtype not in {
            "bfloat16",
            "float16",
            "float32",
        }:
            raise ConfigError("Unsupported mechanistic.activation_storage_dtype")
        if self.mechanistic.random_distribution not in {"rademacher", "gaussian"}:
            raise ConfigError("mechanistic.random_distribution must be rademacher or gaussian")
        if self.mechanistic.direction_norm_epsilon <= 0:
            raise ConfigError("mechanistic.direction_norm_epsilon must be positive")
        if self.mechanistic.gram_schmidt_tolerance <= 0:
            raise ConfigError("mechanistic.gram_schmidt_tolerance must be positive")
        if self.mechanistic.gram_schmidt_max_attempts <= 0:
            raise ConfigError("mechanistic.gram_schmidt_max_attempts must be positive")
        if self.mechanistic_smoke.hint_condition not in {"metadata", "black_square"}:
            raise ConfigError("mechanistic_smoke.hint_condition must be metadata or black_square")
        if self.mechanistic_smoke.layer_idx is not None:
            if self.mechanistic_smoke.layer_idx < 0:
                raise ConfigError("mechanistic_smoke.layer_idx must be non-negative")
            if self.mechanistic_smoke.layer_idx not in self.model.layers:
                raise ConfigError("mechanistic_smoke.layer_idx must be included in model.layers")
        if self.mechanistic_smoke.block_idx < 0:
            raise ConfigError("mechanistic_smoke.block_idx must be non-negative")
        if self.mechanistic_smoke.null_batch_size < 2:
            raise ConfigError("mechanistic_smoke.null_batch_size must be at least 2")
        if (
            self.mechanistic_smoke.null_absolute_tolerance < 0
            or self.mechanistic_smoke.null_relative_tolerance < 0
        ):
            raise ConfigError("mechanistic_smoke null tolerances cannot be negative")
        if self.mechanistic_smoke.additivity_tolerance < 0:
            raise ConfigError("mechanistic_smoke.additivity_tolerance cannot be negative")
        if len(self.prompts.black_square_demos) != 4:
            raise ConfigError("Exactly four fixed Black-Square demonstrations are required")
        demo_ids = [demo.id for demo in self.prompts.black_square_demos]
        if len(set(demo_ids)) != len(demo_ids):
            raise ConfigError("Black-Square demonstration IDs must be unique")
        labels = set(self.prompts.labels)
        for demo in self.prompts.black_square_demos:
            if set(demo.choices) != labels:
                raise ConfigError(f"Demo {demo.id} must contain choices A-D exactly")
            if demo.answer_key not in labels:
                raise ConfigError(f"Demo {demo.id} has an invalid answer_key")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _only_known(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigError(f"Unknown {name} keys: {', '.join(sorted(unknown))}")


def config_from_mapping(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Construct a validated config without adding a third-party schema dependency."""

    _only_known(
        raw,
        {
            "seed",
            "model",
            "decoding",
            "dataset",
            "prompts",
            "screening",
            "text_mediation",
            "mechanistic",
            "mechanistic_smoke",
            "statistics",
        },
        "top-level",
    )

    model_raw = _require_mapping(raw.get("model", {}), "model")
    _only_known(model_raw, set(ModelConfig.__dataclass_fields__), "model")
    model = ModelConfig(
        **{
            **model_raw,
            "layers": tuple(model_raw.get("layers", ModelConfig.layers)),
        }
    )

    decoding_raw = _require_mapping(raw.get("decoding", {}), "decoding")
    _only_known(decoding_raw, set(DecodingConfig.__dataclass_fields__), "decoding")
    robustness_raw = _require_mapping(
        decoding_raw.get("robustness_sampling", {}), "decoding.robustness_sampling"
    )
    _only_known(
        robustness_raw,
        set(RobustnessSamplingConfig.__dataclass_fields__),
        "decoding.robustness_sampling",
    )
    decoding_values = dict(decoding_raw)
    decoding_values["robustness_sampling"] = RobustnessSamplingConfig(**robustness_raw)
    decoding = DecodingConfig(**decoding_values)

    dataset_raw = _require_mapping(raw.get("dataset", {}), "dataset")
    _only_known(dataset_raw, set(DatasetConfig.__dataclass_fields__), "dataset")
    dataset = DatasetConfig(**dataset_raw)

    prompts_raw = _require_mapping(raw.get("prompts", {}), "prompts")
    _only_known(prompts_raw, set(PromptConfig.__dataclass_fields__), "prompts")
    demos_raw = prompts_raw.get("black_square_demos", ())
    if not isinstance(demos_raw, Sequence) or isinstance(demos_raw, (str, bytes)):
        raise ConfigError("prompts.black_square_demos must be a sequence")
    demos = tuple(
        DemoConfig(
            id=str(_require_mapping(item, "demo")["id"]),
            question_stem=str(item["question_stem"]),
            choices={
                str(k): str(v) for k, v in _require_mapping(item["choices"], "choices").items()
            },
            answer_key=str(item["answer_key"]),
        )
        for item in demos_raw
    )
    prompt_values = dict(prompts_raw)
    prompt_values["labels"] = tuple(prompts_raw.get("labels", PromptConfig.labels))
    prompt_values["metadata_neutral_candidates"] = tuple(
        prompts_raw.get("metadata_neutral_candidates", PromptConfig.metadata_neutral_candidates)
    )
    prompt_values["black_square_symbol_pairs"] = tuple(
        tuple(pair)
        for pair in prompts_raw.get(
            "black_square_symbol_pairs", PromptConfig.black_square_symbol_pairs
        )
    )
    prompt_values["black_square_demos"] = demos
    prompts = PromptConfig(**prompt_values)

    screening_raw = _require_mapping(raw.get("screening", {}), "screening")
    _only_known(screening_raw, set(ScreeningConfig.__dataclass_fields__), "screening")
    screening = ScreeningConfig(**screening_raw)

    text_mediation_raw = _require_mapping(raw.get("text_mediation", {}), "text_mediation")
    _only_known(
        text_mediation_raw,
        set(TextMediationConfig.__dataclass_fields__),
        "text_mediation",
    )
    text_mediation = TextMediationConfig(**text_mediation_raw)

    mechanistic_raw = _require_mapping(raw.get("mechanistic", {}), "mechanistic")
    _only_known(
        mechanistic_raw,
        set(MechanisticConfig.__dataclass_fields__),
        "mechanistic",
    )
    mechanistic = MechanisticConfig(**mechanistic_raw)

    mechanistic_smoke_raw = _require_mapping(
        raw.get("mechanistic_smoke", {}), "mechanistic_smoke"
    )
    _only_known(
        mechanistic_smoke_raw,
        set(MechanisticSmokeConfig.__dataclass_fields__),
        "mechanistic_smoke",
    )
    mechanistic_smoke = MechanisticSmokeConfig(**mechanistic_smoke_raw)

    config = ExperimentConfig(
        seed=int(raw.get("seed", 314159)),
        model=model,
        decoding=decoding,
        dataset=dataset,
        prompts=prompts,
        screening=screening,
        text_mediation=text_mediation,
        mechanistic=mechanistic,
        mechanistic_smoke=mechanistic_smoke,
        statistics=dict(_require_mapping(raw.get("statistics", {}), "statistics")),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML lazily so importing the package remains lightweight."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
        raise RuntimeError("PyYAML is required to load experiment configuration") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_mapping(_require_mapping(raw, "configuration"))


def apply_overrides(
    config: ExperimentConfig,
    *,
    model_id: str | None = None,
    device: str | None = None,
    layers: Sequence[int] | None = None,
    dataset_split: str | None = None,
    max_reasoning_tokens: int | None = None,
    output_path: str | None = None,
    limit: int | None = None,
    seed: int | None = None,
    resume: bool | None = None,
) -> ExperimentConfig:
    """Apply the supported CLI overrides to an immutable configuration."""

    model = replace(
        config.model,
        model_id=model_id if model_id is not None else config.model.model_id,
        device=device if device is not None else config.model.device,
        layers=tuple(layers) if layers is not None else config.model.layers,
    )
    decoding = replace(
        config.decoding,
        max_reasoning_tokens=(
            max_reasoning_tokens
            if max_reasoning_tokens is not None
            else config.decoding.max_reasoning_tokens
        ),
    )
    dataset = replace(
        config.dataset,
        split=dataset_split if dataset_split is not None else config.dataset.split,
    )
    screening = replace(
        config.screening,
        output_path=output_path if output_path is not None else config.screening.output_path,
        limit=limit if limit is not None else config.screening.limit,
        resume=resume if resume is not None else config.screening.resume,
    )
    updated = replace(
        config,
        seed=seed if seed is not None else config.seed,
        model=model,
        decoding=decoding,
        dataset=dataset,
        screening=screening,
    )
    updated.validate()
    return updated


def apply_text_mediation_overrides(
    config: ExperimentConfig,
    *,
    model_id: str | None = None,
    device: str | None = None,
    layers: Sequence[int] | None = None,
    input_path: str | None = None,
    output_path: str | None = None,
    summary_json_path: str | None = None,
    summary_csv_path: str | None = None,
    limit: int | None = None,
    resume: bool | None = None,
    eligible_only: bool | None = None,
) -> ExperimentConfig:
    """Apply Phase 2 CLI overrides without changing Phase 1 output settings."""

    model = replace(
        config.model,
        model_id=model_id if model_id is not None else config.model.model_id,
        device=device if device is not None else config.model.device,
        layers=tuple(layers) if layers is not None else config.model.layers,
    )
    text_mediation = replace(
        config.text_mediation,
        input_path=input_path if input_path is not None else config.text_mediation.input_path,
        output_path=output_path if output_path is not None else config.text_mediation.output_path,
        summary_json_path=(
            summary_json_path
            if summary_json_path is not None
            else config.text_mediation.summary_json_path
        ),
        summary_csv_path=(
            summary_csv_path
            if summary_csv_path is not None
            else config.text_mediation.summary_csv_path
        ),
        limit=limit if limit is not None else config.text_mediation.limit,
        resume=resume if resume is not None else config.text_mediation.resume,
        eligible_only=(
            eligible_only if eligible_only is not None else config.text_mediation.eligible_only
        ),
    )
    updated = replace(config, model=model, text_mediation=text_mediation)
    updated.validate()
    return updated


def apply_mechanistic_smoke_overrides(
    config: ExperimentConfig,
    *,
    model_id: str | None = None,
    device: str | None = None,
    layers: Sequence[int] | None = None,
    input_path: str | None = None,
    output_path: str | None = None,
    question_id: str | None = None,
    hint_condition: str | None = None,
    layer_idx: int | None = None,
    block_idx: int | None = None,
    alpha: float | None = None,
    null_batch_size: int | None = None,
    null_absolute_tolerance: float | None = None,
    null_relative_tolerance: float | None = None,
) -> ExperimentConfig:
    """Apply Phase 4 single-example overrides without changing earlier phases."""

    selected_layers = (
        tuple(layers)
        if layers is not None
        else ((layer_idx,) if layer_idx is not None else config.model.layers)
    )
    selected_layer = layer_idx if layer_idx is not None else config.mechanistic_smoke.layer_idx
    if layer_idx is not None and layer_idx not in selected_layers:
        selected_layers = (*selected_layers, layer_idx)
    model = replace(
        config.model,
        model_id=model_id if model_id is not None else config.model.model_id,
        device=device if device is not None else config.model.device,
        layers=selected_layers,
    )
    mechanistic = replace(
        config.mechanistic,
        alpha_primary=alpha if alpha is not None else config.mechanistic.alpha_primary,
    )
    smoke = replace(
        config.mechanistic_smoke,
        input_path=input_path if input_path is not None else config.mechanistic_smoke.input_path,
        output_path=(
            output_path if output_path is not None else config.mechanistic_smoke.output_path
        ),
        question_id=(
            question_id if question_id is not None else config.mechanistic_smoke.question_id
        ),
        hint_condition=(
            hint_condition
            if hint_condition is not None
            else config.mechanistic_smoke.hint_condition
        ),
        layer_idx=selected_layer,
        block_idx=block_idx if block_idx is not None else config.mechanistic_smoke.block_idx,
        null_batch_size=(
            null_batch_size
            if null_batch_size is not None
            else config.mechanistic_smoke.null_batch_size
        ),
        null_absolute_tolerance=(
            null_absolute_tolerance
            if null_absolute_tolerance is not None
            else config.mechanistic_smoke.null_absolute_tolerance
        ),
        null_relative_tolerance=(
            null_relative_tolerance
            if null_relative_tolerance is not None
            else config.mechanistic_smoke.null_relative_tolerance
        ),
    )
    updated = replace(config, model=model, mechanistic=mechanistic, mechanistic_smoke=smoke)
    updated.validate()
    return updated
