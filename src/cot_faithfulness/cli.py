"""Command-line entry points for the research pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from .config import apply_overrides, load_config
from .model_loader import load_model_and_tokenizer
from .screening import run_screening


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 OpenBookQA screening")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id")
    parser.add_argument("--device")
    parser.add_argument("--layers", type=int, nargs="+", help="Zero-based decoder layer indices")
    parser.add_argument("--dataset-split")
    parser.add_argument("--max-reasoning-tokens", type=int)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume an existing JSONL checkpoint (default from YAML)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = apply_overrides(
        load_config(args.config),
        model_id=args.model_id,
        device=args.device,
        layers=args.layers,
        dataset_split=args.dataset_split,
        max_reasoning_tokens=args.max_reasoning_tokens,
        output_path=args.output,
        limit=args.limit,
        seed=args.seed,
        resume=args.resume,
    )
    loaded = load_model_and_tokenizer(config.model)
    summary = run_screening(config, loaded.model, loaded.tokenizer)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
