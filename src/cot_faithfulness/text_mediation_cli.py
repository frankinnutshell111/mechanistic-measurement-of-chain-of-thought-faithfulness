"""Command-line entry point for Phase 2 text-level mediation."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from .config import apply_text_mediation_overrides, load_config
from .model_loader import load_model_and_tokenizer
from .text_mediation import (
    read_screening_records,
    run_text_mediation,
    validate_screening_model,
    validate_screening_sidecar,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 text-level mediation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id")
    parser.add_argument("--device")
    parser.add_argument("--layers", type=int, nargs="+", help="Zero-based decoder layer indices")
    parser.add_argument("--input", help="Phase 1 screening JSONL")
    parser.add_argument("--output", help="Per-condition Phase 2 JSONL")
    parser.add_argument("--summary-json")
    parser.add_argument("--summary-csv")
    parser.add_argument("--limit", type=int, help="Maximum new condition records to score")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume an existing Phase 2 JSONL checkpoint (default from YAML)",
    )
    parser.add_argument(
        "--eligible-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restrict scoring to Phase 1 condition-eligible records (default: true)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = apply_text_mediation_overrides(
        load_config(args.config),
        model_id=args.model_id,
        device=args.device,
        layers=args.layers,
        input_path=args.input,
        output_path=args.output,
        summary_json_path=args.summary_json,
        summary_csv_path=args.summary_csv,
        limit=args.limit,
        resume=args.resume,
        eligible_only=args.eligible_only,
    )
    screening_records = read_screening_records(config.text_mediation.input_path)
    validate_screening_model(screening_records, config)
    validate_screening_sidecar(config.text_mediation.input_path, config)
    loaded = load_model_and_tokenizer(config.model)
    summary = run_text_mediation(
        config,
        loaded.model,
        loaded.tokenizer,
        screening_records=screening_records,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
