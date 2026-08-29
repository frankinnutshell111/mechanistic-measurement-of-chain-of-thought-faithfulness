"""Command-line entry point for the Phase 4 single-example smoke run."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from .config import apply_mechanistic_smoke_overrides, load_config
from .mechanistic_smoke import run_mechanistic_smoke, select_smoke_record
from .model_loader import load_model_and_tokenizer
from .text_mediation import (
    read_screening_records,
    validate_screening_model,
    validate_screening_sidecar,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Phase 4 activation-mediation smoke case"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id")
    parser.add_argument("--device")
    parser.add_argument("--layers", type=int, nargs="+", help="Allowed decoder layer indices")
    parser.add_argument("--input", help="Phase 1 screening JSONL")
    parser.add_argument("--output", help="Phase 4 smoke JSON")
    parser.add_argument("--question-id", help="Eligible question; default is the first eligible")
    parser.add_argument("--hint-condition", choices=("metadata", "black_square"))
    parser.add_argument("--layer", type=int, help="One zero-based decoder layer")
    parser.add_argument("--block", type=int, help="One zero-based segmented reasoning block")
    parser.add_argument("--alpha", type=float, help="Structured-direction intervention strength")
    parser.add_argument("--null-batch-size", type=int)
    parser.add_argument("--null-absolute-tolerance", type=float)
    parser.add_argument("--null-relative-tolerance", type=float)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = apply_mechanistic_smoke_overrides(
        load_config(args.config),
        model_id=args.model_id,
        device=args.device,
        layers=args.layers,
        input_path=args.input,
        output_path=args.output,
        question_id=args.question_id,
        hint_condition=args.hint_condition,
        layer_idx=args.layer,
        block_idx=args.block,
        alpha=args.alpha,
        null_batch_size=args.null_batch_size,
        null_absolute_tolerance=args.null_absolute_tolerance,
        null_relative_tolerance=args.null_relative_tolerance,
    )

    # Validate the small Phase 1 artifact before allocating the model.
    records = read_screening_records(config.mechanistic_smoke.input_path)
    validate_screening_model(records, config)
    validate_screening_sidecar(config.mechanistic_smoke.input_path, config)
    select_smoke_record(
        records,
        hint_condition=config.mechanistic_smoke.hint_condition,
        question_id=config.mechanistic_smoke.question_id,
    )
    loaded = load_model_and_tokenizer(config.model)
    result = run_mechanistic_smoke(
        config,
        loaded.model,
        loaded.tokenizer,
        screening_records=records,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": config.mechanistic_smoke.output_path,
                "question_id": result["question_id"],
                "hint_condition": result["hint_condition"],
                "layer_idx": result["layer_idx"],
                "block_idx": result["block_idx"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
