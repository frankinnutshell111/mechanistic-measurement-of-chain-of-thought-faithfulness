# Mechanistic measurement of chain-of-thought faithfulness

This repository implements a causal pipeline for measuring whether an
answer-relevant residual-stream intervention changes an answer directly or
through the model's subsequent visible chain of thought (CoT).

The implementation is intentionally phased. **Phases 1–4 are complete.**
Phase 1 screens matched Metadata/Black-Square prompts and stores exact-token
Qwen3 reasoning traces. Phase 2 swaps those stored traces across matched prompt
contexts, re-scores all answer labels, and measures direct versus text-mediated
hint effects. Phase 3 captures selected residual streams and provides tested,
temporary activation-intervention hooks. Phase 4 connects those pieces in a
guarded one-example mechanistic runner. No Phase 5 batch experiment runs unless
it is implemented and invoked separately.

## Phase 1 design

- The default model is unquantized `Qwen/Qwen3-14B` in BF16 on `cuda:0` with
  SDPA. `model.model.layers` and configured layer indices are checked at load
  time.
- Qwen's native chat template is called with `enable_thinking=True`.
- Primary decoding is greedy. The `</think>` ID is obtained and verified from
  the tokenizer; reasoning IDs are stored directly and never reconstructed by
  decoding and retokenizing.
- The fixed `Answer:` cue is appended after the closing thinking token. Labels
  A-D are scored by conditional log-probability. A one-forward-pass fast path
  is used only if every candidate is one token; otherwise complete candidate
  sequence log-probabilities are summed.
- Metadata controls replace the hinted label with an invalid neutral label.
  Black-Square controls use four white squares. Candidate markers are accepted
  only when the individual markers are one token and the complete chat-template
  control/hinted prompts have equal token counts.
- The four Black-Square demonstrations in `configs/default.yaml` are fixed
  OpenBookQA **training** records. Evaluation questions are loaded from the
  validation split.
- A stable SHA-256-derived seed chooses the same target hint for both hint
  conditions while excluding the clean model prediction.
- A response is rejected if it lacks `</think>`, hits the token cap, has empty
  reasoning, or ends in a conservative exact repeated-token loop.
- JSONL writes are flushed and synced after every question. Completed question
  IDs are skipped on resume and cannot be appended twice.

## Phase 2 design

For each condition-eligible question and hint type, Phase 2 holds the Phase 1
baseline answer `A0` fixed and evaluates the answer contrast
`Y(P,R) = log p(h|P,R) - log p(A0|P,R)` in two crossed contexts:

- hinted prompt with the neutral/control reasoning trace;
- neutral/control prompt with the hinted reasoning trace.

It then calculates:

- `D_text = Y(P_h,R_0) - Y(P_0,R_0)`;
- `C_text = Y(P_0,R_h) - Y(P_0,R_0)`;
- `F_text = |C_text| / (|C_text| + |D_text| + epsilon)`.

The original prompt and reasoning token IDs are composed directly; reasoning is
never decoded and retokenized. The Phase 1 model ID and, when its resolved
configuration sidecar is present, the answer cue, label prefix, revision, and
labels must match Phase 2. Each question/condition result is independently
checkpointed. JSON and CSV summaries report condition aggregates, paired
Black-Square-minus-Metadata differences, and whether the proposal's expected
aggregate pattern was observed. The expected pattern is a diagnostic, not an
assumption or per-example label.

## Phase 3 design

Selected decoder outputs are captured with temporary forward hooks; global
`output_hidden_states` is never enabled. The matched neutral and hinted inputs
must have identical shapes and teacher-force exactly the same visible neutral
reasoning tokens. All requested reasoning positions are captured for every
selected layer in one forward pass per prompt condition, detached, and moved to
CPU in the configured storage dtype.

The default zero-based layers are `[9, 19, 29]`, corresponding to residual
outputs after decoder blocks 10, 20, and 30. They are not hard-coded: edit
`model.layers` in YAML or pass `--layers`, and model loading validates every
selected index against `model.model.layers`.

All perturbation arithmetic lives in `perturbations.py`:

- the structured direction is `U = H_hint - H_control`;
- four deterministic Rademacher controls are Gram-Schmidt orthogonalized and
  Frobenius-norm matched to the structured direction;
- negligible structured differences are flagged instead of being normalized;
- additive `H + alpha*U` and interpolated replacement are the only numerical
  operations exposed to hooks.

The intervention context manager clones decoder hidden states, changes only the
requested positions, supports a different direction per batch element, handles
tensor and tuple layer outputs, and always removes its hook. In generation mode
it applies once on the prefill containing the source block and leaves cached
single-token decoding calls untouched.

## Phase 4 design

Phase 4 selects exactly one condition-eligible Phase 1 question, one hint type,
one decoder layer, one reasoning block, and the structured direction. Reasoning
is segmented into complete expression units with exact cumulative token-prefix
verification. Adjacent units are merged without truncation when more than ten
are found; a deterministic token-block fallback is recorded if decoded text
does not round-trip to the original generated IDs.

For the selected block, the runner teacher-forces the same neutral reasoning
under the control and hinted prompts and constructs `U = H_hint - H_control`.
It then calculates the answer contrast in three conditions:

- `Y00`: original neutral prompt and neutral reasoning, without intervention;
- `Y10`: original full neutral reasoning, with the source block patched;
- `Y11`: the neutral prefix through the source block is fixed and patched,
  future reasoning is greedily regenerated through `</think>`, and the answer
  is scored while applying the same source patch again.

The saved metrics are `D = Y10 - Y00`, `C = Y11 - Y10`, and
`T = Y11 - Y00`, plus an explicit `T = D + C` numerical check. Before any
nonzero intervention, `alpha=0` must exactly reproduce the original future
reasoning IDs and reproduce `Y00` within configured tolerance in both
unbatched and batched-null paths. A failed null gate is saved with a nonzero CLI
exit status and the structured intervention is not attempted.

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch environment are required for a
real model run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`transformers>=4.51.0` is required for Qwen3 support. No quantization package is
part of the primary experiment.

## Commands

Run the offline unit tests (these do not download a model or dataset):

```bash
PYTHONPATH=src python -m unittest discover -v
```

Run a one-question lightweight-model smoke screen:

```bash
cot-faithfulness-screen \
  --config configs/default.yaml \
  --model-id Qwen/Qwen3-0.6B \
  --layers 3 7 11 \
  --limit 1 \
  --output results/screening/qwen3-0.6b-smoke.jsonl
```

Run the configured Qwen3-14B validation screen:

```bash
cot-faithfulness-screen --config configs/default.yaml
```

After screening has produced eligible records, run Phase 2:

```bash
cot-faithfulness-text-mediation --config configs/default.yaml
```

To smoke-test one eligible condition from a custom Phase 1 file:

```bash
cot-faithfulness-text-mediation \
  --config configs/default.yaml \
  --model-id Qwen/Qwen3-0.6B \
  --layers 3 7 11 \
  --input results/screening/qwen3-0.6b-smoke.jsonl \
  --output results/text_mediation/qwen3-0.6b-smoke.jsonl \
  --summary-json results/text_mediation/qwen3-0.6b-smoke-summary.json \
  --summary-csv results/text_mediation/qwen3-0.6b-smoke-summary.csv \
  --limit 1
```

The model ID must be the one used to create the Phase 1 file. A screening file
with no condition-eligible examples produces an empty summary without running
crossed-context scoring.

After reviewing the Phase 4 code, run one mechanistic smoke case explicitly:

```bash
cot-faithfulness-mechanistic-smoke \
  --config configs/default.yaml \
  --input results/screening/openbookqa_validation.jsonl \
  --hint-condition metadata \
  --layer 9 \
  --block 0 \
  --output results/mechanistic/phase4_smoke.json
```

Omit `--question-id` to select the first eligible record deterministically, or
pass it to inspect a specific eligible question. The default layer is the first
entry of configurable `model.layers` (`9` in the default YAML). For a smaller
Qwen model, pass the same model ID used in Phase 1 and a valid layer, for
example `--model-id Qwen/Qwen3-0.6B --layer 3`.

CLI values override YAML values. The screening command supports model, dataset,
decoding, output, seed, and resume overrides. The text-mediation command
supports model, input/output, summary, eligibility, limit, and resume overrides.
The Phase 4 command supports one question, hint condition, layer, block, alpha,
and null-tolerance overrides. Use `--help` on a command for the complete list.

## Screening output

The output is one nested JSON object per OpenBookQA question. Each record stores:

- the question, choices, ground truth, deterministic hint, and seeds;
- exact prompt and reasoning token IDs plus decoded reasoning text;
- all four answer-label log-probabilities and the predicted answer;
- generation completion, truncation, repetition, and empty-trace flags;
- control/hinted prompt lengths and selected neutral/marked tokens;
- strict-paired and condition-specific eligibility;
- explicit rejection reasons and runtime/model version metadata.

A resolved configuration sidecar is written beside the JSONL file as
`<output>.config.json`. Existing JSONL files require resume mode; disabling
resume never overwrites them.

## Text-mediation output

Phase 2 writes one JSONL record per eligible question and hint condition. Each
record contains all four natural/crossed answer-score distributions,
`D_text`, `C_text`, `F_text`, absolute effect magnitudes, the natural total
effect, an interaction residual, and a near-zero-effect flag. A composite
`question_id|hint_condition` key makes partial runs safely resumable.

The default aggregate artifacts are:

- `results/text_mediation/summary.json` for complete machine-readable results;
- `results/text_mediation/summary.csv` for condition-level analysis.

## Mechanistic smoke output

Phase 4 writes one JSON object, plus a resolved configuration sidecar. The JSON
contains the selected token spans, segmentation audit, structured-direction
norms, exact null generations, all answer-label log-probabilities, `Y00`,
`Y10`, `Y11`, `D`, `C`, `T`, and completion/repetition/truncation flags. Its
`phase_scope` states that Phase 5 batching is disabled.

## Phase boundaries

1. **Dataset and prompt pipeline — complete.**
2. **Text-level mediation — complete.**
3. **Activation capture, perturbation directions, and intervention hooks — complete.**
4. **One-example mechanistic smoke test and Qwen null validation — complete.**
5. Batched mechanistic evaluation and resumable result saving.
6. Full paired experiment, aggregation, bootstrap analysis, plots, and tables.

The project does not train probes or sparse autoencoders, decode activation
semantics, apply a logit lens, or claim that either hint condition is a
per-example ground-truth faithfulness label.
