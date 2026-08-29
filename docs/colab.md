# Colab execution guide (Phases 1–4)

Use a GPU runtime and persist outputs to Google Drive because screening may
outlive a Colab session.

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone the repository, install it, and confirm the offline tests:

```bash
git clone <repository-url> /content/mechanistic-cot-faithfulness
cd /content/mechanistic-cot-faithfulness
pip install -e .
PYTHONPATH=src python -m unittest discover -v
```

First run one question with a small model:

```bash
cot-faithfulness-screen \
  --config configs/default.yaml \
  --model-id Qwen/Qwen3-0.6B \
  --layers 3 7 11 \
  --limit 1 \
  --output /content/drive/MyDrive/cot-faithfulness/qwen3-0.6b-smoke.jsonl
```

Then start or resume the Qwen3-14B screen:

```bash
cot-faithfulness-screen \
  --config configs/default.yaml \
  --output /content/drive/MyDrive/cot-faithfulness/qwen3-14b-screening.jsonl \
  --resume
```

The runner syncs each completed question to JSONL and skips completed IDs after
a restart. It also writes the resolved configuration beside the output.

After screening, run the crossed-context text-mediation phase with the same
model and Phase 1 JSONL:

```bash
cot-faithfulness-text-mediation \
  --config configs/default.yaml \
  --input /content/drive/MyDrive/cot-faithfulness/qwen3-14b-screening.jsonl \
  --output /content/drive/MyDrive/cot-faithfulness/qwen3-14b-text-mediation.jsonl \
  --summary-json /content/drive/MyDrive/cot-faithfulness/qwen3-14b-text-summary.json \
  --summary-csv /content/drive/MyDrive/cot-faithfulness/qwen3-14b-text-summary.csv \
  --resume
```

The Phase 2 runner reuses exact stored reasoning token IDs, writes one terminal
record after each eligible hint condition, and can resume after a disconnected
session. Phase 3 provides capture, perturbation, and hook primitives.

After inspecting the code, Phase 4 can be invoked for exactly one case:

```bash
cot-faithfulness-mechanistic-smoke \
  --config configs/default.yaml \
  --input /content/drive/MyDrive/cot-faithfulness/qwen3-14b-screening.jsonl \
  --hint-condition metadata \
  --layer 9 \
  --block 0 \
  --output /content/drive/MyDrive/cot-faithfulness/qwen3-14b-phase4-smoke.json
```

This command first runs the unbatched and batched `alpha=0` gate. It does not
attempt the structured intervention if exact future-reasoning reproduction or
answer-score tolerance fails. It writes one JSON result and does not start the
Phase 5 batched experiment.

Capture layers are configurable. The primary run defaults to zero-based layers
9, 19, and 29; for smaller Qwen3 variants, set valid indices in `model.layers`
or pass `--layers` to commands that load the model.
