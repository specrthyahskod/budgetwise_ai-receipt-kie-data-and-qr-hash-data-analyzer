# bill-recognition-model

A trained model for extracting structured fields (vendor, date, address, total) from
photographed receipts — the recognition engine behind BudgetWise AI's receipt scanner.

## Why I built this

My [BudgetWise AI](https://github.com/specrthyahskod/BudgetWise-AI) project had a regex-based
receipt parser, which works on clean, standard-layout receipts but breaks down on anything
crumpled, skewed, or laid out unusually — which is most real receipts. I wanted to replace it
with an actual trained model rather than more regex patches.

## The task

This is a **Key Information Extraction (KIE)** problem: given the words on a receipt and their
positions, label each word as belonging to `COMPANY`, `DATE`, `ADDRESS`, `TOTAL`, or nothing.
It's framed as **token classification**, the same family of task as named-entity recognition,
except the model also sees each word's spatial position on the page — critical here, because
"TOTAL" sitting next to a number in the bottom-right of a receipt is a strong signal a
text-only model can't see.

QR code decoding (also part of the scanner) is **not** part of this model — it's a
deterministic algorithm (`pyzbar`), included in `inference.py` but not trained.

## Model

**LayoutLMv3-base** (Microsoft, via Hugging Face `transformers`), fine-tuned rather than
trained from scratch. It takes text, bounding boxes, *and* the receipt image itself as input,
which matters here more than for plain-text NLP tasks — a human reading a receipt uses layout
constantly, and this model does too.

I chose fine-tuning over training from scratch because the base model already understands
general document layout from large-scale pretraining; a receipt dataset of ~1,000 images is
nowhere near enough to learn that from zero, but is enough to adapt an existing model to this
specific task.

## Dataset

[SROIE](https://rrc.cvc.uab.es/?ch=13) (ICDAR 2019 Scanned Receipts OCR and Information
Extraction) — ~1,000 labeled receipts, the standard benchmark for this task. **Not included in
this repo** (see Limitations) — download it separately and point `prepare_dataset.py` at it.

## Pipeline

```
raw SROIE data (images + boxes + entity labels)
        │
        ▼   scripts/prepare_dataset.py
data/processed.jsonl  (BIO-tagged words + normalized bounding boxes)
        │
        ▼   scripts/train.py
model_out/final/  (fine-tuned LayoutLMv3 checkpoint)
        │
        ▼   scripts/inference.py
{ "COMPANY": ..., "DATE": ..., "TOTAL": ..., "qr_payload": ... }
```

### Running it

```bash
pip install -r requirements.txt

# 1. Convert raw SROIE data (adjust --raw-dir to wherever you extracted it)
python scripts/prepare_dataset.py --raw-dir raw --out data/processed.jsonl

# 2. Spot-check label quality before spending compute on training
python scripts/prepare_dataset.py --raw-dir raw --sample 5

# 3. Fine-tune (Google Colab T4 GPU recommended — CPU will be very slow)
python scripts/train.py --data data/processed.jsonl --epochs 15

# 4. Run on a new receipt
python scripts/inference.py path/to/receipt.jpg --model model_out/final
```

## Evaluation

Reported via seqeval during training: precision, recall, and F1 per entity type, plus overall
token accuracy. [Fill in your actual numbers here once trained — this is the section a
reviewer will look at most closely, so don't leave it blank or vague.]

## Limitations

- SROIE is dominated by Southeast Asian retail receipt formats. Performance on Australian
  receipts (Coles/Woolworths/campus bookshops) is untested until fine-tuned further on
  self-collected, de-identified Australian receipt photos.
- The `assign_label` heuristic in `prepare_dataset.py` matches entity text against OCR'd lines
  via substring matching, which is imperfect — some fraction of training labels are wrong, and
  I corrected this by [describe what you actually did: manual review of N samples, a stricter
  matching threshold, etc. — fill in honestly once you've done it].
- No raw receipt images or the SROIE dataset are committed to this repo (see `.gitignore`) —
  check SROIE's license terms before redistributing it, and never commit real users' receipt
  photos regardless of license, since they can contain personal purchase history.

## License

MIT.
