"""
inference.py
Runs a trained receipt KIE model (+ QR decoding) on a single receipt image.

    python inference.py path/to/receipt.jpg --model model_out/final
"""

import argparse
import json
from collections import defaultdict

import pytesseract
from PIL import Image
from pyzbar.pyzbar import decode as decode_qr
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch


def ocr_words_and_boxes(image: Image.Image):
    """Run Tesseract and return (words, boxes normalized to 0-1000)."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    width, height = image.size
    words, boxes = [], []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        box = [
            int(1000 * x / width), int(1000 * y / height),
            int(1000 * (x + w) / width), int(1000 * (y + h) / height),
        ]
        words.append(text)
        boxes.append(box)
    return words, boxes


def group_entities(words, labels):
    """Collapse B-/I- tagged words back into whole entity strings."""
    entities = defaultdict(list)
    current_field, current_words = None, []

    def flush():
        if current_field and current_words:
            entities[current_field].append(" ".join(current_words))

    for word, label in zip(words, labels):
        if label == "O":
            flush()
            current_field, current_words = None, []
        elif label.startswith("B-"):
            flush()
            current_field = label[2:]
            current_words = [word]
        elif label.startswith("I-") and label[2:] == current_field:
            current_words.append(word)
        else:  # stray I- tag with no matching B- — start fresh rather than drop it
            flush()
            current_field = label[2:]
            current_words = [word]
    flush()

    return {field: " ".join(parts) for field, parts in entities.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default="model_out/final")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    qr_results = decode_qr(image)
    qr_payload = qr_results[0].data.decode("utf-8", errors="ignore") if qr_results else None

    words, boxes = ocr_words_and_boxes(image)

    processor = LayoutLMv3Processor.from_pretrained(args.model, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model)
    model.eval()

    encoded = processor(image, words, boxes=boxes, truncation=True,
                         padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
    predictions = logits.argmax(-1).squeeze().tolist()

    word_ids = encoded.word_ids(0)
    if word_ids is None:
        raise RuntimeError("Processor did not return word-to-token alignment; check the input.")

    id2label = model.config.id2label or {}

    seen = set()
    final_words, final_labels = [], []
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        final_words.append(words[wid])
        final_labels.append(id2label[predictions[idx]])

    result: dict = group_entities(final_words, final_labels)
    result["qr_payload"] = qr_payload

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()