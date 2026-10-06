import argparse
import json
from pathlib import Path

from PIL import Image

LABELS = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE",
          "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]


def load_boxes(box_path: Path):
    """Parse a SROIE box file into (text, (x0,y0,x1,y1)) tuples."""
    entries = []
    for line in box_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(",", 8)
        if len(parts) < 9:
            continue
        coords = list(map(float, parts[:8]))
        text = parts[8]
        xs, ys = coords[0::2], coords[1::2]
        entries.append((text, (min(xs), min(ys), max(xs), max(ys))))
    return entries


def normalize_box(box, width, height):
    x0, y0, x1, y1 = box
    return [
        int(1000 * x0 / width), int(1000 * y0 / height),
        int(1000 * x1 / width), int(1000 * y1 / height),
    ]


def assign_label(text: str, entities: dict) -> str:
    """Heuristic: does this line's text match (or sit inside) one of the
    known entity values? Returns the BIO-start label for that field, or O."""
    clean = text.strip().lower()
    if not clean:
        return "O"
    for field_name, key in (("company", "COMPANY"), ("date", "DATE"),
                             ("address", "ADDRESS"), ("total", "TOTAL")):
        value = str(entities.get(field_name, "")).strip().lower()
        if value and (clean == value or clean in value or value in clean):
            return key
    return "O"


def line_to_tokens(text: str, label_key: str):
    """Split a line into words and apply BIO tagging across them."""
    words = text.split()
    if not words:
        return [], []
    if label_key == "O":
        return words, ["O"] * len(words)
    tags = [f"B-{label_key}"] + [f"I-{label_key}"] * (len(words) - 1)
    return words, tags


def process_receipt(img_path: Path, box_path: Path, entity_path: Path):
    entities = json.loads(entity_path.read_text(encoding="utf-8", errors="ignore"))
    with Image.open(img_path) as im:
        width, height = im.size

    words, bboxes, ner_tags = [], [], []
    for text, box in load_boxes(box_path):
        label_key = assign_label(text, entities)
        line_words, line_tags = line_to_tokens(text, label_key)
        norm_box = normalize_box(box, width, height)
        words.extend(line_words)
        ner_tags.extend(line_tags)
        bboxes.extend([norm_box] * len(line_words))  # same box for every word on the line

    return {
        "id": img_path.stem,
        "image_path": str(img_path),
        "words": words,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="raw", help="Folder containing img/ box/ entities/")
    parser.add_argument("--out", default="data/processed.jsonl")
    parser.add_argument("--sample", type=int, default=0,
                         help="If set, print N processed examples instead of writing the file, for spot-checking labels")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    img_dir, box_dir, entity_dir = raw / "img", raw / "box", raw / "entities"

    records = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        box_path, entity_path = box_dir / f"{stem}.txt", entity_dir / f"{stem}.txt"
        if not (box_path.exists() and entity_path.exists()):
            continue
        records.append(process_receipt(img_path, box_path, entity_path))

    if args.sample:
        for r in records[: args.sample]:
            print(json.dumps(r, indent=2))
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} receipts to {out_path}")


if __name__ == "__main__":
    main()
