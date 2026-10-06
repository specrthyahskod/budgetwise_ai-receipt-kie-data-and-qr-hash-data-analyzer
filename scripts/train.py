"""
train.py
Fine-tunes LayoutLMv3-base for receipt key-information extraction.

Run on Google Colab (free T4 GPU is enough for this dataset size):
    !pip install transformers datasets seqeval pillow evaluate -q
    !python train.py --data data/processed.jsonl --epochs 15
"""

import argparse

import numpy as np
from datasets import load_dataset, Features, Sequence, Value, ClassLabel
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

LABELS = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE",
          "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

metric = evaluate.load("seqeval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed.jsonl")
    parser.add_argument("--output-dir", default="model_out")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    features = Features({
        "id": Value("string"),
        "image_path": Value("string"),
        "words": Sequence(Value("string")),
        "bboxes": Sequence(Sequence(Value("int64"))),
        "ner_tags": Sequence(ClassLabel(names=LABELS)),
    })
    dataset = load_dataset("json", data_files=args.data, features=features)["train"]
    dataset = dataset.train_test_split(test_size=args.test_size, seed=42)

    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )

    def encode(batch):
        images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
        encoded = processor(
            images, batch["words"], boxes=batch["bboxes"], word_labels=batch["ner_tags"],
            truncation=True, padding="max_length", max_length=512,
        )
        return encoded

    dataset = dataset.map(encode, batched=True, remove_columns=dataset["train"].column_names)
    dataset.set_format(type="torch")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", id2label=ID2LABEL, label2id=LABEL2ID
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions, true_labels = [], []
        for pred_row, label_row in zip(predictions, labels):
            preds, labs = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                preds.append(ID2LABEL[p])
                labs.append(ID2LABEL[l])
            true_predictions.append(preds)
            true_labels.append(labs)

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if results is None:
            results = {}
        return {
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
            "accuracy": results.get("overall_accuracy", 0.0),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Final eval metrics:", eval_results)

    trainer.save_model(f"{args.output_dir}/final")
    processor.save_pretrained(f"{args.output_dir}/final")
    print(f"Saved model to {args.output_dir}/final")


if __name__ == "__main__":
    main()