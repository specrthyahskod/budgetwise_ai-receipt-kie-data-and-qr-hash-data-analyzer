```markdown
# Bill Recognition Model 🧾

A trained AI model that reads photographed receipts and extracts key details like the **store name, date, address, and total amount**.

This is the machine learning engine behind the receipt scanner feature in my main app, [BudgetWise AI](https://github.com/specrthyahskod/BudgetWise-AI).

---

## Why I Built This

In my main BudgetWise AI app, I originally used simple text pattern matching (regex) to pull details from receipts. That worked okay on clean, digital receipts, but in real life, paper receipts get crumpled, folded, faded, or printed in unusual layouts.

Instead of constantly writing more regex rules every time a receipt failed to read, I built this model so the app can actually understand the visual layout and text of a receipt just like a human does.

---

## How It Works

This project solves what is called **Key Information Extraction (KIE)**.

When you look at a receipt, you don't just read the words—you look at *where* they are on the page. For example, a number next to the word "TOTAL" at the bottom right is almost certainly the final price.

The model reads two things at the same time:
1. The **text** recognized from the receipt.
2. The **visual position** (coordinates and bounding boxes) of each word on the image.

It then labels each word into one of these categories:
- `COMPANY` (Store or merchant name)
- `DATE` (Transaction date)
- `ADDRESS` (Store location)
- `TOTAL` (Final amount paid)
- `O` (Other / irrelevant text)

*(Note: QR code decoding is also included in the inference script via `pyzbar` for digital invoices, but that uses standard barcode reading rather than this machine learning model.)*

---

## The Model

I used **LayoutLMv3-base** from Microsoft (via Hugging Face).

Instead of training a model from scratch—which requires huge computing power and tens of thousands of images—I used **transfer learning (fine-tuning)**. LayoutLMv3 was already pre-trained on millions of document pages to understand general layouts. I fine-tuned it on labeled receipt data so it specializes in store bills.

---

## Dataset

I used the **SROIE dataset** (from the ICDAR 2019 competition), which has around 1,000 scanned receipts with labeled text boxes.

*Because of dataset size and licensing rules, the raw dataset is not hosted directly inside this repo. You can download SROIE separately and run the preprocessing script below.*

---

## Project Flow


```

Raw SROIE Dataset (images + labeled boxes)
│
▼  scripts/prepare_dataset.py
data/processed.jsonl (words + bounding box coordinates)
│
▼  scripts/train.py
model_out/final/ (fine-tuned LayoutLMv3 model)
│
▼  scripts/inference.py
Output: { "COMPANY": ..., "DATE": ..., "TOTAL": ..., "qr_payload": ... }

```

---

## How to Run It

### 1. Setup
Install the required dependencies:
```bash
pip install -r requirements.txt

```

### 2. Prepare the Data

Convert the raw dataset into the format the model expects:

```bash
# Convert raw data (point --raw-dir to your downloaded SROIE folder)
python scripts/prepare_dataset.py --raw-dir raw --out data/processed.jsonl

# Preview 5 samples to make sure labels look right
python scripts/prepare_dataset.py --raw-dir raw --sample 5

```

### 3. Train the Model

Train the model (Google Colab with a free T4 GPU is recommended):

```bash
python scripts/train.py --data data/processed.jsonl --epochs 15

```

### 4. Test on a Receipt

Run inference on any receipt image to extract the fields:

```bash
python scripts/inference.py path/to/receipt.jpg --model model_out/final

```

---

## Results & Accuracy

I evaluated the model using standard precision, recall, and F1 scores on the validation set:

* **Overall Token Accuracy:** [Fill your accuracy, e.g., 94%]
* **Store Name (COMPANY) F1:** [Fill your score, e.g., 88%]
* **Date F1:** [Fill your score, e.g., 91%]
* **Total Amount F1:** [Fill your score, e.g., 85%]

*(Replace bracketed numbers with your actual training output)*

---

## Current Limitations

* **Different Receipt Styles:** SROIE mostly features retail receipts from Southeast Asia. Australian store receipts (like Coles, Woolworths, or campus shops) can have slightly different layouts. I plan to collect local test samples to fine-tune it further.
* **OCR Label Matching:** The script that matches ground-truth text to OCR bounding boxes uses substring matching, which can occasionally mislabel words if the OCR makes a spelling typo.
* **Privacy:** To protect user privacy, never upload real receipts containing personal names, card numbers, or sensitive purchase history to this repository.

---

## License

MIT

```

```