import io
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union

from PIL import Image, ImageOps, UnidentifiedImageError
import pytesseract
from pyzbar.pyzbar import decode as decode_qr

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — reject anything larger before processing
MAX_DIMENSION = 1600  # downscale so the long edge is at most this many pixels


class InvalidImageError(ValueError):
    """Raised when the ingested file isn't a usable image."""


@dataclass
class LineItem:
    description: str
    amount: float
    category: Optional[str] = None


@dataclass
class ParsedReceipt:
    vendor: Optional[str] = None
    date: Optional[datetime] = None
    total: Optional[float] = None
    line_items: List[LineItem] = field(default_factory=list)
    qr_payload: Optional[str] = None
    raw_text: str = ""


def load_image(source: Union[str, bytes, io.BytesIO]) -> Image.Image:
    if isinstance(source, (bytes, bytearray)):
        if len(source) > MAX_UPLOAD_BYTES:
            raise InvalidImageError(f"Image exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit.")
        source = io.BytesIO(source)
    elif isinstance(source, str):
        if os.path.getsize(source) > MAX_UPLOAD_BYTES:
            raise InvalidImageError(f"Image exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit.")

    try:
        image = Image.open(source)
        image.load()  
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError(f"File is not a readable image: {e}") from e

    # Phone cameras often store orientation in EXIF rather than rotating pixels;
    # without this, a sideways photo produces garbage OCR output.
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Downscale oversized photos — OCR/QR accuracy doesn't improve past a point,
    # and large images slow down every downstream step for no benefit.
    if max(image.size) > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

    return image


class ReceiptScanner:
    """Ingests a receipt image, extracts QR codes and OCR text, and parses
    structured fields (vendor, date, total, line items) via heuristics."""

    KNOWN_VENDORS = [
        "coles", "woolworths", "aldi", "iga", "kmart", "target",
        "unsw", "usyd", "co-op bookshop", "jb hi-fi", "officeworks",
    ]

    DATE_PATTERNS = [
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(\d{4}-\d{2}-\d{2})",
    ]

    TOTAL_KEYWORDS = ["total", "amount due", "balance due", "grand total"]
    NON_ITEM_KEYWORDS = TOTAL_KEYWORDS + ["gst", "subtotal", "change", "cash", "eftpos", "tax"]

    def scan(self, source: Union[str, bytes, io.BytesIO]) -> ParsedReceipt:
        """
        source: a file path, raw image bytes, or a BytesIO stream (e.g. straight
        off an UploadFile in FastAPI: `await file.read()`).
        """
        image = load_image(source)
        receipt = ParsedReceipt()

        # Step 1: If a receipt carries a tax-invoice QR code, it's
        # the most reliable source — worth capturing even if we don't parse
        # its payload format yet (that's vendor-specific metadata).
        qr_results = decode_qr(image)
        if qr_results:
            receipt.qr_payload = qr_results[0].data.decode("utf-8", errors="ignore")

        # Step 2: OCR always runs too, since QR codes rarely carry detailed line items.
        receipt.raw_text = pytesseract.image_to_string(image)

        # Step 3: Parsing of the OCR text data.
        receipt.vendor = self._extract_vendor(receipt.raw_text)
        receipt.date = self._extract_date(receipt.raw_text)
        receipt.total = self._extract_total(receipt.raw_text)
        receipt.line_items = self._extract_line_items(receipt.raw_text)

        return receipt

    def _extract_vendor(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for vendor in self.KNOWN_VENDORS:
            if vendor in lowered:
                return vendor.title()
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line 
        return None

    def _extract_date(self, text: str) -> Optional[datetime]:
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text)
            if not match:
                continue
            raw = match.group(1)
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(raw, fmt)
                except ValueError:
                    continue
        return None

    def _extract_total(self, text: str) -> Optional[float]:
        lines = text.lower().splitlines()
        for line in lines:
            if any(keyword in line for keyword in self.TOTAL_KEYWORDS):
                amount = self._find_amount(line)
                if amount is not None:
                    return amount
        amounts = [a for a in (self._find_amount(l) for l in lines) if a is not None]
        return max(amounts) if amounts else None

    @staticmethod
    def _find_amount(line: str) -> Optional[float]:
        match = re.search(r"\$?\s?(\d+\.\d{2})", line)
        return float(match.group(1)) if match else None

    def _extract_line_items(self, text: str) -> List[LineItem]:
        items = []
        for line in text.splitlines():
            amount = self._find_amount(line)
            if amount is None:
                continue
            if any(keyword in line.lower() for keyword in self.NON_ITEM_KEYWORDS):
                continue
            description = re.sub(r"\$?\s?\d+\.\d{2}", "", line).strip(" -:\t")
            if description:
                items.append(LineItem(description=description, amount=amount))
        return items


def categorize_items(receipt: ParsedReceipt, classifier) -> ParsedReceipt:
    """Route each parsed line item through BudgetWise's existing Naive Bayes
    category classifier (models/category_classifier.py or similar) — the
    same model already used for manual transaction entry, so a scanned
    receipt produces categorized entries with zero extra training."""
    for item in receipt.line_items:
        item.category = classifier.predict([item.description])[0]
    return receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan a receipt image and print the parsed fields.")
    parser.add_argument("image", help="Path to a receipt photo (jpg/png)")
    args = parser.parse_args()

    scanner = ReceiptScanner()
    try:
        result = scanner.scan(args.image)
    except InvalidImageError as e:
        print(f"Could not process image: {e}")
        raise SystemExit(1)

    print(f"Vendor: {result.vendor}")
    print(f"Date:   {result.date}")
    print(f"Total:  ${result.total}" if result.total is not None else "Total:  (not found)")
    print(f"QR:     {result.qr_payload or '(none found)'}")
    print("Line items:")
    for item in result.line_items:
        tag = f" [{item.category}]" if item.category else ""
        print(f"  - {item.description}: ${item.amount}{tag}")