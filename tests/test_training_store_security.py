"""
tests/test_training_store_security.py
-------------------------------------
Security-focused tests for training_store.py.

Run from project root:
    python tests/test_training_store_security.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from modules.sensitive_detector import detect_sensitive
from modules.training_store import (
    _build_record,
    _dedupe_matches,
    _sanitize_training_text,
)


PASS = "  PASS"
FAIL = "  FAIL"


def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} - {label}")
    return condition


SENSITIVE_TEXT = (
    "Employee PAN ABCDE1234F email sample@gmail.com "
    "passport J8369854 phone 9876543210"
)


def test_1_sanitize_masks_detected_values():
    print("\n[Test 1] Sanitized OCR text masks detected values...")
    detection = detect_sensitive(SENSITIVE_TEXT)
    safe_text = _sanitize_training_text(SENSITIVE_TEXT, detection["matches"])

    for raw in ["ABCDE1234F", "sample@gmail.com", "J8369854", "9876543210"]:
        check(raw not in safe_text, f"Raw value not stored: {raw}")
    check("<PAN:" in safe_text or "<EMAIL:" in safe_text,
          "Masked placeholders are present")


def test_2_record_stores_sanitized_full_text():
    print("\n[Test 2] Training record full_ocr_text is sanitized...")
    detection = detect_sensitive(SENSITIVE_TEXT)
    safe_text = _sanitize_training_text(SENSITIVE_TEXT, detection["matches"])
    match = detection["matches"][0]
    record = _build_record("sample.jpg", match, safe_text)

    for raw in ["ABCDE1234F", "sample@gmail.com", "J8369854", "9876543210"]:
        check(raw not in record["full_ocr_text"], f"full_ocr_text excludes {raw}")
        check(raw not in record["ocr_text_window"], f"ocr_text_window excludes {raw}")
    check(record["finding_value"].startswith("<"), "finding_value remains masked")


def test_3_dedupe_matches_prevents_duplicate_records():
    print("\n[Test 3] Duplicate matches are removed before storage...")
    match = {
        "type": "email",
        "value": "sample@gmail.com",
        "tokens": ["sample@gmail.com"],
        "confidence": "high",
        "source": "regex",
    }
    deduped = _dedupe_matches([match, dict(match), dict(match)])
    check(len(deduped) == 1, f"deduped length is 1 (got {len(deduped)})")


if __name__ == "__main__":
    print("=" * 60)
    print("  Training Store Security Tests")
    print("=" * 60)

    test_1_sanitize_masks_detected_values()
    test_2_record_stores_sanitized_full_text()
    test_3_dedupe_matches_prevents_duplicate_records()

    print("\n-- All security tests done --")
