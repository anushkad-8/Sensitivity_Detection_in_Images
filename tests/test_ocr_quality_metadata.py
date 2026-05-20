"""
tests/test_ocr_quality_metadata.py
----------------------------------
Tests for OCR dropped-word metadata used by confidence_engine.py.

Run from project root:
    python tests/test_ocr_quality_metadata.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.ocr_engine import _build_quality_metadata, _filter_words


PASS = "  PASS"
FAIL = "  FAIL"


def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} - {label}")
    return condition


def make_raw_words() -> pd.DataFrame:
    return pd.DataFrame([
        {"text": "ABCDE1234F", "left": 1, "top": 1, "width": 10, "height": 8, "conf": 92},
        {"text": "garbled", "left": 1, "top": 15, "width": 10, "height": 8, "conf": 21},
        {"text": "sample@gmail.com", "left": 1, "top": 30, "width": 10, "height": 8, "conf": 88},
        {"text": "", "left": 1, "top": 45, "width": 10, "height": 8, "conf": -1},
        {"text": None, "left": 1, "top": 60, "width": 10, "height": 8, "conf": -1},
    ])


def test_1_quality_counts_use_non_empty_words():
    print("\n[Test 1] Raw count excludes layout blanks...")
    raw = make_raw_words()
    filtered = _filter_words(raw, confidence_threshold=60)
    quality = _build_quality_metadata(raw, filtered)

    check(quality["raw_word_count"] == 3,
          f"raw_word_count=3 (got {quality['raw_word_count']})")
    check(quality["filtered_word_count"] == 2,
          f"filtered_word_count=2 (got {quality['filtered_word_count']})")
    check(quality["dropped_words"] == 1,
          f"dropped_words=1 (got {quality['dropped_words']})")


def test_2_drop_ratio_is_exact():
    print("\n[Test 2] Dropped word ratio is calculated from real counts...")
    raw = make_raw_words()
    filtered = _filter_words(raw, confidence_threshold=60)
    quality = _build_quality_metadata(raw, filtered)

    check(quality["dropped_word_ratio"] == 0.333,
          f"dropped_word_ratio=0.333 (got {quality['dropped_word_ratio']})")


if __name__ == "__main__":
    print("=" * 60)
    print("  OCR Quality Metadata Tests")
    print("=" * 60)

    test_1_quality_counts_use_non_empty_words()
    test_2_drop_ratio_is_exact()

    print("\n-- All OCR metadata tests done --")
