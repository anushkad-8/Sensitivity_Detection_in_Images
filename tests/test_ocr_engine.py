"""
tests/test_ocr_engine.py
------------------------
Sanity-check tests for Module 2: ocr_engine.py

Run from project root:
    python tests/test_ocr_engine.py input/your_image.png

Tests:
    1. Invalid input (None) raises ValueError
    2. Invalid input (wrong type) raises ValueError
    3. Real image runs end-to-end without error
    4. Output dict has required keys: "text" and "words"
    5. "text" is a non-empty string
    6. "words" DataFrame has required columns
    7. All confidence values in filtered output are >= threshold
    8. find_word_boxes() returns correct subset
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.preprocess import preprocess_image
from modules.ocr_engine import run_ocr, find_word_boxes

CONFIDENCE_THRESHOLD = 60
REQUIRED_WORD_COLUMNS = {"word", "left", "top", "width", "height", "conf"}


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_none_input():
    print("\n[Test 1] ValueError on None input...")
    try:
        run_ocr(None)
        print("  FAIL — No error raised!")
    except ValueError as e:
        print(f"  PASS — Caught ValueError: {e}")


def test_wrong_type_input():
    print("\n[Test 2] ValueError on wrong type input...")
    try:
        run_ocr("not_an_image")
        print("  FAIL — No error raised!")
    except ValueError as e:
        print(f"  PASS — Caught ValueError: {e}")


def test_end_to_end(image_path: str):
    print(f"\n[Test 3] End-to-end run on: {image_path}")
    try:
        clean = preprocess_image(image_path, save_debug=False)
        result = run_ocr(clean, confidence_threshold=CONFIDENCE_THRESHOLD)
        print("  PASS — run_ocr completed without error.")
        return result
    except Exception as e:
        print(f"  FAIL — Exception: {e}")
        return None


def test_output_keys(result: dict):
    print("\n[Test 4] Checking output dict has 'text' and 'words'...")
    if result is None:
        print("  SKIP")
        return
    if "text" in result and "words" in result:
        print("  PASS — Both keys present.")
    else:
        print(f"  FAIL — Keys found: {list(result.keys())}")


def test_text_is_string(result: dict):
    print("\n[Test 5] Checking 'text' is a non-empty string...")
    if result is None:
        print("  SKIP")
        return
    text = result.get("text", None)
    if isinstance(text, str) and len(text.strip()) > 0:
        print(f"  PASS — Text length: {len(text)} chars")
    elif isinstance(text, str):
        print("  WARN — Text is a string but empty. Check image quality or PSM setting.")
    else:
        print(f"  FAIL — 'text' is not a string. Got: {type(text)}")


def test_words_columns(result: dict):
    print("\n[Test 6] Checking 'words' DataFrame has required columns...")
    if result is None:
        print("  SKIP")
        return
    df = result.get("words")
    if not isinstance(df, pd.DataFrame):
        print(f"  FAIL — 'words' is not a DataFrame. Got: {type(df)}")
        return
    actual_cols = set(df.columns)
    missing = REQUIRED_WORD_COLUMNS - actual_cols
    if not missing:
        print(f"  PASS — All required columns present: {sorted(actual_cols)}")
    else:
        print(f"  FAIL — Missing columns: {missing}")


def test_confidence_filter(result: dict):
    print(f"\n[Test 7] Checking all words have conf >= {CONFIDENCE_THRESHOLD}...")
    if result is None:
        print("  SKIP")
        return
    df = result.get("words")
    if df is None or df.empty:
        print("  SKIP — words DataFrame is empty.")
        return
    below = df[df["conf"] < CONFIDENCE_THRESHOLD]
    if below.empty:
        print(f"  PASS — All {len(df)} words meet confidence threshold.")
    else:
        print(f"  FAIL — {len(below)} words below threshold found.")
        print(below[["word", "conf"]].to_string(index=False))


def test_find_word_boxes(result: dict):
    print("\n[Test 8] Testing find_word_boxes() helper...")
    if result is None:
        print("  SKIP")
        return
    df = result.get("words")
    if df is None or df.empty:
        print("  SKIP — words DataFrame is empty.")
        return

    # Pick the first 2 real words as fake "sensitive" matches
    sample_targets = df["word"].head(2).tolist()
    matched = find_word_boxes(df, sample_targets)

    if isinstance(matched, pd.DataFrame) and not matched.empty:
        print(f"  PASS — Matched {len(matched)} boxes for targets: {sample_targets}")
        print(matched[["word", "left", "top", "width", "height"]].to_string(index=False))
    else:
        print(f"  FAIL — No boxes returned for targets: {sample_targets}")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 50)
    print("  OCR Engine — Module 2 Tests")
    print("═" * 50)

    # Tests that don't need a real image
    test_none_input()
    test_wrong_type_input()

    # Tests that need a real image
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
        result = test_end_to_end(img_path)
        test_output_keys(result)
        test_text_is_string(result)
        test_words_columns(result)
        test_confidence_filter(result)
        test_find_word_boxes(result)
    else:
        print("\n[Tests 3–8] Skipped — pass an image path to run full tests.")
        print("  Usage: python tests/test_ocr_engine.py input/your_image.png")

    print("\n── All tests done ──")