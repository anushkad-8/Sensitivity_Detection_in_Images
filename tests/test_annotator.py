"""
tests/test_annotator.py
------------------------
Sanity-check tests for Module 4: annotator.py

Run from project root:
    python tests/test_annotator.py input/test1_idcard.jpg

Tests:
    1.  Invalid mode raises ValueError
    2.  Missing image raises FileNotFoundError
    3.  No findings → saves clean image, returns valid path
    4.  Highlight mode runs end-to-end, saves output file
    5.  Redact mode runs end-to-end, saves output file
    6.  Output file actually exists on disk after save
    7.  Output image dimensions match original (nothing got cropped)
    8.  Annotated image is BGR (3-channel), not grayscale
    9.  Merged box logic: multi-token match produces one box
    10. Missing token gracefully skipped (no crash)
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.preprocess import preprocess_image
from modules.ocr_engine import run_ocr
from modules.sensitive_detector import detect_sensitive
from modules.annotator import annotate_image, _get_merged_box

PASS = "  PASS ✓"
FAIL = "  FAIL ✗"

def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} — {label}")
    return condition


# ─────────────────────────────────────────────
# FIXTURES — built once, reused across tests
# ─────────────────────────────────────────────

def build_fixtures(image_path: str):
    """Run the full pipeline to get real OCR + detection results."""
    clean      = preprocess_image(image_path, save_debug=False)
    ocr_result = run_ocr(clean)
    detection  = detect_sensitive(ocr_result["text"])
    return ocr_result, detection


def make_empty_detection():
    return {"is_sensitive": False, "total": 0, "matches": []}


def make_fake_detection():
    return {
        "is_sensitive": True,
        "total": 1,
        "matches": [{
            "type"      : "email",
            "value"     : "sample@gmail.com",
            "tokens"    : ["sample@gmail.com"],
            "confidence": "high",
            "source"    : "regex"
        }]
    }


def make_fake_words_df():
    """Minimal words DataFrame with one known word."""
    return pd.DataFrame([{
        "word"  : "sample@gmail.com",
        "left"  : 100,
        "top"   : 200,
        "width" : 120,
        "height": 18,
        "conf"  : 92.0
    }])


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_invalid_mode(image_path: str):
    print("\n[Test 1] Invalid mode raises ValueError...")
    try:
        annotate_image(image_path, make_empty_detection(), pd.DataFrame(), mode="blur")
        print(f"  FAIL — No error raised!")
    except ValueError as e:
        print(f"  PASS ✓ — Caught ValueError: {e}")


def test_missing_image():
    print("\n[Test 2] Missing image raises FileNotFoundError...")
    try:
        annotate_image("input/does_not_exist.jpg", make_empty_detection(), pd.DataFrame())
        print("  FAIL — No error raised!")
    except FileNotFoundError as e:
        print(f"  PASS ✓ — Caught FileNotFoundError: {e}")


def test_no_findings(image_path: str):
    print("\n[Test 3] No findings → saves clean image...")
    out = annotate_image(image_path, make_empty_detection(), pd.DataFrame(), mode="highlight")
    check(out is not None and isinstance(out, str), "Returns a string path")
    check(os.path.exists(out), f"Clean file saved at: {out}")


def test_highlight_mode(image_path: str, ocr_result: dict, detection: dict):
    print("\n[Test 4] Highlight mode end-to-end...")
    if not detection["is_sensitive"]:
        print("  SKIP — no sensitive content in test image.")
        return None
    out = annotate_image(image_path, detection, ocr_result["words"], mode="highlight")
    check(isinstance(out, str), "Returns output path string")
    check(os.path.exists(out), f"Highlighted image saved: {out}")
    return out


def test_redact_mode(image_path: str, ocr_result: dict, detection: dict):
    print("\n[Test 5] Redact mode end-to-end...")
    if not detection["is_sensitive"]:
        print("  SKIP — no sensitive content in test image.")
        return None
    out = annotate_image(image_path, detection, ocr_result["words"], mode="redact")
    check(isinstance(out, str), "Returns output path string")
    check(os.path.exists(out), f"Redacted image saved: {out}")
    return out


def test_output_file_exists(highlight_path: str, redact_path: str):
    print("\n[Test 6] Output files exist on disk...")
    if highlight_path:
        check(os.path.exists(highlight_path), f"Highlight output exists: {highlight_path}")
    if redact_path:
        check(os.path.exists(redact_path),   f"Redact output exists:    {redact_path}")


def test_output_dimensions_match(image_path: str, output_path: str):
    print("\n[Test 7] Output dimensions match original image...")
    if not output_path or not os.path.exists(output_path):
        print("  SKIP — no output path.")
        return
    orig = cv2.imread(image_path)
    out  = cv2.imread(output_path)
    if orig is None or out is None:
        print("  SKIP — could not load images.")
        return
    check(
        orig.shape[0] == out.shape[0] and orig.shape[1] == out.shape[1],
        f"Dimensions match: original {orig.shape[:2]} == output {out.shape[:2]}"
    )


def test_output_is_color(output_path: str):
    print("\n[Test 8] Output image is BGR (3-channel, not grayscale)...")
    if not output_path or not os.path.exists(output_path):
        print("  SKIP — no output path.")
        return
    img = cv2.imread(output_path)
    check(len(img.shape) == 3 and img.shape[2] == 3,
          f"Image is 3-channel BGR. Shape: {img.shape}")


def test_merged_box_multi_token():
    print("\n[Test 9] Merged box spans all tokens of a multi-token match...")
    words_df = pd.DataFrame([
        {"word": "+91",   "left": 100, "top": 50, "width": 30,  "height": 14, "conf": 90},
        {"word": "99999", "left": 135, "top": 50, "width": 50,  "height": 14, "conf": 92},
        {"word": "99999", "left": 190, "top": 50, "width": 50,  "height": 14, "conf": 91},
    ])
    box = _get_merged_box(["+91", "99999"], words_df)
    check(box is not None, "Box returned (not None)")
    if box:
        check(box["left"] <= 100,  f"Left edge covers first token (left={box['left']})")
        check(box["right"] >= 240, f"Right edge covers last token (right={box['right']})")
        print(f"  Merged box: {box}")


def test_missing_token_graceful(image_path: str):
    print("\n[Test 10] Missing token gracefully skipped — no crash...")
    detection = {
        "is_sensitive": True,
        "total": 1,
        "matches": [{
            "type"      : "pan",
            "value"     : "ZZZZZ9999Z",        # Token not in OCR output
            "tokens"    : ["ZZZZZ9999Z"],
            "confidence": "high",
            "source"    : "regex"
        }]
    }
    empty_words = pd.DataFrame(columns=["word", "left", "top", "width", "height", "conf"])
    try:
        out = annotate_image(image_path, detection, empty_words, mode="highlight")
        print(f"  PASS ✓ — No crash. Output: {out}")
    except Exception as e:
        print(f"  FAIL — Unexpected exception: {e}")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 55)
    print("  Annotator — Module 4 Tests")
    print("═" * 55)

    # Tests that don't need a real image
    test_missing_image()
    test_merged_box_multi_token()

    if len(sys.argv) < 2:
        print("\n[Tests 1, 3–10] Skipped — pass an image path to run full tests.")
        print("  Usage: python tests/test_annotator.py input/your_image.png")
        sys.exit(0)

    img_path = sys.argv[1]
    print(f"\nBuilding pipeline fixtures from: {img_path}")
    ocr_result, detection = build_fixtures(img_path)
    print(f"Detection result: is_sensitive={detection['is_sensitive']}, total={detection['total']}")

    test_invalid_mode(img_path)
    test_no_findings(img_path)
    h_path = test_highlight_mode(img_path, ocr_result, detection)
    r_path = test_redact_mode(img_path, ocr_result, detection)
    test_output_file_exists(h_path, r_path)
    test_output_dimensions_match(img_path, h_path)
    test_output_is_color(h_path)
    test_missing_token_graceful(img_path)

    print("\n── All 10 tests done ──")