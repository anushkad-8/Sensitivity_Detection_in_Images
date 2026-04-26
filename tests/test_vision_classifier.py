"""
tests/test_vision_classifier.py
-------------------------------
Tests for vision_classifier.py

Run from project root:
    python tests/test_vision_classifier.py

Tests:
    1.  classify_image() returns required keys
    2.  document_type is one of the valid 6 types
    3.  type_confidence is float 0.0-1.0
    4.  sensitivity_level is valid
    5.  Low OCR word count + structured image flags OCR failure risk
    6.  Higher OCR word count does not flag OCR failure risk
    7.  vision_findings carry source="vision"
    8.  Missing image path falls back gracefully
    9.  OpenCV heuristics return valid output
    10. model_used field is populated
    11. ID-card-like image classified as id_card
    12. Cheque-like image classified as cheque
    13. Passport-like image classified as passport or document
    14. Screenshot-like image classified as screenshot or document
    15. Blank image classified as unknown
    16. Different image types produce different document_type outputs
    17. Vision result merges correctly with confidence_engine output
    18. Vision finding has stable downstream scoring fields after merge
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from modules.vision_classifier import (
    classify_image,
    _compute_image_statistics,
    _classify_with_opencv,
    VALID_DOCUMENT_TYPES,
)
from main import _merge_vision_result


PASS = "  PASS"
FAIL = "  FAIL"

VALID_SENSITIVITY = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
REQUIRED_KEYS = [
    "document_type", "type_confidence", "sensitivity_level",
    "ocr_failure_risk", "vision_findings", "model_used", "fallback"
]


def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} - {label}")
    return condition


def _write_image(name: str, image: np.ndarray) -> str:
    path = os.path.join(tempfile.gettempdir(), name)
    cv2.imwrite(path, image)
    return path


def make_id_card() -> str:
    img = np.full((380, 600, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (20, 25), (580, 355), (30, 30, 30), 3)
    cv2.rectangle(img, (45, 70), (180, 210), (80, 80, 80), 2)
    for y in [80, 120, 160, 220, 260, 300]:
        cv2.rectangle(img, (220, y), (535, y + 14), (20, 20, 20), -1)
    return _write_image("vision_id_card.jpg", img)


def make_cheque() -> str:
    img = np.full((300, 900, 3), 250, dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (875, 275), (20, 20, 20), 3)
    cv2.rectangle(img, (610, 55), (830, 95), (30, 30, 30), 2)
    cv2.line(img, (95, 150), (780, 150), (20, 20, 20), 3)
    cv2.line(img, (95, 200), (825, 200), (20, 20, 20), 2)
    for x in range(80, 780, 90):
        cv2.rectangle(img, (x, 235), (x + 42, 250), (20, 20, 20), -1)
    return _write_image("vision_cheque.jpg", img)


def make_passport() -> str:
    img = np.full((620, 440, 3), 248, dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (415, 595), (25, 25, 25), 3)
    cv2.rectangle(img, (55, 65), (385, 120), (35, 35, 35), 2)
    cv2.rectangle(img, (60, 155), (190, 315), (45, 45, 45), 2)
    for y in [165, 205, 245, 340, 380, 420, 470, 515]:
        cv2.rectangle(img, (220 if y < 320 else 60, y), (380, y + 13), (25, 25, 25), -1)
    return _write_image("vision_passport.jpg", img)


def make_screenshot() -> str:
    img = np.full((500, 1000, 3), 238, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (999, 55), (35, 35, 35), -1)
    cv2.rectangle(img, (0, 55), (180, 499), (70, 70, 70), -1)
    for y in range(90, 450, 58):
        cv2.rectangle(img, (220, y), (930, y + 35), (25, 25, 25), 2)
        cv2.rectangle(img, (245, y + 10), (580, y + 20), (25, 25, 25), -1)
    return _write_image("vision_screenshot.jpg", img)


def make_blank() -> str:
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    return _write_image("vision_blank.jpg", img)


ID_CARD = make_id_card()
CHEQUE = make_cheque()
PASSPORT = make_passport()
SCREENSHOT = make_screenshot()
BLANK = make_blank()


def test_1_required_keys():
    print("\n[Test 1] classify_image() returns required keys...")
    r = classify_image(ID_CARD, ocr_word_count=0)
    for key in REQUIRED_KEYS:
        check(key in r, f"Key '{key}' present")


def test_2_valid_document_type():
    print("\n[Test 2] document_type is valid...")
    r = classify_image(ID_CARD)
    check(r["document_type"] in VALID_DOCUMENT_TYPES,
          f"document_type={r['document_type']} is valid")


def test_3_confidence_range():
    print("\n[Test 3] type_confidence is float 0.0-1.0...")
    r = classify_image(ID_CARD)
    c = r["type_confidence"]
    check(isinstance(c, float) and 0.0 <= c <= 1.0,
          f"type_confidence={c} is valid")


def test_4_sensitivity_valid():
    print("\n[Test 4] sensitivity_level is valid...")
    r = classify_image(CHEQUE)
    check(r["sensitivity_level"] in VALID_SENSITIVITY,
          f"sensitivity_level={r['sensitivity_level']} is valid")


def test_5_low_text_ocr_failure_risk():
    print("\n[Test 5] Structured image + low OCR count flags failure risk...")
    r = classify_image(ID_CARD, ocr_word_count=3)
    check(r["ocr_failure_risk"] is True, "ocr_failure_risk=True")


def test_6_high_word_count_no_failure_risk():
    print("\n[Test 6] Higher OCR count clears OCR failure risk...")
    r = classify_image(ID_CARD, ocr_word_count=25)
    check(r["ocr_failure_risk"] is False, "ocr_failure_risk=False")


def test_7_findings_source_vision():
    print("\n[Test 7] vision_findings carry source='vision'...")
    r = classify_image(ID_CARD)
    for f in r["vision_findings"]:
        check(f["source"] == "vision", f"{f['type']} source='vision'")


def test_8_missing_image_graceful_fallback():
    print("\n[Test 8] Missing image path falls back gracefully...")
    r = classify_image(os.path.join(tempfile.gettempdir(), "does_not_exist.jpg"))
    check(r["document_type"] == "unknown", "missing image returns unknown")
    check(r["fallback"] is True, "fallback=True")


def test_9_opencv_heuristics_valid():
    print("\n[Test 9] OpenCV heuristics return valid output...")
    img = cv2.imread(ID_CARD)
    stats = _compute_image_statistics(img)
    dtype, conf = _classify_with_opencv(stats)
    check(dtype in VALID_DOCUMENT_TYPES, f"heuristic type={dtype} valid")
    check(0.0 <= conf <= 1.0, f"heuristic confidence={conf} valid")


def test_10_model_used_populated():
    print("\n[Test 10] model_used field is populated...")
    r = classify_image(ID_CARD)
    check(isinstance(r["model_used"], str) and len(r["model_used"]) > 0,
          f"model_used='{r['model_used']}'")


def test_11_id_card_classification():
    print("\n[Test 11] ID-card-like image classified as id_card...")
    r = classify_image(ID_CARD)
    check(r["document_type"] == "id_card",
          f"document_type={r['document_type']}")


def test_12_cheque_classification():
    print("\n[Test 12] Cheque-like image classified as cheque...")
    r = classify_image(CHEQUE)
    check(r["document_type"] == "cheque",
          f"document_type={r['document_type']}")


def test_13_passport_classification():
    print("\n[Test 13] Passport-like image classified as passport/document...")
    r = classify_image(PASSPORT)
    check(r["document_type"] in ("passport", "document"),
          f"document_type={r['document_type']}")


def test_14_screenshot_classification():
    print("\n[Test 14] Screenshot-like image classified as screenshot/document...")
    r = classify_image(SCREENSHOT)
    check(r["document_type"] in ("screenshot", "document"),
          f"document_type={r['document_type']}")


def test_15_blank_unknown():
    print("\n[Test 15] Blank image classified as unknown...")
    r = classify_image(BLANK)
    check(r["document_type"] == "unknown",
          f"document_type={r['document_type']}")


def test_16_different_types():
    print("\n[Test 16] Different image types produce different outputs...")
    types = {
        classify_image(ID_CARD)["document_type"],
        classify_image(CHEQUE)["document_type"],
        classify_image(BLANK)["document_type"],
    }
    check(len(types) >= 3, f"distinct document types: {types}")


def test_17_merge_with_confidence_output():
    print("\n[Test 17] Vision result merges with confidence_engine output...")
    scored = {
        "is_sensitive": False,
        "total": 0,
        "matches": [],
        "overall_risk": "NONE",
        "ocr_quality": "poor",
        "score_summary": {},
        "new_findings": [],
        "context_flags": [],
    }
    vision = classify_image(CHEQUE, ocr_word_count=0)
    merged = _merge_vision_result(scored, vision)
    check(merged["total"] == len(merged["matches"]), "total matches merged length")
    check(any(m["source"] == "vision" for m in merged["matches"]),
          "merged result contains vision finding")


def test_18_merged_finding_scored():
    print("\n[Test 18] Merged vision finding has downstream score fields...")
    scored = {
        "is_sensitive": False,
        "total": 0,
        "matches": [],
        "overall_risk": "NONE",
        "ocr_quality": "poor",
        "score_summary": {},
        "new_findings": [],
        "context_flags": [],
    }
    merged = _merge_vision_result(scored, classify_image(CHEQUE, ocr_word_count=0))
    vf = merged["matches"][0]
    for key in ["unified_score", "risk_level", "score_detail"]:
        check(key in vf, f"Vision finding has '{key}'")
    check(merged["overall_risk"] == "CRITICAL", "cheque vision finding drives CRITICAL risk")


if __name__ == "__main__":
    print("=" * 60)
    print("  Vision Classifier - Test Suite (18 tests)")
    print("=" * 60)

    test_1_required_keys()
    test_2_valid_document_type()
    test_3_confidence_range()
    test_4_sensitivity_valid()
    test_5_low_text_ocr_failure_risk()
    test_6_high_word_count_no_failure_risk()
    test_7_findings_source_vision()
    test_8_missing_image_graceful_fallback()
    test_9_opencv_heuristics_valid()
    test_10_model_used_populated()
    test_11_id_card_classification()
    test_12_cheque_classification()
    test_13_passport_classification()
    test_14_screenshot_classification()
    test_15_blank_unknown()
    test_16_different_types()
    test_17_merge_with_confidence_output()
    test_18_merged_finding_scored()

    print("\n-- All 18 tests done --")
