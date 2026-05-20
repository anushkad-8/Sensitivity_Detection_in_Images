"""
tests/test_ocr_failure_scenarios.py
-----------------------------------
Review-meeting OCR failure scenarios for the vision compensation layer.

These tests use synthetic but realistic image degradations so they run without
shipping private documents in the repository.

Run from project root:
    python tests/test_ocr_failure_scenarios.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.vision_classifier import classify_image


PASS = "  PASS"
FAIL = "  FAIL"


def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} - {label}")
    return condition


def _save(name: str, image: np.ndarray) -> str:
    path = os.path.join(tempfile.gettempdir(), name)
    cv2.imwrite(path, image)
    return path


def _base_document(width=720, height=460) -> np.ndarray:
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (width - 25, height - 25), (35, 35, 35), 2)
    cv2.rectangle(img, (55, 70), (210, 235), (50, 50, 50), 2)
    for y in [80, 120, 160, 230, 275, 320, 365]:
        cv2.rectangle(img, (250, y), (width - 70, y + 13), (25, 25, 25), -1)
    return img


def make_low_contrast_doc() -> str:
    img = _base_document()
    low = cv2.addWeighted(img, 0.22, np.full_like(img, 235), 0.78, 0)
    return _save("ocr_fail_low_contrast.jpg", low)


def make_blurry_photo_doc() -> str:
    img = _base_document()
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    noise = np.random.default_rng(42).normal(0, 8, blurred.shape).astype(np.int16)
    noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return _save("ocr_fail_blurry_photo.jpg", noisy)


def make_styled_font_doc() -> str:
    img = np.full((460, 720, 3), 248, dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (695, 435), (25, 25, 25), 2)
    for i, y in enumerate([95, 150, 205, 260, 315]):
        cv2.putText(
            img, "CONFIDENTIAL ID RECORD",
            (65 + i * 8, y),
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            1.1,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
    return _save("ocr_fail_styled_font.jpg", img)


def make_obfuscated_doc() -> str:
    img = _base_document()
    for x in range(80, 640, 38):
        cv2.line(img, (x, 55), (x + 18, 400), (250, 250, 250), 3)
    for y in range(90, 380, 55):
        cv2.putText(img, "A B C D E 1 2 3 4 F", (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    return _save("ocr_fail_obfuscated.jpg", img)


def make_angled_photo_doc() -> str:
    img = _base_document()
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32([[60, 25], [w - 70, 0], [15, h - 35], [w - 20, h - 5]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderValue=(255, 255, 255))
    return _save("ocr_fail_angled_photo.jpg", warped)


SCENARIOS = {
    "low_contrast": make_low_contrast_doc,
    "blurry_photo": make_blurry_photo_doc,
    "styled_font": make_styled_font_doc,
    "obfuscated": make_obfuscated_doc,
    "angled_photo": make_angled_photo_doc,
}


def test_1_failure_scenarios_flag_ocr_risk():
    print("\n[Test 1] OCR failure scenarios are flagged by vision layer...")
    for name, factory in SCENARIOS.items():
        result = classify_image(factory(), ocr_word_count=0)
        check(result["ocr_failure_risk"] is True,
              f"{name}: ocr_failure_risk=True")


def test_2_failure_scenarios_produce_sensitive_type():
    print("\n[Test 2] OCR failure scenarios still produce useful document types...")
    valid_sensitive_types = {"id_card", "passport", "cheque", "document", "screenshot"}
    for name, factory in SCENARIOS.items():
        result = classify_image(factory(), ocr_word_count=0)
        check(result["document_type"] in valid_sensitive_types,
              f"{name}: document_type={result['document_type']}")
        check(len(result["vision_findings"]) >= 1,
              f"{name}: vision finding emitted")


if __name__ == "__main__":
    print("=" * 60)
    print("  OCR Failure Scenario Tests")
    print("=" * 60)

    test_1_failure_scenarios_flag_ocr_risk()
    test_2_failure_scenarios_produce_sensitive_type()

    print("\n-- All OCR failure scenario tests done --")
