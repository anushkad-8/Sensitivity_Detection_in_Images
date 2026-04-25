"""
tests/test_preprocess.py
------------------------
Quick sanity-check tests for Module 1: preprocess.py

Run from project root:
    python tests/test_preprocess.py

Tests:
    1. File not found raises correct error
    2. A real image preprocesses without crashing
    3. Output is a valid binary image (only 0s and 255s)
    4. Output is 2D (grayscale, not color)
"""

import sys
import os
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.preprocess import preprocess_image


def test_file_not_found():
    print("\n[Test 1] FileNotFoundError on bad path...")
    try:
        preprocess_image("input/does_not_exist.png")
        print("  FAIL — No error raised!")
    except FileNotFoundError as e:
        print(f"  PASS — Caught FileNotFoundError: {e}")


def test_real_image(image_path: str):
    print(f"\n[Test 2] Preprocessing real image: {image_path}")
    try:
        result = preprocess_image(image_path, save_debug=True)
        print(f"  PASS — Preprocessing completed. Output shape: {result.shape}")
    except Exception as e:
        print(f"  FAIL — Exception: {e}")
        return None
    return result


def test_output_is_binary(result: np.ndarray):
    print("\n[Test 3] Checking output is binary (only 0 and 255)...")
    if result is None:
        print("  SKIP — No result to check.")
        return

    unique_values = np.unique(result)
    if set(unique_values).issubset({0, 255}):
        print(f"  PASS — Only values found: {unique_values}")
    else:
        print(f"  FAIL — Non-binary values found: {unique_values}")


def test_output_is_grayscale(result: np.ndarray):
    print("\n[Test 4] Checking output is 2D (grayscale)...")
    if result is None:
        print("  SKIP — No result to check.")
        return

    if len(result.shape) == 2:
        print(f"  PASS — Output is 2D: {result.shape}")
    else:
        print(f"  FAIL — Output has unexpected shape: {result.shape}")


if __name__ == "__main__":
    test_file_not_found()

    # If you pass an image path as argument, run full tests on it
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
        result = test_real_image(img_path)
        test_output_is_binary(result)
        test_output_is_grayscale(result)
    else:
        print("\n[Tests 2–4] Skipped — pass an image path to run full tests.")
        print("  Usage: python tests/test_preprocess.py input/your_image.png")

    print("\n── All tests done ──")