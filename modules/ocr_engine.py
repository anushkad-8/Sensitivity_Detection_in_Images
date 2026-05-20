"""
Module 2: ocr_engine.py
-----------------------
Extracts text and word-level bounding boxes from a preprocessed image
using Tesseract OCR.

Input  : Cleaned image (numpy array) from Module 1 — preprocess.py
Output : Dictionary with:
            "text"  → full extracted text string (for Module 3 regex detection)
            "words" → pandas DataFrame with word positions (for highlighting)

Windows Note:
    Tesseract binary path is configured in _configure_tesseract().
    Update TESSERACT_PATH if your installation is in a different location.
"""

import os
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image


# ─────────────────────────────────────────────
# WINDOWS TESSERACT PATH — UPDATE IF NEEDED
# ─────────────────────────────────────────────

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ─────────────────────────────────────────────
# PAGE SEGMENTATION MODES (PSM) — QUICK REFERENCE
# ─────────────────────────────────────────────
# psm 3  → Fully automatic (Tesseract decides layout) — good default fallback
# psm 6  → Assume a single uniform block of text — best for scans, screenshots
# psm 11 → Sparse text, find as much as possible — best for ID cards, forms
#
# We default to psm 6. Caller can override via the `psm` argument.

DEFAULT_PSM = 6
DEFAULT_CONFIDENCE_THRESHOLD = 60   # Words below this score are dropped


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_ocr(
    clean_image: np.ndarray,
    psm: int = DEFAULT_PSM,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD
) -> dict:
    """
    Full OCR pipeline on a preprocessed image.

    Args:
        clean_image          : Numpy array output from preprocess_image() — Module 1.
        psm                  : Tesseract Page Segmentation Mode (default: 6).
                               Use 6 for docs/screenshots, 11 for ID cards/forms.
        confidence_threshold : Drop words with Tesseract confidence below this (default: 60).
                               Range: 0–100. Higher = stricter filtering.

    Returns:
        {
            "text"  : str        — full cleaned text string, ready for regex detection
            "words" : DataFrame  — filtered word table with columns:
                                   [word, left, top, width, height, conf]
        }

    Raises:
        ValueError   : If clean_image is None or not a numpy array.
        RuntimeError : If Tesseract binary is not found at TESSERACT_PATH.
    """

    # ── Step 0: Configure Tesseract path (Windows) ───────────────────────────
    _configure_tesseract()

    # ── Step 1: Validate input ────────────────────────────────────────────────
    _validate_image(clean_image)

    # ── Step 2: Run Tesseract → get full word-level data ─────────────────────
    raw_df = _extract_word_data(clean_image, psm)

    # ── Step 3: Filter out low-confidence and empty words ────────────────────
    filtered_df = _filter_words(raw_df, confidence_threshold)

    # ── Step 4: Build clean full-text string from remaining words ─────────────
    full_text = _build_text_string(filtered_df)
    quality = _build_quality_metadata(raw_df, filtered_df)

    print(f"\n[OCR] Extraction complete.")
    print(f"[OCR] Total words found     : {quality['raw_word_count']}")
    print(f"[OCR] Words after filtering : {quality['filtered_word_count']}")
    print(f"[OCR] Dropped word ratio    : {quality['dropped_word_ratio']:.0%}")
    print(f"[OCR] Extracted text preview: {full_text[:120]}{'...' if len(full_text) > 120 else ''}")

    return {
        "text"               : full_text,
        "words"              : filtered_df,
        "raw_words"          : _normalise_word_rows(raw_df),
        "raw_word_count"     : quality["raw_word_count"],
        "filtered_word_count": quality["filtered_word_count"],
        "dropped_words"      : quality["dropped_words"],
        "dropped_word_ratio" : quality["dropped_word_ratio"],
    }


# ─────────────────────────────────────────────
# INTERNAL STEPS
# ─────────────────────────────────────────────

def _configure_tesseract() -> None:
    """
    Point pytesseract to the Tesseract binary on Windows.

    On Linux/macOS, Tesseract is on PATH automatically and this is a no-op.
    On Windows, the installer does NOT always add it to PATH, so we set it explicitly.

    Update TESSERACT_PATH at the top of this file if your install location differs.
    Common alternative:
        r"C:\\Users\\<you>\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
    """
    if os.name == "nt":  # Windows only
        if not os.path.exists(TESSERACT_PATH):
            raise RuntimeError(
                f"Tesseract not found at: {TESSERACT_PATH}\n"
                "Please:\n"
                "  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  2. Install it\n"
                "  3. Update TESSERACT_PATH in modules/ocr_engine.py"
            )
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        print(f"[OCR] Tesseract path set: {TESSERACT_PATH}")


def _validate_image(image: np.ndarray) -> None:
    """
    Ensure the input is a valid numpy array before passing to Tesseract.
    Catches the most common mistake: forgetting to run preprocess first.
    """
    if image is None:
        raise ValueError(
            "[OCR] Input image is None. "
            "Make sure you pass the output of preprocess_image() here."
        )
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"[OCR] Expected numpy array, got {type(image)}. "
            "Pass the output of preprocess_image() directly."
        )
    print(f"[OCR] Input image validated. Shape: {image.shape} | dtype: {image.dtype}")


def _extract_word_data(image: np.ndarray, psm: int) -> pd.DataFrame:
    """
    Run Tesseract image_to_data() and return results as a DataFrame.

    image_to_data() returns one row per word with:
        level, page_num, block_num, par_num, line_num, word_num,
        left, top, width, height, conf, text

    We only keep what we need:
        text, left, top, width, height, conf

    PSM config string format: '--psm 6' tells Tesseract how to interpret layout.
    """
    config = f"--psm {psm}"
    print(f"[OCR] Running Tesseract with config: '{config}'")

    # Convert numpy array to PIL Image (pytesseract works with both,
    # but PIL is more reliable for edge cases like single-channel arrays)
    pil_image = Image.fromarray(image)

    raw_output = pytesseract.image_to_data(
        pil_image,
        config=config,
        output_type=pytesseract.Output.DATAFRAME  # Returns pandas DataFrame directly
    )

    # Keep only the columns we need
    columns_to_keep = ["text", "left", "top", "width", "height", "conf"]
    df = raw_output[columns_to_keep].copy()

    print(f"[OCR] Raw Tesseract output: {len(df)} rows")
    return df


def _filter_words(df: pd.DataFrame, confidence_threshold: int) -> pd.DataFrame:
    """
    Remove noise from the raw Tesseract output.

    Two filters applied:
        1. Drop rows where 'text' is empty, NaN, or whitespace-only.
           Tesseract outputs blank rows for layout gaps — we don't need those.

        2. Drop rows where confidence < threshold.
           Low-confidence words are usually OCR garbage (e.g. "I@B3", "rn0").
           These would cause false positives in Module 3 regex detection.

    Note: Tesseract sometimes returns conf = -1 for non-word rows (headers, spaces).
    These are also dropped by the confidence filter.
    """

    # Filter 1: Remove empty/whitespace text
    df = df[df["text"].notna()]
    # Coerce to string first — Tesseract occasionally returns int/float
    # in the 'text' column when the output has only layout/header rows.
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]

    before_conf_filter = len(df)

    # Filter 2: Remove low-confidence words
    df = df[df["conf"] >= confidence_threshold]

    dropped = before_conf_filter - len(df)
    print(f"[OCR] Dropped {dropped} low-confidence words (threshold: {confidence_threshold})")

    # Reset index cleanly
    df = df.reset_index(drop=True)

    # Rename 'text' → 'word' for clarity downstream
    df = df.rename(columns={"text": "word"})

    return df


def _normalise_word_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return non-empty OCR word rows before confidence filtering."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["word", "left", "top", "width", "height", "conf"])

    rows = df[df["text"].notna()].copy()
    rows = rows[rows["text"].astype(str).str.strip() != ""]
    rows = rows.rename(columns={"text": "word"})
    return rows.reset_index(drop=True)


def _build_quality_metadata(raw_df: pd.DataFrame, filtered_df: pd.DataFrame) -> dict:
    """Build OCR quality counters used by the confidence engine."""
    raw_words = _normalise_word_rows(raw_df)
    raw_count = len(raw_words)
    filtered_count = len(filtered_df) if filtered_df is not None else 0
    dropped_words = max(0, raw_count - filtered_count)
    dropped_ratio = dropped_words / raw_count if raw_count > 0 else 0.0

    return {
        "raw_word_count"     : raw_count,
        "filtered_word_count": filtered_count,
        "dropped_words"      : dropped_words,
        "dropped_word_ratio" : round(float(dropped_ratio), 3),
    }


def _build_text_string(df: pd.DataFrame) -> str:
    """
    Join all filtered words into a single text string.

    Simple space-join is sufficient for regex matching in Module 3.
    We do NOT attempt to reconstruct line breaks or paragraphs —
    regex patterns don't need that structure.

    Returns empty string if no words passed filtering.
    """
    if df.empty:
        print("[OCR] Warning: No words remained after filtering. Text will be empty.")
        return ""

    text = " ".join(df["word"].astype(str).tolist())
    return text


# ─────────────────────────────────────────────
# HELPER — used by main.py for highlighting
# ─────────────────────────────────────────────

def find_word_boxes(words_df: pd.DataFrame, target_words: list) -> pd.DataFrame:
    """
    Given the word DataFrame and a list of matched sensitive words/tokens,
    return only the rows whose 'word' value appears in target_words.

    Used by main.py to know WHERE to draw bounding boxes on the image.

    Args:
        words_df     : DataFrame from run_ocr()["words"]
        target_words : List of word strings detected as sensitive by Module 3

    Returns:
        Subset DataFrame with bounding box info for matched words only.

    Example:
        target_words = ["9876", "5432", "1098"]   ← parts of an Aadhaar number
        → returns rows for each of those words with their (left, top, width, height)
    """
    if not target_words or words_df.empty:
        return pd.DataFrame()

    matched = words_df[words_df["word"].isin(target_words)].copy()
    print(f"[OCR] find_word_boxes: {len(matched)} word boxes matched for highlighting.")
    return matched


# ─────────────────────────────────────────────
# QUICK TEST (run this file directly to verify)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from modules.preprocess import preprocess_image

    if len(sys.argv) < 2:
        print("Usage: python modules/ocr_engine.py <path_to_image>")
        print("Example: python modules/ocr_engine.py input/sample.png")
        sys.exit(1)

    img_path = sys.argv[1]

    print("\n── Step 1: Preprocessing ──")
    clean, scale_factor = preprocess_image(img_path, save_debug=False)

    print("\n── Step 2: Running OCR ──")
    result = run_ocr(clean)

    print("\n── Full Extracted Text ──")
    print(result["text"])

    print("\n── Word Table (first 10 rows) ──")
    print(result["words"].head(10).to_string(index=False))