"""
Module 2: ocr_engine.py
-----------------------
Extracts text and word-level bounding boxes from a preprocessed image
using Tesseract OCR.

Input  : Cleaned image (numpy array) from Module 1 — preprocess.py
         Channel images dict from preprocess.py (for obfuscation detection)
Output : Dictionary with:
            "text"             → raw extracted text string
            "text_normalized"  → cleaned text (USE THIS for regex detection)
            "digit_runs"       → all digit sequences extracted (for safer Aadhaar/PAN matching)
            "words"            → pandas DataFrame with word positions (for highlighting)
            "obfuscation"      → full result dict from anti_obfuscation.detect_all()
            "obfuscation_flags"→ simplified bool flags for decision engine

CHANGES (Anti-Obfuscation Update):
    1. _filter_words() replaced by reassemble_fragments() — spatially adjacent
       low-confidence digit tokens are rejoined before being discarded.
    2. normalize_ocr_text() added — strips zero-width chars, maps homoglyphs
       (Cyrillic/Greek → Latin), rejoins fragmented digit sequences.
    3. extract_digit_runs() added — pulls all digit sequences for downstream regex.
    4. _run_channel_ocr() added — runs OCR on each colour channel image from
       preprocess.py and merges any unique tokens into the main text.
    5. anti_obfuscation.detect_all() integrated — called after normalization.
       Returns decoded_text (base64/hex decoded) for Module 3 to scan.
    6. run_ocr() return dict extended — backward compatible (text + words unchanged).

Windows Note:
    Tesseract binary path is configured in _configure_tesseract().
    Update TESSERACT_PATH if your installation is in a different location.
"""

import os
import re
import unicodedata
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

from modules.anti_obfuscation import detect_all as obfuscation_detect


# ─────────────────────────────────────────────
# WINDOWS TESSERACT PATH — UPDATE IF NEEDED
# ─────────────────────────────────────────────

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_PSM                = 6
DEFAULT_CONFIDENCE_THRESHOLD = 60

# Homoglyph map: visually identical characters from other scripts → Latin equivalent
# Attackers use these to fool regex matching: "Рrаkаsh" instead of "Prakash"
HOMOGLYPH_MAP = {
    # Cyrillic → Latin
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',
    'у': 'y', 'х': 'x', 'В': 'B', 'Е': 'E', 'К': 'K',
    'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C',
    'Т': 'T', 'У': 'Y', 'Х': 'X',
    # Greek → Latin
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H',
    'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O',
    'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
    # Fullwidth digits → ASCII digits
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    # Common visual substitutions Tesseract sometimes returns
    '|': 'I', 'ⅼ': 'l',
}

# Zero-width and invisible Unicode characters attackers insert to break pattern matching
ZERO_WIDTH_CHARS = [
    '\u200b',  # Zero width space
    '\u200c',  # Zero width non-joiner
    '\u200d',  # Zero width joiner
    '\u200e',  # Left-to-right mark
    '\u200f',  # Right-to-left mark
    '\ufeff',  # Byte order mark / zero width no-break space
    '\u00ad',  # Soft hyphen (invisible but breaks word boundaries)
]


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_ocr(
    clean_image           : np.ndarray,
    channel_images        : dict = None,
    psm                   : int  = DEFAULT_PSM,
    confidence_threshold  : int  = DEFAULT_CONFIDENCE_THRESHOLD
) -> dict:
    """
    Full OCR pipeline on a preprocessed image.

    Args:
        clean_image          : Numpy array output from preprocess_image() — Module 1.
        channel_images       : Dict {"R": arr, "G": arr, "B": arr} from preprocess_image().
                               Pass None if preprocess was run without channel splitting.
        psm                  : Tesseract Page Segmentation Mode (default: 6).
        confidence_threshold : Drop words below this confidence (default: 60).

    Returns:
        {
            "text"             : str        — raw joined text (unchanged from before)
            "text_normalized"  : str        — USE THIS for regex detection in Module 3
            "digit_runs"       : list[str]  — all digit sequences found
            "words"            : DataFrame  — word table [word, left, top, width, height, conf]
            "obfuscation"      : dict       — full anti_obfuscation result
            "obfuscation_flags": dict       — simplified booleans for decision engine
        }

    Raises:
        ValueError   : If clean_image is None or not a numpy array.
        RuntimeError : If Tesseract binary is not found at TESSERACT_PATH.
    """

    _configure_tesseract()
    _validate_image(clean_image)

    # ── Step 1: Run Tesseract on main preprocessed image ─────────────────────
    raw_df   = _extract_word_data(clean_image, psm)
    filtered_df = reassemble_fragments(raw_df, confidence_threshold)

    # ── Step 2: Run OCR on colour channels and merge unique tokens ────────────
    if channel_images:
        filtered_df = _run_channel_ocr(channel_images, filtered_df, psm, confidence_threshold)

    # ── Step 3: Build raw text string ─────────────────────────────────────────
    raw_text = _build_text_string(filtered_df)

    # ── Step 4: Normalize — homoglyphs, zero-width chars, digit rejoining ─────
    normalized_text = normalize_ocr_text(raw_text)
    digit_runs      = extract_digit_runs(normalized_text)

    # ── Step 5: Anti-obfuscation scan ─────────────────────────────────────────
    # detect_all() returns decoded_text with base64/hex tokens replaced by
    # their decoded plaintext. This is what Module 3 should scan.
    obfuscation_result = obfuscation_detect(normalized_text)

    obfuscation_flags = {
        "homoglyphs_cleaned"  : normalized_text != raw_text,
        "encoded_tokens_found": obfuscation_result["has_encoded"],
        "encrypted_content"   : obfuscation_result["has_encrypted"],
        "hash_patterns_found" : obfuscation_result["has_hashes"],
        "escalate"            : obfuscation_result["escalate"],
        "low_conf_ratio"      : _compute_low_conf_ratio(raw_df, filtered_df),
    }

    print(f"\n[OCR] Extraction complete.")
    print(f"[OCR] Total words found       : {len(raw_df)}")
    print(f"[OCR] Words after filtering   : {len(filtered_df)}")
    print(f"[OCR] Raw text preview        : {raw_text[:100]}{'...' if len(raw_text) > 100 else ''}")
    print(f"[OCR] Normalized text preview : {normalized_text[:100]}{'...' if len(normalized_text) > 100 else ''}")
    print(f"[OCR] Obfuscation flags       : {obfuscation_flags}")

    # The "text_for_detection" is the cleanest version:
    # - normalized (homoglyphs cleaned)
    # - encoded tokens replaced with decoded plaintext
    # Module 3 should use this.
    text_for_detection = obfuscation_result["decoded_text"]

    return {
        "text"             : raw_text,           # Original — unchanged for backward compat
        "text_normalized"  : normalized_text,    # After homoglyph/ZWC cleaning
        "text_for_detection": text_for_detection, # USE THIS in Module 3 — cleanest
        "digit_runs"       : digit_runs,
        "words"            : filtered_df,
        "obfuscation"      : obfuscation_result,
        "obfuscation_flags": obfuscation_flags,
    }


# ─────────────────────────────────────────────
# CHANNEL OCR — ANTI-OBFUSCATION
# ─────────────────────────────────────────────

def _run_channel_ocr(
    channel_images       : dict,
    main_df              : pd.DataFrame,
    psm                  : int,
    confidence_threshold : int
) -> pd.DataFrame:
    """
    Run Tesseract on each colour channel image (R, G, B) separately.
    Merge any tokens found in channels but NOT in the main OCR result
    into the main DataFrame.

    WHY: Attackers can write sensitive text using a single colour channel.
    Grayscale OCR completely misses it. Channel OCR catches it.

    Merged channel tokens are added WITHOUT bounding boxes (left/top/width/height = -1)
    because their coordinates are in the channel's coordinate space, not the original.
    They will appear in text but not be highlighted — acceptable trade-off.
    """
    if not channel_images:
        return main_df

    existing_words = set(main_df["word"].str.lower().tolist()) if not main_df.empty else set()
    new_rows = []

    for ch_name, ch_image in channel_images.items():
        print(f"\n[OCR] Running channel OCR: {ch_name}")
        try:
            ch_raw = _extract_word_data(ch_image, psm)
            ch_filtered = reassemble_fragments(ch_raw, confidence_threshold)

            if ch_filtered.empty:
                continue

            for _, row in ch_filtered.iterrows():
                word = str(row["word"]).strip().lower()
                if word and word not in existing_words and len(word) > 1:
                    print(f"[OCR] Channel {ch_name} found new token: '{row['word']}'")
                    existing_words.add(word)
                    new_rows.append({
                        "word"  : row["word"],
                        "left"  : -1,   # No reliable coordinate in original image space
                        "top"   : -1,
                        "width" : -1,
                        "height": -1,
                        "conf"  : row["conf"],
                    })
        except Exception as e:
            print(f"[OCR] Channel {ch_name} OCR failed: {e}")
            continue

    if new_rows:
        extra_df = pd.DataFrame(new_rows)
        merged   = pd.concat([main_df, extra_df], ignore_index=True)
        print(f"[OCR] Channel OCR added {len(new_rows)} new token(s).")
        return merged

    return main_df


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize_ocr_text(text: str) -> str:
    """
    Clean OCR text to remove deliberate obfuscation techniques.

    Handles:
        1. Zero-width / invisible Unicode characters
        2. Unicode normalization (NFKC catches many homoglyphs automatically)
        3. Homoglyph substitution (Cyrillic/Greek/fullwidth → Latin/ASCII)
        4. Fragmented digit sequences ("9 1 8 3" → "9183")

    Args:
        text : Raw joined text from _build_text_string()

    Returns:
        Cleaned text string ready for anti_obfuscation and regex detection.
    """

    # Step 1: Strip zero-width and invisible characters
    for zwc in ZERO_WIDTH_CHARS:
        text = text.replace(zwc, '')

    # Step 2: Unicode normalization — NFKC decomposes compatibility characters
    # e.g. fullwidth letters, ligatures, some homoglyphs
    text = unicodedata.normalize('NFKC', text)

    # Step 3: Homoglyph map — character-by-character substitution
    text = ''.join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)

    # Step 4: Rejoin fragmented digit sequences
    # "9 1 8 3 0 0 7 4" → "91830074" (OCR spacing artifacts on ID numbers)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    # Run twice to catch multi-step fragments: "9 1 8 3" needs 3 passes
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    return text


def extract_digit_runs(text: str) -> list:
    """
    Extract all contiguous digit sequences from normalized text.
    Used by Module 3 as a safer alternative to running Aadhaar/PAN
    regex on noisy raw text.

    Returns:
        List of digit-only strings, e.g. ["9183", "0074", "6619", "1990"]
    """
    return re.findall(r'\d+', text)


# ─────────────────────────────────────────────
# FRAGMENT REASSEMBLY — replaces old _filter_words()
# ─────────────────────────────────────────────

def reassemble_fragments(df: pd.DataFrame, confidence_threshold: int = 60) -> pd.DataFrame:
    """
    Improved word filtering that reassembles spatially adjacent digit fragments
    before discarding low-confidence tokens.

    WHY THIS MATTERS:
        Aadhaar numbers like "9183 0074 6619" are sometimes OCR'd as:
        "9", "18", "3", "00", "74", "66", "19" — each individually low-confidence.
        The old filter would drop ALL of them. This function detects that they
        form a digit group when concatenated and keeps the merged token.

    Process:
        1. Sort words by position (top-to-bottom, left-to-right)
        2. For high-confidence words: keep as-is
        3. For low-confidence words: check if spatially adjacent neighbours
           form a digit sequence when merged. If yes → keep merged token.
           If no → discard as noise.

    Args:
        df                   : Raw Tesseract DataFrame from _extract_word_data()
        confidence_threshold : Words below this are candidates for reassembly or discard

    Returns:
        Filtered DataFrame with merged digit fragments where appropriate.
    """
    # Basic cleaning first
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    # Sort by reading order: top-to-bottom, left-to-right
    df = df.sort_values(["top", "left"]).reset_index(drop=True)

    kept_rows   = []
    skip_indices = set()
    total_before = len(df)

    i = 0
    while i < len(df):
        if i in skip_indices:
            i += 1
            continue

        row = df.iloc[i]
        word_text = str(row["text"]).strip()
        conf      = float(row["conf"])

        if conf >= confidence_threshold:
            # High confidence — keep directly
            kept_rows.append({
                "word"  : word_text,
                "left"  : int(row["left"]),
                "top"   : int(row["top"]),
                "width" : int(row["width"]),
                "height": int(row["height"]),
                "conf"  : conf,
            })
            i += 1
            continue

        # Low confidence — try to reassemble with neighbours
        group_words  = [word_text]
        group_confs  = [conf]
        group_indices = [i]

        # Look ahead at next 3 tokens for spatial adjacency
        j = i + 1
        while j < min(i + 4, len(df)) and j not in skip_indices:
            next_row  = df.iloc[j]
            next_word = str(next_row["text"]).strip()
            next_conf = float(next_row["conf"])

            # Horizontal gap between current group end and next word start
            current_right = int(row["left"]) + int(row["width"])
            horizontal_gap = int(next_row["left"]) - current_right
            vertical_diff  = abs(int(next_row["top"]) - int(row["top"]))

            # Tokens are spatially adjacent if gap < 40px and same line (vertical < 15px)
            if horizontal_gap < 40 and vertical_diff < 15:
                group_words.append(next_word)
                group_confs.append(next_conf)
                group_indices.append(j)
                j += 1
            else:
                break

        # Check if group forms a valid digit sequence
        merged = ''.join(group_words)
        if re.match(r'^\d+$', merged) and len(merged) >= 4:
            # Looks like a fragmented number — keep it merged
            avg_conf = sum(group_confs) / len(group_confs)
            print(f"[OCR] Fragment reassembled: {group_words} → '{merged}' (avg conf: {avg_conf:.0f})")
            kept_rows.append({
                "word"  : merged,
                "left"  : int(df.iloc[group_indices[0]]["left"]),
                "top"   : int(df.iloc[group_indices[0]]["top"]),
                "width" : int(df.iloc[group_indices[-1]]["left"])
                          + int(df.iloc[group_indices[-1]]["width"])
                          - int(df.iloc[group_indices[0]]["left"]),
                "height": int(row["height"]),
                "conf"  : avg_conf,
            })
            for idx in group_indices:
                skip_indices.add(idx)
        # else: low-confidence, non-digit → discard as noise

        i = j if len(group_indices) > 1 else i + 1

    result_df = pd.DataFrame(kept_rows)
    dropped   = total_before - len(result_df)
    print(f"[OCR] reassemble_fragments: kept {len(result_df)}, dropped {dropped} of {total_before}")
    return result_df


# ─────────────────────────────────────────────
# INTERNAL STEPS
# ─────────────────────────────────────────────

def _configure_tesseract() -> None:
    if os.name == "nt":
        if not os.path.exists(TESSERACT_PATH):
            raise RuntimeError(
                f"Tesseract not found at: {TESSERACT_PATH}\n"
                "Please:\n"
                "  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  2. Install it\n"
                "  3. Update TESSERACT_PATH in modules/ocr_engine.py"
            )
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def _validate_image(image: np.ndarray) -> None:
    if image is None:
        raise ValueError("[OCR] Input image is None. Pass the output of preprocess_image().")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"[OCR] Expected numpy array, got {type(image)}.")
    print(f"[OCR] Input image validated. Shape: {image.shape} | dtype: {image.dtype}")


def _extract_word_data(image: np.ndarray, psm: int) -> pd.DataFrame:
    config     = f"--psm {psm}"
    pil_image  = Image.fromarray(image)
    raw_output = pytesseract.image_to_data(
        pil_image,
        config=config,
        output_type=pytesseract.Output.DATAFRAME
    )
    columns_to_keep = ["text", "left", "top", "width", "height", "conf"]
    return raw_output[columns_to_keep].copy()


def _build_text_string(df: pd.DataFrame) -> str:
    if df.empty:
        print("[OCR] Warning: No words after filtering.")
        return ""
    return " ".join(df["word"].astype(str).tolist())


def _compute_low_conf_ratio(raw_df: pd.DataFrame, filtered_df: pd.DataFrame) -> float:
    """Ratio of words dropped vs total. High ratio signals possible obfuscation."""
    if len(raw_df) == 0:
        return 0.0
    dropped = len(raw_df) - len(filtered_df)
    return round(dropped / len(raw_df), 2)


# ─────────────────────────────────────────────
# HELPER — used by main.py for highlighting
# ─────────────────────────────────────────────

def find_word_boxes(words_df: pd.DataFrame, target_words: list) -> pd.DataFrame:
    """
    Return bounding box rows for words matching target_words.
    Excludes channel-OCR tokens (left == -1) since they have no valid coordinates.
    """
    if not target_words or words_df.empty:
        return pd.DataFrame()

    matched = words_df[
        words_df["word"].isin(target_words) &
        (words_df["left"] >= 0)   # Exclude channel-OCR tokens (no valid bbox)
    ].copy()

    print(f"[OCR] find_word_boxes: {len(matched)} word boxes matched for highlighting.")
    return matched


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from modules.preprocess import preprocess_image

    if len(sys.argv) < 2:
        print("Usage: python modules/ocr_engine.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]

    print("\n── Step 1: Preprocessing ──")
    clean, scale_factor, channel_images = preprocess_image(img_path, save_debug=False)

    print("\n── Step 2: Running OCR ──")
    result = run_ocr(clean, channel_images=channel_images)

    print("\n── Full Extracted Text (raw) ──")
    print(result["text"])

    print("\n── Text for Detection (normalized + decoded) ──")
    print(result["text_for_detection"])

    print("\n── Digit Runs ──")
    print(result["digit_runs"])

    print("\n── Obfuscation Flags ──")
    print(result["obfuscation_flags"])

    print("\n── Word Table (first 10 rows) ──")
    print(result["words"].head(10).to_string(index=False))