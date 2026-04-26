"""
Module 2: ocr_engine.py
-----------------------
Extracts text and word-level bounding boxes from a preprocessed image
using Tesseract OCR.

Input  : Cleaned image (numpy array) from Module 1 — preprocess.py
         Channel images dict from preprocess.py (for obfuscation detection)
         Gray raw image from preprocess.py (for second-pass OCR)
Output : Dictionary with:
            "text"              → raw extracted text string
            "text_normalized"   → cleaned text (USE THIS for regex detection)
            "digit_runs"        → all digit sequences extracted
            "words"             → pandas DataFrame with word positions
            "obfuscation"       → full result dict from anti_obfuscation.detect_all()
            "obfuscation_flags" → simplified bool flags for decision engine

CHANGES v3 — Two-Pass OCR (Root Cause Fix):
    ROOT CAUSE IDENTIFIED:
        OTSU binarization destroys thin characters: dashes (-----), plus (+),
        forward slash (/), equals (=). These are exactly the characters that make
        up PEM headers and base64 key body lines. After OTSU, Tesseract finds
        0 tokens for those lines — they never reach any filter or scanner.

    FIX:
        run_ocr() now accepts gray_raw (grayscale before OTSU) from preprocess.py.
        After first-pass OCR, _looks_like_partial_pem() checks if output shows
        signs of a partial PEM block (base64 body found, but no header/footer).
        If yes, a SECOND PASS runs on gray_raw with PSM 6 and confidence threshold
        lowered to 30. The two passes are merged before normalization and scanning.

    OTHER CHANGES:
        - _extract_pem_header_rows() pre-pass retained (catches fragmented headers).
        - Base64/hex token preservation retained (v2).
        - Dash token preservation retained.
        - DEFAULT_PSM remains 4.
        - run_ocr() signature extended with gray_raw=None (backward compatible).
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

DEFAULT_PSM                  = 4    # Single column, variable sizes
DEFAULT_CONFIDENCE_THRESHOLD = 60
SECOND_PASS_PSM              = 6    # Uniform text block — better for PEM
SECOND_PASS_CONF_THRESHOLD   = 30   # Lower — PEM dashes/slashes always low-conf

HOMOGLYPH_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',
    'у': 'y', 'х': 'x', 'В': 'B', 'Е': 'E', 'К': 'K',
    'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C',
    'Т': 'T', 'У': 'Y', 'Х': 'X',
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H',
    'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O',
    'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    '|': 'I', 'ⅼ': 'l',
}

ZERO_WIDTH_CHARS = [
    '\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '\ufeff', '\u00ad',
]

_PEM_LINE_RE = re.compile(
    r'-{3,}\s*(?:BEGIN|END)\s+[\w\s]+\s*-{3,}',
    re.IGNORECASE
)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_ocr(
    clean_image           : np.ndarray,
    channel_images        : dict  = None,
    psm                   : int   = DEFAULT_PSM,
    confidence_threshold  : int   = DEFAULT_CONFIDENCE_THRESHOLD,
    gray_raw              : np.ndarray = None,   # FIX v3: pre-OTSU grayscale
) -> dict:
    """
    Full OCR pipeline on a preprocessed image.

    Args:
        clean_image          : OTSU-binarized numpy array from preprocess_image().
        channel_images       : {"R": arr, "G": arr, "B": arr} from preprocess_image().
        psm                  : Tesseract PSM for first pass (default: 4).
        confidence_threshold : Confidence cutoff for first pass (default: 60).
        gray_raw             : Pre-OTSU grayscale from preprocess_image() [NEW v3].
                               Pass this to enable second-pass PEM recovery.

    Returns:
        {
            "text"              : str
            "text_normalized"   : str
            "text_for_detection": str
            "digit_runs"        : list[str]
            "words"             : DataFrame
            "obfuscation"       : dict
            "obfuscation_flags" : dict
        }
    """
    _configure_tesseract()
    _validate_image(clean_image)

    # ── Step 1: First-pass OCR on OTSU-binarized image ────────────────────────
    raw_df      = _extract_word_data(clean_image, psm)
    filtered_df = reassemble_fragments(raw_df, confidence_threshold)

    # ── Step 2: Channel OCR ───────────────────────────────────────────────────
    if channel_images:
        filtered_df = _run_channel_ocr(channel_images, filtered_df, psm, confidence_threshold)

    # ── Step 3: Build raw text from first pass ────────────────────────────────
    raw_text = _build_text_string(filtered_df)

    # ── FIX v3: Second-pass OCR on raw grayscale if PEM detected partially ────
    #
    # WHY: OTSU binarization kills thin PEM characters (dashes, +, /, =).
    # After first pass, if we see base64-like content but NO PEM header/footer,
    # we re-run OCR on the raw grayscale (pre-OTSU) with:
    #   - PSM 6 (uniform block — better for structured PEM text)
    #   - Confidence threshold 30 (PEM dashes always get low Tesseract scores)
    # Results are merged with first pass before normalization.
    #
    if gray_raw is not None and _looks_like_partial_pem(raw_text, filtered_df):
        print("\n[OCR] ⚠️  Partial PEM detected — running second pass on raw grayscale.")
        print(f"[OCR] Second pass: PSM={SECOND_PASS_PSM}, conf_threshold={SECOND_PASS_CONF_THRESHOLD}")

        raw_df2      = _extract_word_data(gray_raw, SECOND_PASS_PSM)
        filtered_df2 = reassemble_fragments(raw_df2, SECOND_PASS_CONF_THRESHOLD)
        raw_text2    = _build_text_string(filtered_df2)

        print(f"[OCR] Second pass extracted {len(filtered_df2)} tokens.")
        print(f"[OCR] Second pass text: {raw_text2[:150]}")

        # Merge: append second-pass words and rebuild combined text
        filtered_df = pd.concat([filtered_df, filtered_df2], ignore_index=True)
        raw_text    = _build_text_string(filtered_df)

    # ── Step 4: Normalize ─────────────────────────────────────────────────────
    normalized_text = normalize_ocr_text(raw_text)
    digit_runs      = extract_digit_runs(normalized_text)

    # ── Step 5: Anti-obfuscation scan ─────────────────────────────────────────
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
    print(f"[OCR] Raw text preview        : {raw_text[:120]}{'...' if len(raw_text) > 120 else ''}")
    print(f"[OCR] Normalized text preview : {normalized_text[:120]}{'...' if len(normalized_text) > 120 else ''}")
    print(f"[OCR] Obfuscation flags       : {obfuscation_flags}")

    text_for_detection = obfuscation_result["decoded_text"]

    return {
        "text"              : raw_text,
        "text_normalized"   : normalized_text,
        "text_for_detection": text_for_detection,
        "digit_runs"        : digit_runs,
        "words"             : filtered_df,
        "obfuscation"       : obfuscation_result,
        "obfuscation_flags" : obfuscation_flags,
    }


# ─────────────────────────────────────────────
# FIX v3: PARTIAL PEM DETECTOR
# ─────────────────────────────────────────────

def _looks_like_partial_pem(text: str, words_df: pd.DataFrame) -> bool:
    """
    Heuristic: did first-pass OCR catch PEM body content but miss the headers?

    Returns True (trigger second pass) when ALL of:
        1. Text contains base64-like tokens (likely key body lines were read)
        2. Text does NOT contain PEM header markers (BEGIN/END or -----)
        3. Word count is low — fewer tokens than a full PEM block would produce

    This avoids running the second pass on normal documents.

    Args:
        text     : Raw text from first OCR pass.
        words_df : Filtered word DataFrame from first pass.

    Returns:
        bool — True if second pass should be triggered.
    """
    has_base64_body = bool(re.search(r'[A-Za-z0-9+/]{15,}', text))
    has_pem_header  = bool(re.search(r'(-{3,}|BEGIN|END\s+\w)', text, re.IGNORECASE))
    low_word_count  = len(words_df) < 10

    trigger = has_base64_body and not has_pem_header and low_word_count

    if trigger:
        print(f"[OCR] _looks_like_partial_pem → True "
              f"(base64_body={has_base64_body}, pem_header={has_pem_header}, "
              f"word_count={len(words_df)})")
    return trigger


# ─────────────────────────────────────────────
# PEM HEADER PRE-PASS
# ─────────────────────────────────────────────

def _extract_pem_header_rows(df: pd.DataFrame) -> tuple:
    """
    Pre-pass: group tokens by line, merge any line whose joined text matches
    a PEM header/footer pattern, and return them separately so they survive
    the confidence filter.

    Args:
        df : Raw Tesseract DataFrame.

    Returns:
        (remaining_df, pem_rows)
    """
    if df.empty:
        return df, []

    df = df.sort_values(["top", "left"]).reset_index(drop=True)

    lines        = []
    current_line = [0]

    for idx in range(1, len(df)):
        prev_top = int(df.iloc[current_line[0]]["top"])
        curr_top = int(df.iloc[idx]["top"])
        if abs(curr_top - prev_top) < 20:
            current_line.append(idx)
        else:
            lines.append(current_line)
            current_line = [idx]
    lines.append(current_line)

    pem_rows     = []
    drop_indices = set()

    for line_idxs in lines:
        tokens    = [str(df.iloc[i]["text"]).strip() for i in line_idxs]
        line_text = " ".join(tokens)

        if _PEM_LINE_RE.search(line_text):
            merged   = re.sub(r'\s+', '', line_text)
            avg_conf = sum(float(df.iloc[i]["conf"]) for i in line_idxs) / len(line_idxs)
            first    = df.iloc[line_idxs[0]]

            print(f"[OCR] PEM header pre-pass merged: '{line_text.strip()}' → '{merged}'")
            pem_rows.append({
                "word"  : merged,
                "left"  : int(first["left"]),
                "top"   : int(first["top"]),
                "width" : -1,
                "height": int(first["height"]),
                "conf"  : avg_conf,
            })
            for i in line_idxs:
                drop_indices.add(i)

    remaining_df = df.drop(index=list(drop_indices)).reset_index(drop=True)
    return remaining_df, pem_rows


# ─────────────────────────────────────────────
# FRAGMENT REASSEMBLY
# ─────────────────────────────────────────────

def reassemble_fragments(df: pd.DataFrame, confidence_threshold: int = 60) -> pd.DataFrame:
    """
    Filter and reassemble OCR tokens.

    Processing order:
        1. PEM header pre-pass  — merge header/footer fragments before filtering
        2. High confidence      — kept as-is
        3. Digit fragments      — spatially merged if they form a number
        4. Base64-like tokens   — always preserved (v2)
        5. Hex-like tokens      — always preserved (v2)
        6. Dash tokens          — always preserved (likely PEM header fragment)
        7. Everything else      — discarded as noise
    """
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)
    df = df.sort_values(["top", "left"]).reset_index(drop=True)

    total_before = len(df)

    # ── PEM header pre-pass ───────────────────────────────────────────────────
    df, pem_rows = _extract_pem_header_rows(df)
    kept_rows    = list(pem_rows)
    skip_indices = set()

    i = 0
    while i < len(df):
        if i in skip_indices:
            i += 1
            continue

        row       = df.iloc[i]
        word_text = str(row["text"]).strip()
        conf      = float(row["conf"])

        # ── High confidence ───────────────────────────────────────────────────
        if conf >= confidence_threshold:
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

        # ── Low confidence: look-ahead spatial reassembly ─────────────────────
        group_words   = [word_text]
        group_confs   = [conf]
        group_indices = [i]

        j = i + 1
        while j < min(i + 4, len(df)) and j not in skip_indices:
            next_row   = df.iloc[j]
            next_word  = str(next_row["text"]).strip()
            next_conf  = float(next_row["conf"])

            current_right  = int(row["left"]) + int(row["width"])
            horizontal_gap = int(next_row["left"]) - current_right
            vertical_diff  = abs(int(next_row["top"]) - int(row["top"]))

            if horizontal_gap < 40 and vertical_diff < 15:
                group_words.append(next_word)
                group_confs.append(next_conf)
                group_indices.append(j)
                j += 1
            else:
                break

        merged   = ''.join(group_words)
        avg_conf = sum(group_confs) / len(group_confs)

        # ── Case 1: Digit sequence ────────────────────────────────────────────
        if re.match(r'^\d+$', merged) and len(merged) >= 4:
            print(f"[OCR] Digit fragment reassembled: {group_words} → '{merged}' (avg conf: {avg_conf:.0f})")
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

        # ── Case 2: Base64-like token ─────────────────────────────────────────
        elif re.match(r'^[A-Za-z0-9+/=]{6,}$', word_text):
            print(f"[OCR] Base64-like token preserved (low conf {conf:.0f}): '{word_text[:40]}'")
            kept_rows.append({
                "word"  : word_text,
                "left"  : int(row["left"]),
                "top"   : int(row["top"]),
                "width" : int(row["width"]),
                "height": int(row["height"]),
                "conf"  : conf,
            })

        # ── Case 3: Hex-like token ────────────────────────────────────────────
        elif re.match(r'^[0-9a-fA-F]{8,}$', word_text):
            print(f"[OCR] Hex-like token preserved (low conf {conf:.0f}): '{word_text[:40]}'")
            kept_rows.append({
                "word"  : word_text,
                "left"  : int(row["left"]),
                "top"   : int(row["top"]),
                "width" : int(row["width"]),
                "height": int(row["height"]),
                "conf"  : conf,
            })

        # ── Case 4: Dash token — likely PEM header fragment ───────────────────
        elif re.match(r'^-{3,}', word_text):
            print(f"[OCR] Dash token preserved (low conf {conf:.0f}): '{word_text}'")
            kept_rows.append({
                "word"  : word_text,
                "left"  : int(row["left"]),
                "top"   : int(row["top"]),
                "width" : int(row["width"]),
                "height": int(row["height"]),
                "conf"  : conf,
            })

        # ── Case 5: Noise — discard ───────────────────────────────────────────

        i = j if len(group_indices) > 1 else i + 1

    result_df = pd.DataFrame(kept_rows) if kept_rows else pd.DataFrame(
        columns=["word", "left", "top", "width", "height", "conf"]
    )
    print(f"[OCR] reassemble_fragments: kept {len(result_df)}, dropped {total_before - len(result_df) + len(pem_rows)} of {total_before}")
    return result_df


# ─────────────────────────────────────────────
# CHANNEL OCR
# ─────────────────────────────────────────────

def _run_channel_ocr(
    channel_images       : dict,
    main_df              : pd.DataFrame,
    psm                  : int,
    confidence_threshold : int
) -> pd.DataFrame:
    """Run OCR on each colour channel and merge unique tokens."""
    if not channel_images:
        return main_df

    existing_words = set(main_df["word"].str.lower().tolist()) if not main_df.empty else set()
    new_rows = []

    for ch_name, ch_image in channel_images.items():
        print(f"\n[OCR] Running channel OCR: {ch_name}")
        try:
            ch_raw      = _extract_word_data(ch_image, psm)
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
                        "left"  : -1,
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
    for zwc in ZERO_WIDTH_CHARS:
        text = text.replace(zwc, '')
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)
    for _ in range(3):
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    return text


def extract_digit_runs(text: str) -> list:
    return re.findall(r'\d+', text)


# ─────────────────────────────────────────────
# INTERNAL STEPS
# ─────────────────────────────────────────────

def _configure_tesseract() -> None:
    if os.name == "nt":
        if not os.path.exists(TESSERACT_PATH):
            raise RuntimeError(
                f"Tesseract not found at: {TESSERACT_PATH}\n"
                "  1. Download: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  2. Install and update TESSERACT_PATH in ocr_engine.py"
            )
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def _validate_image(image: np.ndarray) -> None:
    if image is None:
        raise ValueError("[OCR] Input image is None.")
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
    if len(raw_df) == 0:
        return 0.0
    dropped = len(raw_df) - len(filtered_df)
    return round(dropped / len(raw_df), 2)


# ─────────────────────────────────────────────
# HELPER — used by main.py for highlighting
# ─────────────────────────────────────────────

def find_word_boxes(words_df: pd.DataFrame, target_words: list) -> pd.DataFrame:
    if not target_words or words_df.empty:
        return pd.DataFrame()
    matched = words_df[
        words_df["word"].isin(target_words) &
        (words_df["left"] >= 0)
    ].copy()
    print(f"[OCR] find_word_boxes: {len(matched)} word boxes matched.")
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
    # FIX v3: Unpack 4 return values — gray_raw is new
    clean, scale_factor, channel_images, gray_raw = preprocess_image(img_path, save_debug=False)

    print("\n── Step 2: Running OCR ──")
    # FIX v3: Pass gray_raw to enable second-pass PEM recovery
    result = run_ocr(clean, channel_images=channel_images, gray_raw=gray_raw)

    print("\n── Full Extracted Text (raw) ──")
    print(result["text"])

    print("\n── Text for Detection (normalized + decoded) ──")
    print(result["text_for_detection"])

    print("\n── Digit Runs ──")
    print(result["digit_runs"])

    print("\n── Obfuscation Flags ──")
    for flag, value in result["obfuscation_flags"].items():
        icon = "⚠️ " if value and flag != "low_conf_ratio" else "   "
        print(f"  {icon}{flag}: {value}")

    print("\n── Obfuscation Findings ──")
    for f in result["obfuscation"].get("findings", []):
        print(f"  → {f['technique']:<28} | risk: {f['risk']} | action: {f['action']}")

    print("\n── Word Table (first 10 rows) ──")
    print(result["words"].head(10).to_string(index=False))