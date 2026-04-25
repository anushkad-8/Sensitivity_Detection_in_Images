"""
Module 4: annotator.py
-----------------------
Visual output layer — draws highlights or redaction boxes on the original image
over detected sensitive regions.

Architecture position:
    sensitive_detector.py (matches + tokens + obfuscation_flags)
            +
    ocr_engine.py (word bounding boxes)
            ↓
    annotator.py → annotated / redacted image saved to output/

Two modes:
    HIGHLIGHT : Red bounding box drawn around sensitive word(s)
    REDACT    : Black filled box covering sensitive word(s) completely

CHANGES (Anti-Obfuscation Update):
    1. annotate_image() now reads detection_result["obfuscation_flags"] and
       detection_result["escalate"] to show an escalation banner when non-recoverable
       obfuscation (encryption/hashing) is detected.
    2. _draw_escalation_banner() added — draws a prominent red banner at the top
       of images that require human review due to obfuscation techniques.
    3. _draw_watermark() extended to include risk_level in the watermark text.
    4. All core drawing logic (highlight, redact, bounding boxes) unchanged.
    5. scale_factor bug fix retained from previous version.
"""

import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────
# VISUAL STYLE CONFIG
# ─────────────────────────────────────────────

STYLE = {
    "highlight_box_color"    : (0, 0, 255),
    "highlight_box_thickness": 2,
    "label_bg_color"         : (0, 0, 255),
    "label_text_color"       : (255, 255, 255),
    "label_font_scale"       : 0.45,
    "label_font_thickness"   : 1,
    "label_padding"          : 3,
    "redact_fill_color"      : (0, 0, 0),
    "redact_label_color"     : (80, 80, 80),
    "box_padding"            : 4,
    "watermark_text"         : "SENSITIVE CONTENT DETECTED — OCR DLP",
    "watermark_color"        : (0, 0, 180),
    "watermark_font_scale"   : 0.55,
    "watermark_thickness"    : 1,
    # Escalation banner — shown when encryption/hashing is detected
    "escalation_banner_color": (0, 0, 200),      # Deep red (BGR)
    "escalation_text_color"  : (255, 255, 255),
    "escalation_font_scale"  : 0.55,
    "escalation_thickness"   : 1,
    "escalation_banner_height": 28,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def annotate_image(
    original_image_path : str,
    detection_result    : dict,
    words_df            : pd.DataFrame,
    mode                : str   = "highlight",
    scale_factor        : float = 1.0
) -> str:
    """
    Annotate or redact the original image based on detection results.

    Args:
        original_image_path : Path to the original input image.
        detection_result    : Output dict from sensitive_detector.detect_sensitive().
                              Now includes: obfuscation_flags, escalate, risk_level.
        words_df            : Word DataFrame from ocr_engine.run_ocr()["words"].
        mode                : "highlight" → red boxes | "redact" → black fill.
        scale_factor        : From preprocess_image(). Divide OCR coords by this value.

    Returns:
        output_path : Path where the annotated image was saved.
    """
    _validate_inputs(original_image_path, mode)

    image = cv2.imread(original_image_path)
    if image is None:
        raise ValueError(f"[Annotator] Could not load image: {original_image_path}")

    obfuscation_flags = detection_result.get("obfuscation_flags", {})
    escalate          = detection_result.get("escalate", False)
    risk_level        = detection_result.get("risk_level", "none")

    print(f"\n[Annotator] Loaded: {original_image_path}")
    print(f"[Annotator] Mode: {mode.upper()} | Matches: {detection_result['total']} "
          f"| Risk: {risk_level.upper()} | Escalate: {escalate} | Scale: {scale_factor}")

    # No sensitive content AND no obfuscation → save clean
    if not detection_result["is_sensitive"] and not escalate:
        print("[Annotator] No sensitive content — saving clean image.")
        return _save_image(image, original_image_path, mode, clean=True)

    annotated   = image.copy()
    drawn_count = 0

    # ── Draw escalation banner FIRST (top of image) ───────────────────────────
    # This appears even if we couldn't draw bounding boxes (e.g. hashed content
    # was detected but regex couldn't match the hash back to a word box).
    if escalate:
        _draw_escalation_banner(annotated, obfuscation_flags)

    # ── Draw bounding boxes for regex matches ─────────────────────────────────
    for match in detection_result["matches"]:
        box = _get_merged_box(match["tokens"], words_df, scale_factor)
        if box is None:
            print(f"[Annotator] No bounding box for: '{match['value']}' — skipping.")
            continue
        if mode == "highlight":
            _draw_highlight(annotated, box, match)
        else:
            _draw_redaction(annotated, box, match)
        drawn_count += 1

    # ── Draw watermark with risk level ────────────────────────────────────────
    if drawn_count > 0 or escalate:
        _draw_watermark(annotated, detection_result["total"], risk_level, escalate)

    print(f"[Annotator] Drew {drawn_count}/{detection_result['total']} findings on image.")
    output_path = _save_image(annotated, original_image_path, mode)
    return output_path


# ─────────────────────────────────────────────
# ESCALATION BANNER — NEW
# ─────────────────────────────────────────────

def _draw_escalation_banner(image: np.ndarray, obfuscation_flags: dict) -> None:
    """
    Draw a prominent red banner at the top of the image when non-recoverable
    obfuscation (encryption or hashing) is detected.

    This banner signals that the image MUST be reviewed by a human — automated
    regex detection alone is insufficient because the sensitive data could not
    be recovered from the encrypted/hashed form.

    The banner is always drawn regardless of whether bounding boxes were found.
    """
    h, w    = image.shape[:2]
    bh      = STYLE["escalation_banner_height"]
    color   = STYLE["escalation_banner_color"]
    t_color = STYLE["escalation_text_color"]
    fscale  = STYLE["escalation_font_scale"]
    thick   = STYLE["escalation_thickness"]

    # Build banner text from flags
    reasons = []
    if obfuscation_flags.get("encrypted_content"):
        reasons.append("ENCRYPTED CONTENT")
    if obfuscation_flags.get("hash_patterns_found"):
        reasons.append("HASH PATTERNS")
    if obfuscation_flags.get("encoded_tokens_found"):
        reasons.append("ENCODED TOKENS")

    reason_str = " | ".join(reasons) if reasons else "OBFUSCATION DETECTED"
    banner_text = f"⚠ HUMAN REVIEW REQUIRED — {reason_str}"

    # Draw semi-transparent background strip
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Draw text centered in banner
    (tw, th), baseline = cv2.getTextSize(banner_text, FONT, fscale, thick)
    text_x = max(8, (w - tw) // 2)
    text_y = (bh + th) // 2

    cv2.putText(image, banner_text, (text_x, text_y),
                FONT, fscale, t_color, thick, cv2.LINE_AA)

    print(f"[Annotator] Escalation banner drawn: {reason_str}")


# ─────────────────────────────────────────────
# BOUNDING BOX LOGIC (unchanged)
# ─────────────────────────────────────────────

def _get_merged_box(tokens: list, words_df: pd.DataFrame, scale_factor: float = 1.0) -> dict | None:
    """
    Find and merge bounding boxes for all tokens of a match.
    Excludes channel-OCR tokens (left == -1, no valid coordinate).
    """
    if words_df.empty or not tokens:
        return None

    # Exclude tokens added by channel OCR (no valid bounding box)
    valid_df = words_df[words_df["left"] >= 0] if "left" in words_df.columns else words_df
    matched_rows = valid_df[valid_df["word"].isin(tokens)]

    if matched_rows.empty:
        return None

    pad    = STYLE["box_padding"]
    left   = int(matched_rows["left"].min())
    top    = int(matched_rows["top"].min())
    right  = int((matched_rows["left"] + matched_rows["width"]).max())
    bottom = int((matched_rows["top"]  + matched_rows["height"]).max())

    left   = int(left   / scale_factor) - pad
    top    = int(top    / scale_factor) - pad
    right  = int(right  / scale_factor) + pad
    bottom = int(bottom / scale_factor) + pad

    left = max(0, left)
    top  = max(0, top)

    return {"left": left, "top": top, "right": right, "bottom": bottom}


# ─────────────────────────────────────────────
# DRAWING — HIGHLIGHT MODE (unchanged)
# ─────────────────────────────────────────────

def _draw_highlight(image: np.ndarray, box: dict, match: dict) -> None:
    x1, y1 = box["left"],  box["top"]
    x2, y2 = box["right"], box["bottom"]

    color     = STYLE["highlight_box_color"]
    thickness = STYLE["highlight_box_thickness"]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label      = match["type"].upper().replace("_", " ")
    conf       = match["confidence"][0].upper()
    label_text = f"{label} [{conf}]"

    font_scale = STYLE["label_font_scale"]
    font_thick = STYLE["label_font_thickness"]
    pad        = STYLE["label_padding"]

    (text_w, text_h), baseline = cv2.getTextSize(label_text, FONT, font_scale, font_thick)
    label_y1 = max(0, y1 - text_h - 2 * pad - baseline)
    label_y2 = y1
    label_x2 = min(image.shape[1], x1 + text_w + 2 * pad)

    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2),
                  STYLE["label_bg_color"], cv2.FILLED)
    cv2.putText(image, label_text, (x1 + pad, y1 - pad - baseline),
                FONT, font_scale, STYLE["label_text_color"], font_thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
# DRAWING — REDACT MODE (unchanged)
# ─────────────────────────────────────────────

def _draw_redaction(image: np.ndarray, box: dict, match: dict) -> None:
    x1, y1 = box["left"],  box["top"]
    x2, y2 = box["right"], box["bottom"]

    cv2.rectangle(image, (x1, y1), (x2, y2), STYLE["redact_fill_color"], cv2.FILLED)

    label_text = f"[{match['type'].upper().replace('_', ' ')} REDACTED]"
    font_scale = STYLE["label_font_scale"]
    font_thick = STYLE["label_font_thickness"]

    label_x = x2 + 6
    label_y = y1 + (y2 - y1) // 2

    (text_w, _), _ = cv2.getTextSize(label_text, FONT, font_scale, font_thick)
    if label_x + text_w > image.shape[1]:
        label_x = max(0, x1)
        label_y = max(12, y1 - 6)

    cv2.putText(image, label_text, (label_x, label_y),
                FONT, font_scale, STYLE["redact_label_color"], font_thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
# WATERMARK — extended with risk level
# ─────────────────────────────────────────────

def _draw_watermark(image: np.ndarray, finding_count: int,
                    risk_level: str = "medium", escalate: bool = False) -> None:
    h, w      = image.shape[:2]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk_tag  = f"RISK:{risk_level.upper()}"
    escalate_tag = " | ESCALATED" if escalate else ""
    text = (f"{STYLE['watermark_text']} | {finding_count} finding(s) "
            f"| {risk_tag}{escalate_tag} | {timestamp}")

    font_scale = STYLE["watermark_font_scale"]
    thickness  = STYLE["watermark_thickness"]

    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    x = 10
    y = h - 10 - baseline

    strip_y1 = y - text_h - 6
    overlay  = image.copy()
    cv2.rectangle(overlay, (0, strip_y1), (w, h), (255, 255, 255), cv2.FILLED)
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
    cv2.putText(image, text, (x, y), FONT, font_scale,
                STYLE["watermark_color"], thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────
# SAVE OUTPUT (unchanged)
# ─────────────────────────────────────────────

def _save_image(image: np.ndarray, original_path: str, mode: str, clean: bool = False) -> str:
    os.makedirs("output", exist_ok=True)
    base        = os.path.splitext(os.path.basename(original_path))[0]
    suffix      = "clean" if clean else ("annotated" if mode == "highlight" else "redacted")
    output_path = os.path.join("output", f"{base}_{suffix}.jpg")
    cv2.imwrite(output_path, image)
    print(f"[Annotator] Saved: {output_path}")
    return output_path


def _validate_inputs(image_path: str, mode: str) -> None:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[Annotator] Image not found: {image_path}")
    if mode not in ("highlight", "redact"):
        raise ValueError(f"[Annotator] Invalid mode: '{mode}'. Must be 'highlight' or 'redact'.")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from modules.preprocess        import preprocess_image
    from modules.ocr_engine        import run_ocr
    from modules.sensitive_detector import detect_sensitive

    if len(sys.argv) < 2:
        print("Usage: python modules/annotator.py <image_path> [highlight|redact]")
        sys.exit(1)

    img_path = sys.argv[1]
    mode     = sys.argv[2] if len(sys.argv) >= 3 else "highlight"

    print("\n── Step 1: Preprocess ──")
    clean, scale_factor, channel_images = preprocess_image(img_path, save_debug=False)

    print("\n── Step 2: OCR ──")
    ocr_result = run_ocr(clean, channel_images=channel_images)

    print("\n── Step 3: Detect ──")
    detection = detect_sensitive(
        ocr_result["text_for_detection"],
        obfuscation_flags=ocr_result["obfuscation_flags"]
    )

    print("\n── Step 4: Annotate ──")
    out = annotate_image(img_path, detection, ocr_result["words"],
                         mode=mode, scale_factor=scale_factor)

    print(f"\n── Done. Output saved: {out} ──")