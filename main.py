"""
main.py
-------
Entry point for the OCR DLP pipeline.

Wires all 5 modules together in order:
    1. preprocess.py        → clean image + scale factor + channel images
    2. ocr_engine.py        → extracted text + word bounding boxes + obfuscation scan
    3. anti_obfuscation.py  → called internally by ocr_engine (no direct call needed)
    4. sensitive_detector.py → regex detection + risk level + escalation flag
    5. annotator.py         → annotated / redacted output image

Usage:
    python main.py <image_path> [--mode highlight|redact] [--debug]

Examples:
    python main.py input/aadhaar.jpg
    python main.py input/passport.png --mode redact
    python main.py input/sample.jpg --mode highlight --debug

Output:
    Annotated image saved to: output/<filename>_annotated.jpg  (highlight mode)
                           or: output/<filename>_redacted.jpg   (redact mode)
    JSON report saved to     : output/<filename>_report.json
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ── Add project root to path so 'modules' folder is importable ───────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.preprocess         import preprocess_image
from modules.ocr_engine         import run_ocr
from modules.sensitive_detector import detect_sensitive
from modules.annotator          import annotate_image


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(image_path: str, mode: str = "highlight", debug: bool = False) -> dict:
    """
    Run the full OCR DLP pipeline on a single image.

    Args:
        image_path : Path to input image (JPG, PNG, BMP, TIFF).
        mode       : "highlight" → red bounding boxes drawn on image.
                     "redact"    → black fill over sensitive words.
        debug      : If True, saves intermediate preprocessing images to output/debug/

    Returns:
        Full result dict with keys:
            image_path, mode, ocr_text, risk_level, escalate,
            total_matches, matches, obfuscation_flags, output_image_path, report_path
    """

    print("\n" + "═" * 60)
    print(f"  OCR DLP PIPELINE")
    print(f"  Image : {image_path}")
    print(f"  Mode  : {mode.upper()}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    # ─────────────────────────────────────────
    # STEP 1: PREPROCESS
    # ─────────────────────────────────────────
    print("\n── STEP 1: Preprocessing image ──")
    clean_image, scale_factor, channel_images = preprocess_image(
        image_path,
        save_debug=debug
    )

    # ─────────────────────────────────────────
    # STEP 2: OCR + ANTI-OBFUSCATION
    # ─────────────────────────────────────────
    # run_ocr() internally calls:
    #   - reassemble_fragments()     → fixes split digit sequences
    #   - normalize_ocr_text()       → cleans homoglyphs + zero-width chars
    #   - channel OCR merge          → catches colour-channel-hidden text
    #   - anti_obfuscation.detect_all() → handles encoding, encryption, hashing
    print("\n── STEP 2: OCR extraction + anti-obfuscation ──")
    ocr_result = run_ocr(
        clean_image,
        channel_images=channel_images
    )

    print(f"\n[Main] Raw text       : {ocr_result['text'][:120]}")
    print(f"[Main] Cleaned text   : {ocr_result['text_normalized'][:120]}")
    print(f"[Main] Detection text : {ocr_result['text_for_detection'][:120]}")
    print(f"[Main] Digit runs     : {ocr_result['digit_runs']}")
    print(f"[Main] Obfuscation    : {ocr_result['obfuscation_flags']}")

    # ─────────────────────────────────────────
    # STEP 3: SENSITIVE DATA DETECTION
    # ─────────────────────────────────────────
    # IMPORTANT: We pass text_for_detection — the cleanest version:
    #   - homoglyphs already mapped to Latin
    #   - zero-width chars stripped
    #   - base64/hex tokens replaced with their decoded plaintext
    # We also pass obfuscation_flags so the detector can set escalation.
    print("\n── STEP 3: Sensitive data detection ──")
    detection = detect_sensitive(
        ocr_result["text_for_detection"],
        obfuscation_flags=ocr_result["obfuscation_flags"]
    )

    print(f"\n[Main] is_sensitive : {detection['is_sensitive']}")
    print(f"[Main] risk_level   : {detection['risk_level'].upper()}")
    print(f"[Main] escalate     : {detection['escalate']}")
    print(f"[Main] total matches: {detection['total']}")

    # ─────────────────────────────────────────
    # STEP 4: ANNOTATE / REDACT IMAGE
    # ─────────────────────────────────────────
    print("\n── STEP 4: Annotating image ──")
    output_image_path = annotate_image(
        original_image_path=image_path,
        detection_result=detection,
        words_df=ocr_result["words"],
        mode=mode,
        scale_factor=scale_factor
    )

    # ─────────────────────────────────────────
    # STEP 5: SAVE JSON REPORT
    # ─────────────────────────────────────────
    report_path = _save_report(image_path, ocr_result, detection, output_image_path, mode)

    # ─────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Risk level     : {detection['risk_level'].upper()}")
    print(f"  Matches found  : {detection['total']}")
    print(f"  Escalate flag  : {detection['escalate']}")
    print(f"  Output image   : {output_image_path}")
    print(f"  Report saved   : {report_path}")
    print("═" * 60 + "\n")

    return {
        "image_path"        : image_path,
        "mode"              : mode,
        "ocr_text"          : ocr_result["text"],
        "ocr_text_cleaned"  : ocr_result["text_for_detection"],
        "digit_runs"        : ocr_result["digit_runs"],
        "risk_level"        : detection["risk_level"],
        "escalate"          : detection["escalate"],
        "is_sensitive"      : detection["is_sensitive"],
        "total_matches"     : detection["total"],
        "matches"           : detection["matches"],
        "obfuscation_flags" : ocr_result["obfuscation_flags"],
        "obfuscation_detail": ocr_result["obfuscation"]["findings"],
        "output_image_path" : output_image_path,
        "report_path"       : report_path,
    }


# ─────────────────────────────────────────────
# JSON REPORT
# ─────────────────────────────────────────────

def _save_report(image_path: str, ocr_result: dict,
                 detection: dict, output_image_path: str, mode: str) -> str:
    """
    Save a structured JSON report of the pipeline run.
    Useful for audit trails and integration with SIEM/ticketing systems.
    """
    os.makedirs("output", exist_ok=True)
    base        = os.path.splitext(os.path.basename(image_path))[0]
    report_path = os.path.join("output", f"{base}_report.json")

    report = {
        "timestamp"          : datetime.now().isoformat(),
        "image_path"         : image_path,
        "output_image"       : output_image_path,
        "mode"               : mode,
        "ocr_summary": {
            "raw_text"           : ocr_result["text"],
            "normalized_text"    : ocr_result["text_normalized"],
            "text_for_detection" : ocr_result["text_for_detection"],
            "digit_runs"         : ocr_result["digit_runs"],
            "total_words"        : len(ocr_result["words"]),
        },
        "obfuscation": {
            "flags"   : ocr_result["obfuscation_flags"],
            "findings": ocr_result["obfuscation"]["findings"],
        },
        "detection": {
            "is_sensitive": detection["is_sensitive"],
            "risk_level"  : detection["risk_level"],
            "escalate"    : detection["escalate"],
            "total"       : detection["total"],
            "matches"     : detection["matches"],
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Main] Report saved: {report_path}")
    return report_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="OCR DLP Pipeline — detects sensitive information in images."
    )
    parser.add_argument(
        "image_path",
        help="Path to the input image (JPG, PNG, BMP, TIFF)"
    )
    parser.add_argument(
        "--mode",
        choices=["highlight", "redact"],
        default="highlight",
        help="Output mode: 'highlight' draws red boxes, 'redact' fills with black. (default: highlight)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate preprocessing images to output/debug/"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()

    if not os.path.exists(args.image_path):
        print(f"[Error] Image not found: {args.image_path}")
        sys.exit(1)

    result = run_pipeline(
        image_path=args.image_path,
        mode=args.mode,
        debug=args.debug
    )

    # Exit code signals risk to calling scripts/CI:
    # 0 = clean, 1 = sensitive found, 2 = escalation required
    if result["escalate"]:
        sys.exit(2)
    elif result["is_sensitive"]:
        sys.exit(1)
    else:
        sys.exit(0)