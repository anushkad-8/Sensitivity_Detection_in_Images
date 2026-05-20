"""
main.py — Full Pipeline Entry Point (Phase 1 + Phase 2)
---------------------------------------------------------
OCR-based Sensitive Information Detection System
Barclays Image DLP

Usage:
    python main.py <image_path> [options]

Options:
    --mode        highlight | redact       (default: highlight)
    --psm         Tesseract PSM 3/6/11     (default: 6)
    --conf        OCR confidence threshold (default: 60)
    --no-nlp      Skip NLP classification
    --no-ner      Skip NER (faster, context + doc labels still run)
    --no-encrypt  Save plain JSON report (debug only)
    --debug       Save preprocessing debug images

Examples:
    python main.py input/passport.jpg
    python main.py input/idcard.jpg --mode redact
    python main.py input/screenshot.jpg --psm 11 --no-ner
    python main.py input/scan.jpg --debug --no-encrypt

Pipeline (Phase 1 + Phase 2):
    Image → Preprocess → OCR → Regex Detect → NLP Classify
          → Confidence Score → Annotate → Report → Training Store
"""

import os
import sys
import argparse
import time
from datetime import datetime

from modules.preprocess          import preprocess_image
from modules.ocr_engine          import run_ocr
from modules.sensitive_detector  import detect_sensitive
from modules.nlp_classifier      import classify, is_nlp_available
from modules.confidence_engine   import score_findings, risk_level_icon
from modules.vision_classifier   import classify_image
from modules.annotator           import annotate_image
from modules.reporter            import generate_report
from modules.training_store      import save_scan_records


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR-based Sensitive Information Detector — Barclays Image DLP"
    )
    parser.add_argument("image_path",
        help="Path to input image (JPG, PNG, BMP, TIFF)")
    parser.add_argument("--mode", choices=["highlight", "redact"],
        default="highlight",
        help="Annotation mode (default: highlight)")
    parser.add_argument("--psm", type=int, default=6, choices=[3, 6, 11],
        help="Tesseract PSM: 3=auto, 6=block, 11=sparse/ID cards (default: 6)")
    parser.add_argument("--conf", type=int, default=60,
        help="OCR word confidence threshold 0-100 (default: 60)")
    parser.add_argument("--no-nlp", action="store_true",
        help="Skip NLP classification entirely (Phase 1 only)")
    parser.add_argument("--no-ner", action="store_true",
        help="Skip NER entity detection (context + doc labels still run)")
    parser.add_argument("--no-vision", action="store_true",
        help="Skip Phase 3 vision classification")
    parser.add_argument("--no-encrypt", action="store_true",
        help="Save plain JSON report (debug only)")
    parser.add_argument("--debug", action="store_true",
        help="Save preprocessing debug images to output/debug/")
    return parser.parse_args()


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    image_path  : str,
    mode        : str  = "highlight",
    psm         : int  = 6,
    conf        : int  = 60,
    run_nlp     : bool = True,
    run_ner     : bool = True,
    run_vision  : bool = True,
    encrypt     : bool = True,
    debug       : bool = False,
) -> dict:
    """
    Execute the full Phase 1 + Phase 2 DLP pipeline on a single image.
    """
    pipeline_start = time.time()
    _print_header(image_path, mode, psm, conf, run_nlp, run_vision)

    if not os.path.exists(image_path):
        _exit_error(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
        _exit_error(f"Unsupported file type: {ext}")

    # ─────────────────────────────────────────
    # STEP 1 — PREPROCESS
    # ─────────────────────────────────────────
    _print_step(1, 8, "Preprocessing image")
    t = time.time()
    clean_image, scale_factor = preprocess_image(image_path, save_debug=debug)
    _print_step_done(1, time.time() - t,
                     f"scale={scale_factor}x | shape={clean_image.shape}")

    # ─────────────────────────────────────────
    # STEP 2 — OCR
    # ─────────────────────────────────────────
    _print_step(2, 8, f"Extracting text (PSM {psm}, confidence ≥ {conf})")
    t = time.time()
    ocr_result     = run_ocr(clean_image, psm=psm, confidence_threshold=conf)
    extracted_text = ocr_result["text"]
    words_df       = ocr_result["words"]
    total_words    = ocr_result.get("filtered_word_count", len(words_df))
    raw_words      = ocr_result.get("raw_word_count", total_words)
    dropped_words  = ocr_result.get("dropped_words", max(0, raw_words - total_words))
    _print_step_done(2, time.time() - t,
                     f"{total_words}/{raw_words} words kept | "
                     f"{dropped_words} dropped | {len(extracted_text)} chars")

    if not extracted_text.strip():
        _print_warn("OCR returned empty text — try --psm 11 or lower --conf")

    # ─────────────────────────────────────────
    # STEP 3 — REGEX DETECT
    # ─────────────────────────────────────────
    _print_step(3, 8, "Regex DLP detection (14 pattern types)")
    t = time.time()
    regex_result = detect_sensitive(extracted_text)
    _print_step_done(3, time.time() - t,
                     f"{regex_result['total']} regex finding(s)")

    # ─────────────────────────────────────────
    # STEP 4 — NLP CLASSIFY
    # ─────────────────────────────────────────
    if run_nlp:
        _print_step(4, 8, "NLP classification (context + doc labels + NER)")
        t = time.time()
        nlp_result = classify(
            ocr_text       = extracted_text,
            regex_result   = regex_result,
            run_ner        = run_ner,
            run_doc_labels = True,
            run_context    = True,
        )
        _print_step_done(4, time.time() - t,
                         f"+{len(nlp_result['new_findings'])} NLP findings | "
                         f"{len(nlp_result['context_flags'])} FP flags")
    else:
        print(f"\n  [4/8] NLP skipped (--no-nlp)")
        nlp_result = {
            "is_sensitive" : regex_result["is_sensitive"],
            "total"        : regex_result["total"],
            "matches"      : regex_result["matches"],
            "new_findings" : [],
            "context_flags": [],
            "nlp_available": False,
        }

    # ─────────────────────────────────────────
    # STEP 5 — CONFIDENCE SCORING
    # ─────────────────────────────────────────
    _print_step(5, 8, "Unified confidence scoring")
    t = time.time()

    scored_result = score_findings(
        nlp_result     = nlp_result,
        words_df       = words_df,
        ocr_word_count = total_words,
        dropped_words  = dropped_words,
    )
    # Stash ocr_word_count so _merge_vision_result can apply the evidence gate
    scored_result["_ocr_word_count"]     = total_words
    scored_result["_ocr_raw_word_count"] = raw_words
    _print_step_done(5, time.time() - t,
                     f"overall_risk={scored_result['overall_risk']} | "
                     f"ocr_quality={scored_result['ocr_quality']}")

    # ─────────────────────────────────────────
    # STEP 6 — VISION CLASSIFICATION
    # ─────────────────────────────────────────
    if run_vision:
        _print_step(6, 8, "Vision classification (document type + OCR failure risk)")
        t = time.time()
        vision_result = classify_image(
            image_path      = image_path,
            ocr_word_count = total_words,
        )
        scored_result = _merge_vision_result(scored_result, vision_result)
        _print_step_done(6, time.time() - t,
                         f"{vision_result['document_type']} | "
                         f"{vision_result['sensitivity_level']} | "
                         f"backend={vision_result['model_used']}")
    else:
        print(f"\n  [6/8] Vision skipped (--no-vision)")
        vision_result = {
            "document_type"    : "unknown",
            "type_confidence"  : 0.0,
            "sensitivity_level": "LOW",
            "ocr_failure_risk" : False,
            "vision_findings"  : [],
            "model_used"       : "disabled",
            "fallback"         : False,
        }

    # ─────────────────────────────────────────
    # STEP 7 — ANNOTATE
    # ─────────────────────────────────────────
    _print_step(7, 8, f"Annotating image (mode={mode})")
    t = time.time()

    # Use scored_result for annotation (has all enriched matches)
    annotated_path = annotate_image(
        original_image_path = image_path,
        detection_result    = scored_result,
        words_df            = words_df,
        mode                = mode,
        scale_factor        = scale_factor,
    )
    _print_step_done(7, time.time() - t, f"→ {annotated_path}")

    # ─────────────────────────────────────────
    # STEP 8 — REPORT + TRAINING STORE
    # ─────────────────────────────────────────
    _print_step(8, 8, "Generating report + saving training data")
    t = time.time()

    report = generate_report(
        image_path       = image_path,
        detection_result = scored_result,
        ocr_word_count   = total_words,
        annotated_path   = annotated_path,
        encrypt          = encrypt,
    )

    training = save_scan_records(
        image_path       = image_path,
        detection_result = scored_result,
        ocr_text         = extracted_text,
        nlp_result       = nlp_result,
    )
    _print_step_done(8, time.time() - t,
                     f"report → {os.path.basename(report['report_path'])} | "
                     f"training → {training['saved']} record(s)")

    # ─────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────
    duration = round(time.time() - pipeline_start, 2)
    _print_footer(scored_result, annotated_path, report, duration)

    return {
        "ocr_text"      : extracted_text,
        "regex_result"  : regex_result,
        "nlp_result"    : nlp_result,
        "vision_result" : vision_result,
        "scored_result" : scored_result,
        "annotated_path": annotated_path,
        "report"        : report,
        "training"      : training,
        "duration_sec"  : duration,
    }


# ─────────────────────────────────────────────
# CONSOLE FORMATTING
# ─────────────────────────────────────────────

W = 62

def _print_header(path, mode, psm, conf, run_nlp, run_vision):
    phase = "1+2+3" if run_nlp and run_vision else "1+3" if run_vision else "1+2" if run_nlp else "1"
    print(f"\n{'═' * W}")
    print(f"  🔍 OCR IMAGE DLP — Phase {phase} Pipeline")
    print(f"{'─' * W}")
    print(f"  Image    : {os.path.basename(path)}")
    print(f"  Mode     : {mode.upper()} | PSM: {psm} | Conf: {conf}")
    print(f"  NLP      : {'enabled' if run_nlp else 'disabled (--no-nlp)'}")
    print(f"  Vision   : {'enabled' if run_vision else 'disabled (--no-vision)'}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─' * W}")

def _print_step(n, total, label):
    print(f"\n  [{n}/{total}] {label}...")

def _print_step_done(n, elapsed, detail):
    print(f"  [{n}] ✅ {elapsed:.2f}s — {detail}")

def _print_warn(msg):
    print(f"\n  ⚠️  WARNING: {msg}")

def _print_footer(scored, annotated_path, report, duration):
    print(f"\n{'═' * W}")
    print(f"  PIPELINE COMPLETE — {duration}s total")
    print(f"{'─' * W}")

    risk  = scored["overall_risk"]
    icon  = risk_level_icon(risk)
    total = scored["total"]

    if scored["is_sensitive"]:
        print(f"  {icon} RESULT  : {risk} — {total} finding(s)")
        print(f"{'─' * W}")
        for m in scored["matches"]:
            m_icon = risk_level_icon(m["risk_level"])
            fp_tag = " [FP?]" if m.get("fp_risk") else ""
            print(f"  {m_icon} {m['type']:<18} "
                  f"score={m['unified_score']:.2f}  "
                  f"{m['risk_level']}{fp_tag}")
        dist = scored.get("score_summary", {})
        if dist:
            print(f"{'─' * W}")
            print(f"  Distribution: " +
                  "  ".join(f"{k}:{v}" for k, v in dist.items()))
    else:
        print(f"  ✅ RESULT  : CLEAN — No sensitive content detected")

    print(f"{'─' * W}")
    print(f"  🖼  Annotated : {annotated_path}")
    print(f"  📄 Report    : {report['report_path']}")
    print(f"  📋 Audit log : {report['audit_log_path']}")
    print(f"{'═' * W}\n")

def _exit_error(msg):
    print(f"\n  ❌ ERROR: {msg}\n")
    sys.exit(1)


def _merge_vision_result(scored_result: dict, vision_result: dict) -> dict:
    """
    Merge Phase 3 vision findings into the scored pipeline result using
    weighted evidence-fusion rather than naive appending.

    Architecture principle:
        OCR + Regex + NLP  →  PRIMARY evidence   (can independently trigger alert)
        Vision             →  SUPPORTING evidence (modulates score, rarely triggers alone)

    Decision rules
    ──────────────
    Case 1 — OCR/NLP found evidence AND vision agrees
        Vision boosts the overall score by VISION_BOOST_FACTOR.
        Overall risk can escalate by at most one level.

    Case 2 — OCR failed / low words AND vision flags a sensitive document type
              AND the vision model is not a low-confidence heuristic fallback
        Vision is allowed to contribute a capped MEDIUM finding so the image
        is still flagged for human review — not silently passed.
        This preserves true-positive detection of blurry IDs / scanned docs.

    Case 3 — OCR found nothing AND vision-only detection from heuristic fallback
        Vision finding is injected as REVIEW (not HIGH/CRITICAL) with fp_risk=True.
        The image is queued for analyst review without raising a false alarm.

    Case 4 — OCR found nothing AND vision document_type == "unknown"
        Vision contributes nothing.  Result unchanged.

    In ALL cases:
        is_sensitive is driven by OCR/NLP matches OR by an escalated vision
        finding that passes the evidence gate.  A lone heuristic vision finding
        never sets is_sensitive=True on its own.
    """
    # ── Constants ──────────────────────────────────────────────────────────────
    VISION_BOOST_FACTOR      = 0.08   # score boost when vision corroborates OCR
    VISION_SOLO_CAP_SCORE    = 0.45   # max unified_score for vision-only findings (below MEDIUM=0.50)
    VISION_SOLO_FALLBACK_CAP = 0.28   # max score when backend is heuristic fallback (→ REVIEW)
    OCR_LOW_WORD_THRESHOLD   = 5      # filtered words below this → OCR is sparse
    OCR_RAW_WORD_THRESHOLD   = 8      # raw words below this → Tesseract truly found little text
    MIN_VISION_CONF_SOLO     = 0.50   # vision must clear this confidence to solo-contribute

    merged = dict(scored_result)
    ocr_matches  = list(merged.get("matches", []))
    has_ocr_evidence = bool(ocr_matches)  # any OCR/NLP/regex finding exists

    is_fallback    = vision_result.get("fallback", True)
    doc_type       = vision_result.get("document_type", "unknown")
    vision_conf    = float(vision_result.get("type_confidence", 0.0))
    ocr_failure    = vision_result.get("ocr_failure_risk", False)
    ocr_word_count = merged.get("_ocr_word_count", 0)
    ocr_raw_count  = merged.get("_ocr_raw_word_count", ocr_word_count)

    # TRUE OCR failure: Tesseract found almost no text at all (raw_words < threshold).
    # NOISY OCR: Tesseract found words but filtering removed them (raw high, filtered low).
    # Only true OCR failure justifies elevating a vision-only finding above REVIEW.
    ocr_truly_failed = ocr_word_count < OCR_LOW_WORD_THRESHOLD and ocr_raw_count < OCR_RAW_WORD_THRESHOLD
    ocr_sparse       = ocr_word_count < OCR_LOW_WORD_THRESHOLD

    vision_findings = vision_result.get("vision_findings", [])

    # ── Case 4: vision found nothing meaningful ────────────────────────────────
    if doc_type == "unknown" and not ocr_failure:
        merged["vision_result"] = vision_result
        return merged

    new_vision_matches = []

    for finding in vision_findings:
        vf = dict(finding)
        raw_sensitivity = vf.get("sensitivity_level", vision_result.get("sensitivity_level", "LOW"))
        raw_vision_conf = float(vf.get("vision_confidence", vision_conf))

        # ── Case 1: OCR/NLP has evidence → vision corroborates ────────────────
        if has_ocr_evidence:
            # Boost existing OCR-based scores instead of adding a new finding.
            # Only boost if vision is confident and agrees this is a sensitive doc.
            if raw_vision_conf >= MIN_VISION_CONF_SOLO and raw_sensitivity in ("HIGH", "CRITICAL", "MEDIUM"):
                for m in ocr_matches:
                    boosted = min(1.0, m.get("unified_score", 0.0) + VISION_BOOST_FACTOR)
                    m["unified_score"] = round(boosted, 3)
                    # Re-derive risk level with the same thresholds used by confidence_engine
                    m["risk_level"] = _score_to_risk_level(boosted)
                    m.setdefault("score_detail", {})["vision_boost"] = VISION_BOOST_FACTOR
                    m["score_detail"]["vision_type"]  = doc_type
                    m["score_detail"]["vision_conf"]  = raw_vision_conf
            # Do NOT add an independent vision finding — OCR evidence is primary.
            continue

        # ── Cases 2 & 3: No OCR evidence (vision-only path) ───────────────────
        vf.setdefault("nlp_confidence", "none")

        if ocr_truly_failed and not is_fallback and raw_vision_conf >= MIN_VISION_CONF_SOLO:
            # Case 2: OCR legitimately failed on a structured document + vision
            #         is confident + using a real model (not pure heuristic).
            #         Contribute a MEDIUM finding to flag for analyst review.
            capped_score = min(raw_vision_conf, VISION_SOLO_CAP_SCORE)
            vf["unified_score"] = round(capped_score, 3)
            vf["risk_level"]    = _score_to_risk_level(capped_score)   # ≤ MEDIUM
            vf["fp_risk"]       = False
            vf["vision_solo"]   = True
            vf["vision_evidence_gate"] = "ocr_failure_confirmed"
        else:
            # Case 3: Heuristic or low-confidence fallback with no OCR support.
            #         Cap at REVIEW and mark fp_risk so analysts can review.
            if is_fallback:
                capped_score = min(raw_vision_conf, VISION_SOLO_FALLBACK_CAP)
            else:
                capped_score = min(raw_vision_conf, VISION_SOLO_CAP_SCORE)
            vf["unified_score"] = round(capped_score, 3)
            vf["risk_level"]    = "REVIEW"
            vf["fp_risk"]       = True
            vf["vision_solo"]   = True
            vf["vision_evidence_gate"] = "heuristic_only_no_ocr"

        vf["score_detail"] = {
            "vision_score"    : raw_vision_conf,
            "capped_score"    : vf["unified_score"],
            "model_used"      : vision_result.get("model_used", "opencv_heuristics"),
            "fallback"        : is_fallback,
            "ocr_failure_risk": ocr_failure,
            "evidence_gate"   : vf.get("vision_evidence_gate"),
        }
        new_vision_matches.append(vf)

    # ── Assemble final match list ──────────────────────────────────────────────
    all_matches = ocr_matches + new_vision_matches

    # is_sensitive: driven by OCR/NLP, OR by a non-REVIEW vision finding
    #   (Case 2 only — OCR failure with confident non-fallback model)
    non_review_vision = [m for m in new_vision_matches if m.get("risk_level") != "REVIEW"]
    is_sensitive = has_ocr_evidence or bool(non_review_vision)

    merged["matches"]      = all_matches
    merged["total"]        = len(all_matches)
    merged["is_sensitive"] = is_sensitive
    merged["overall_risk"] = _highest_risk(all_matches) if is_sensitive else "NONE"
    merged["score_summary"]= _score_summary(all_matches)
    merged["vision_result"]= vision_result
    return merged


def _score_to_risk_level(score: float) -> str:
    """Mirror the confidence_engine thresholds so vision-boosted scores map consistently."""
    if score >= 0.85: return "CRITICAL"
    if score >= 0.70: return "HIGH"
    if score >= 0.50: return "MEDIUM"
    if score >= 0.30: return "LOW"
    return "REVIEW"


def _highest_risk(matches: list) -> str:
    priority = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "REVIEW": 1, "NONE": 0}
    if not matches:
        return "NONE"
    return max((m.get("risk_level", "NONE") for m in matches),
               key=lambda r: priority.get(r, 0))


def _score_summary(matches: list) -> dict:
    summary = {}
    for match in matches:
        level = match.get("risk_level", "REVIEW")
        summary[level] = summary.get(level, 0) + 1
    return summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        image_path = args.image_path,
        mode       = args.mode,
        psm        = args.psm,
        conf       = args.conf,
        run_nlp    = not args.no_nlp,
        run_ner    = not args.no_ner,
        run_vision = not args.no_vision,
        encrypt    = not args.no_encrypt,
        debug      = args.debug,
    )