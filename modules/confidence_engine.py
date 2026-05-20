"""
modules/confidence_engine.py
------------------------------
Phase 2 — Unified Confidence Scoring Engine.

BARCLAYS FEEDBACK:
    "Confidence scoring is critical — how are partial OCR results
     and low confidence outputs handled?"

    This module answers that directly. It takes all signals from the
    pipeline and produces one unified confidence score per finding.

WHAT IT DOES:
    Each finding has been scored independently by:
        - Tesseract OCR     → per-word confidence (0-100)
        - Regex detector    → pattern specificity (high/medium/low)
        - NLP classifier    → context verification + entity score

    This module combines all three into a single UNIFIED score (0.0 - 1.0)
    and a final RISK LEVEL (CRITICAL / HIGH / MEDIUM / LOW / REVIEW).

    It also handles the edge cases Barclays asked about:
        - Partial OCR results (low word count, many dropped words)
        - Low OCR confidence words feeding into detections
        - FP-flagged matches from NLP context verification

SCORING FORMULA (weighted combination):
    unified_score = (
        regex_weight    * regex_score   +
        nlp_weight      * nlp_score     +
        ocr_weight      * ocr_score
    )

    Weights (tunable):
        regex : 0.50  — regex pattern specificity is the primary signal
        nlp   : 0.35  — NLP context adds strong secondary signal
        ocr   : 0.15  — OCR word-level confidence is a quality modifier

RISK LEVELS:
    CRITICAL  : score >= 0.85 — act immediately, no review needed
    HIGH      : score >= 0.70 — very likely sensitive, review recommended
    MEDIUM    : score >= 0.50 — probably sensitive, analyst review needed
    LOW       : score >= 0.30 — possible sensitive, low priority review
    REVIEW    : score <  0.30 or fp_risk=True — likely false positive

OUTPUT:
    Each finding gets two new fields:
        unified_score : float 0.0-1.0
        risk_level    : CRITICAL / HIGH / MEDIUM / LOW / REVIEW

    The pipeline result gets:
        overall_risk  : highest risk level across all findings
        ocr_quality   : assessment of OCR output quality
        score_summary : breakdown of score distribution
"""

import os
from typing import Optional


# ─────────────────────────────────────────────
# SCORING WEIGHTS
# ─────────────────────────────────────────────
# Tunable — adjust based on empirical performance.
# Must sum to 1.0.

WEIGHTS = {
    "regex": 0.50,
    "nlp"  : 0.35,
    "ocr"  : 0.15,
}

# ─────────────────────────────────────────────
# REGEX PATTERN SCORES
# ─────────────────────────────────────────────
# Base score contribution from regex confidence level.
# Reflects how specific/unambiguous the pattern is.

REGEX_SCORES = {
    "high"  : 1.00,   # PAN, passport, GST, IFSC, SWIFT, email — near-zero FP
    "medium": 0.65,   # Aadhaar, bank card, voter ID, MICR — some FP risk
    "low"   : 0.35,   # DOB, driving licence — context-dependent
    "none"  : 0.00,   # NLP-only finding — no regex score
}

# ─────────────────────────────────────────────
# NLP SCORE MAPPING
# ─────────────────────────────────────────────
# Maps NLP classification output to a numeric score.

NLP_SCORES = {
    "high"  : 1.00,   # Document label, high-conf NER, confirmed context
    "medium": 0.65,   # NER entity, upgraded regex match
    "low"   : 0.30,   # Unverified context, low-conf NER
    "none"  : 0.50,   # No NLP ran — neutral, don't penalise
    "spacy" : 0.65,   # spaCy NER default
}

# ─────────────────────────────────────────────
# RISK LEVEL THRESHOLDS
# ─────────────────────────────────────────────

RISK_THRESHOLDS = [
    (0.85, "CRITICAL"),
    (0.70, "HIGH"),
    (0.50, "MEDIUM"),
    (0.30, "LOW"),
    (0.00, "REVIEW"),   # also forced when fp_risk=True
]

# ─────────────────────────────────────────────
# OCR QUALITY THRESHOLDS
# ─────────────────────────────────────────────
# Used to assess overall OCR reliability for a scan.
# Affects the OCR component of the score.

OCR_QUALITY_LEVELS = {
    "good"    : (70, 1.00),   # mean conf >= 70 → full OCR contribution
    "moderate": (50, 0.70),   # mean conf >= 50 → partial OCR contribution
    "poor"    : (0,  0.40),   # mean conf < 50  → low OCR contribution
}

# ─────────────────────────────────────────────
# FINDING TYPE OCR WORD MAPPING
# ─────────────────────────────────────────────
# Maps finding type → expected tokens in OCR word list.
# Used to look up per-word confidence from Tesseract.

MULTI_TOKEN_TYPES = {
    "phone", "aadhaar", "bank_card", "dob",
    "driving_licence", "mrz_line"
}


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def score_findings(
    nlp_result      : dict,
    words_df        ,             # pandas DataFrame from ocr_engine
    ocr_word_count  : int = 0,
    dropped_words   : int = 0,
) -> dict:
    """
    Compute unified confidence scores for all findings.

    Args:
        nlp_result     : Output from nlp_classifier.classify()
                         Contains enriched matches with regex + NLP signals.
        words_df       : Word DataFrame from ocr_engine.run_ocr()["words"]
                         Used to look up per-word OCR confidence scores.
        ocr_word_count : Number of words kept after confidence filtering.
        dropped_words  : Number of words dropped by OCR confidence filter.
                         High ratio = poor OCR quality = lower overall trust.

    Returns:
        {
            "is_sensitive"  : bool
            "total"         : int
            "matches"       : list — all findings with unified_score + risk_level
            "overall_risk"  : str  — highest risk level across all findings
            "ocr_quality"   : str  — good / moderate / poor
            "score_summary" : dict — count per risk level
            "new_findings"  : list — NLP-only findings (passed through)
            "context_flags" : list — FP-flagged findings (passed through)
        }
    """

    print(f"\n[ConfidenceEngine] Scoring {nlp_result.get('total', 0)} findings...")

    # ── Assess OCR quality ────────────────────────────────────────────────────
    ocr_quality, ocr_quality_score = _assess_ocr_quality(
        words_df, ocr_word_count, dropped_words
    )
    print(f"[ConfidenceEngine] OCR quality: {ocr_quality} "
          f"(score modifier: {ocr_quality_score:.2f})")

    # ── Score each finding ────────────────────────────────────────────────────
    scored_matches = []
    for match in nlp_result.get("matches", []):
        scored = _score_finding(match, words_df, ocr_quality_score)
        scored_matches.append(scored)
        print(
            f"[ConfidenceEngine]   {scored['type']:<18} "
            f"score={scored['unified_score']:.2f}  "
            f"risk={scored['risk_level']}"
        )

    # ── Overall risk level ────────────────────────────────────────────────────
    overall_risk  = _compute_overall_risk(scored_matches)
    score_summary = _build_score_summary(scored_matches)

    print(f"[ConfidenceEngine] Overall risk: {overall_risk}")
    print(f"[ConfidenceEngine] Score distribution: {score_summary}")

    return {
        "is_sensitive" : nlp_result.get("is_sensitive", False),
        "total"        : len(scored_matches),
        "matches"      : scored_matches,
        "overall_risk" : overall_risk,
        "ocr_quality"  : ocr_quality,
        "score_summary": score_summary,
        "new_findings" : nlp_result.get("new_findings", []),
        "context_flags": nlp_result.get("context_flags", []),
        "nlp_available": nlp_result.get("nlp_available", False),
    }


# ─────────────────────────────────────────────
# PER-FINDING SCORER
# ─────────────────────────────────────────────

def _score_finding(
    match           : dict,
    words_df        ,
    ocr_quality_score: float
) -> dict:
    """
    Compute unified_score and risk_level for a single finding.

    Three-signal scoring:
        1. Regex score   → from pattern type confidence (high/medium/low)
        2. NLP score     → from context verification + NER result
        3. OCR score     → mean Tesseract confidence of matching words
                           weighted by overall OCR quality

    FP override:
        If NLP flagged this match as fp_risk=True, cap risk at REVIEW
        regardless of score. Safety is maintained — match stays in results
        but is clearly marked for human review.
    """
    m = dict(match)

    # ── Signal 1: Regex score ─────────────────────────────────────────────────
    regex_conf  = m.get("confidence", "low")
    regex_score = REGEX_SCORES.get(regex_conf, 0.35)

    # NLP-only findings (source=nlp) have no regex pattern
    if m.get("source") == "nlp":
        regex_score = 0.50   # neutral — NLP found it without regex

    # ── Signal 2: NLP score ───────────────────────────────────────────────────
    nlp_conf  = m.get("nlp_confidence", "none")
    nlp_score = NLP_SCORES.get(str(nlp_conf).lower(), 0.50)

    # Document labels are always high NLP confidence
    if m.get("type") == "document_label":
        nlp_score = 1.00

    # ── Signal 3: OCR word confidence ─────────────────────────────────────────
    ocr_score = _get_ocr_word_score(m, words_df, ocr_quality_score)

    # ── Weighted combination ──────────────────────────────────────────────────
    unified_score = (
        WEIGHTS["regex"] * regex_score +
        WEIGHTS["nlp"]   * nlp_score   +
        WEIGHTS["ocr"]   * ocr_score
    )
    unified_score = round(min(1.0, max(0.0, unified_score)), 3)
    if m.get("fp_risk") is True:
        unified_score = min(unified_score, 0.29)

    # ── Risk level ────────────────────────────────────────────────────────────
    risk_level = _score_to_risk(unified_score)

    # FP override — NLP flagged as likely false positive
    if m.get("fp_risk") is True:
        risk_level = "REVIEW"

    # ── Attach scores to match ────────────────────────────────────────────────
    m["unified_score"] = unified_score
    m["risk_level"]    = risk_level
    m["score_detail"]  = {
        "regex_score"     : round(regex_score, 3),
        "nlp_score"       : round(nlp_score, 3),
        "ocr_score"       : round(ocr_score, 3),
        "regex_weight"    : WEIGHTS["regex"],
        "nlp_weight"      : WEIGHTS["nlp"],
        "ocr_weight"      : WEIGHTS["ocr"],
    }

    return m


# ─────────────────────────────────────────────
# OCR WORD SCORE
# ─────────────────────────────────────────────

def _get_ocr_word_score(
    match            : dict,
    words_df         ,
    ocr_quality_score: float
) -> float:
    """
    Get mean Tesseract confidence for the words that form this finding.

    If word tokens are found in the OCR DataFrame, use their actual
    Tesseract confidence scores. If not found (e.g. NLP-only finding
    or OCR variance), fall back to the overall OCR quality score.

    Normalised from Tesseract's 0-100 scale to 0.0-1.0.
    """
    try:
        if words_df is None or words_df.empty:
            return ocr_quality_score

        tokens   = match.get("tokens", [])
        if not tokens:
            return ocr_quality_score

        matched  = words_df[words_df["word"].isin(tokens)]
        if matched.empty:
            return ocr_quality_score

        mean_conf = matched["conf"].mean()
        return round(float(mean_conf) / 100.0, 3)

    except Exception:
        return ocr_quality_score


# ─────────────────────────────────────────────
# OCR QUALITY ASSESSMENT
# ─────────────────────────────────────────────

def _assess_ocr_quality(
    words_df        ,
    ocr_word_count  : int,
    dropped_words   : int
) -> tuple:
    """
    Assess overall OCR quality for this scan.

    Two signals:
        1. Mean Tesseract confidence of retained words
        2. Drop ratio = dropped / (dropped + retained)
           High drop ratio → many low-confidence words → poor image quality

    This directly addresses Barclays' question about how partial OCR
    results are handled — poor OCR quality reduces the OCR signal weight
    in the unified score, making the system more conservative.

    Returns:
        (quality_label, quality_score)
        quality_label : "good" / "moderate" / "poor"
        quality_score : float 0.0-1.0 used as OCR signal in scoring
    """
    try:
        # Drop ratio check
        total_words = ocr_word_count + dropped_words
        drop_ratio  = dropped_words / total_words if total_words > 0 else 0

        # Mean confidence of retained words
        if words_df is not None and not words_df.empty and "conf" in words_df.columns:
            mean_conf = float(words_df["conf"].mean())
        else:
            mean_conf = 60.0   # assume moderate if no data

        # Penalise for high drop ratio
        # >50% dropped = significant OCR degradation
        if drop_ratio > 0.50:
            mean_conf *= 0.75
            print(f"[ConfidenceEngine] High word drop ratio: "
                  f"{drop_ratio:.0%} — OCR quality penalised.")

        # Map to quality level
        if mean_conf >= 70:
            return "good", OCR_QUALITY_LEVELS["good"][1]
        elif mean_conf >= 50:
            return "moderate", OCR_QUALITY_LEVELS["moderate"][1]
        else:
            return "poor", OCR_QUALITY_LEVELS["poor"][1]

    except Exception:
        return "moderate", 0.70


# ─────────────────────────────────────────────
# RISK LEVEL
# ─────────────────────────────────────────────

def _score_to_risk(score: float) -> str:
    """Map unified score (0.0-1.0) to risk level string."""
    for threshold, level in RISK_THRESHOLDS:
        if score >= threshold:
            return level
    return "REVIEW"


def _compute_overall_risk(scored_matches: list) -> str:
    """
    Overall risk = highest risk level across all findings.

    Risk priority order:
        CRITICAL > HIGH > MEDIUM > LOW > REVIEW > NONE
    """
    priority = {
        "CRITICAL": 5,
        "HIGH"    : 4,
        "MEDIUM"  : 3,
        "LOW"     : 2,
        "REVIEW"  : 1,
        "NONE"    : 0,
    }

    if not scored_matches:
        return "NONE"

    return max(
        (m.get("risk_level", "NONE") for m in scored_matches),
        key=lambda r: priority.get(r, 0)
    )


def _build_score_summary(scored_matches: list) -> dict:
    """Count findings per risk level for report summary."""
    summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "REVIEW": 0}
    for m in scored_matches:
        level = m.get("risk_level", "REVIEW")
        summary[level] = summary.get(level, 0) + 1
    # Remove zero-count levels for cleaner output
    return {k: v for k, v in summary.items() if v > 0}


# ─────────────────────────────────────────────
# UTILITY — for main.py console display
# ─────────────────────────────────────────────

def risk_level_icon(risk_level: str) -> str:
    """Return an emoji icon for a risk level — used in console output."""
    return {
        "CRITICAL": "🔴",
        "HIGH"    : "🟠",
        "MEDIUM"  : "🟡",
        "LOW"     : "🟢",
        "REVIEW"  : "⚪",
        "NONE"    : "✅",
    }.get(risk_level, "⚪")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from modules.sensitive_detector import detect_sensitive
    from modules.nlp_classifier     import classify

    # Simulate OCR words DataFrame
    sample_words = pd.DataFrame([
        {"word": "sample@gmail.com", "left": 10, "top": 50,
         "width": 120, "height": 14, "conf": 94.2},
        {"word": "ABCDE1234F",       "left": 10, "top": 80,
         "width": 80,  "height": 14, "conf": 97.1},
        {"word": "12/04/1990",       "left": 10, "top": 110,
         "width": 70,  "height": 14, "conf": 88.5},
        {"word": "J8369854",         "left": 10, "top": 140,
         "width": 60,  "height": 14, "conf": 91.3},
        {"word": "+91",              "left": 10, "top": 170,
         "width": 25,  "height": 14, "conf": 85.0},
        {"word": "99999",            "left": 40, "top": 170,
         "width": 40,  "height": 14, "conf": 93.0},
    ])

    print("═" * 60)
    print("  TEST 1: Passport text with Date of Birth context")
    print("═" * 60)
    text1   = (
        "CONFIDENTIAL Passport Holder RAMADUGULA SITA "
        "Date of Birth 12/04/1990 Passport No J8369854 "
        "Email sample@gmail.com PAN ABCDE1234F "
        "Phone +91 99999"
    )
    regex1  = detect_sensitive(text1)
    nlp1    = classify(text1, regex1, run_ner=True, run_doc_labels=True)
    result1 = score_findings(nlp1, sample_words,
                             ocr_word_count=43, dropped_words=15)

    print(f"\n  Overall risk : {result1['overall_risk']}")
    print(f"  OCR quality  : {result1['ocr_quality']}")
    print(f"  Score dist   : {result1['score_summary']}")
    print(f"\n  {'Type':<20} {'Score':>6}  {'Risk':<10}  {'FP?'}")
    print(f"  {'─'*20} {'─'*6}  {'─'*10}  {'─'*5}")
    for m in result1["matches"]:
        fp  = "⚠ YES" if m.get("fp_risk") else "no"
        icon = risk_level_icon(m["risk_level"])
        print(f"  {m['type']:<20} {m['unified_score']:>6.3f}  "
              f"{icon} {m['risk_level']:<8}  {fp}")

    print("\n═" * 60)
    print("  TEST 2: Invoice date — should score LOW / REVIEW")
    print("═" * 60)
    text2   = "Invoice date 12/04/1990 Order ref 45231 Receipt"
    regex2  = detect_sensitive(text2)
    nlp2    = classify(text2, regex2, run_ner=False, run_doc_labels=False)
    result2 = score_findings(nlp2, pd.DataFrame(),
                             ocr_word_count=8, dropped_words=2)
    for m in result2["matches"]:
        icon = risk_level_icon(m["risk_level"])
        fp   = "⚠ YES" if m.get("fp_risk") else "no"
        print(f"  {m['type']:<20} score={m['unified_score']:.3f}  "
              f"{icon} {m['risk_level']}  FP={fp}")

    print("\n═" * 60)
    print("  TEST 3: Poor OCR quality simulation")
    print("═" * 60)
    text3   = "ABCDE1234F sample@gmail.com"
    regex3  = detect_sensitive(text3)
    nlp3    = classify(text3, regex3, run_ner=False, run_doc_labels=False)
    # Simulate poor OCR: many dropped words
    result3 = score_findings(nlp3, pd.DataFrame(),
                             ocr_word_count=5, dropped_words=30)
    print(f"  OCR quality: {result3['ocr_quality']}")
    for m in result3["matches"]:
        print(f"  {m['type']:<20} score={m['unified_score']:.3f}  "
              f"{m['risk_level']}")
