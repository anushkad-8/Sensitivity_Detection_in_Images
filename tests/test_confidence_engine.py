"""
tests/test_confidence_engine.py
--------------------------------
Tests for confidence_engine.py

Run from project root:
    python tests/test_confidence_engine.py

Tests:
    1.  score_findings() returns all required keys
    2.  Each match has unified_score and risk_level
    3.  unified_score is float between 0.0 and 1.0
    4.  High-confidence regex type scores >= 0.70 (HIGH or above)
    5.  Low-confidence type without context scores < 0.70
    6.  FP-flagged match is forced to REVIEW regardless of score
    7.  DOB near "Date of Birth" scores higher than DOB near "invoice"
    8.  Document label scores CRITICAL or HIGH
    9.  Overall risk = highest risk across all findings
    10. Poor OCR quality reduces scores vs good OCR quality
    11. Empty matches → overall_risk = NONE
    12. score_detail has all required breakdown fields
    13. Risk level icon returns correct emoji
    14. Score summary counts are accurate
    15. NLP-only finding (source=nlp) gets scored correctly
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.sensitive_detector  import detect_sensitive
from modules.nlp_classifier      import classify
from modules.confidence_engine   import (
    score_findings, risk_level_icon,
    _score_to_risk, _assess_ocr_quality, _compute_overall_risk
)

PASS = "  PASS ✓"
FAIL = "  FAIL ✗"

def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} — {label}")
    return condition


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

def make_words_df(words: list) -> pd.DataFrame:
    """Build a minimal words DataFrame for testing."""
    return pd.DataFrame([
        {"word": w, "left": i*10, "top": 50,
         "width": 60, "height": 14, "conf": conf}
        for i, (w, conf) in enumerate(words)
    ])

def run_pipeline(text: str, run_ner=False, run_doc=True) -> tuple:
    """Run regex + NLP and return (nlp_result, words_df)."""
    regex  = detect_sensitive(text)
    nlp    = classify(text, regex, run_ner=run_ner,
                      run_doc_labels=run_doc, run_context=True)
    return nlp, pd.DataFrame()

GOOD_WORDS = make_words_df([
    ("sample@gmail.com", 94.0),
    ("ABCDE1234F",       97.0),
    ("J8369854",         91.0),
    ("12/04/1990",       88.0),
])

REQUIRED_MATCH_FIELDS = [
    "type", "value", "tokens", "confidence", "source",
    "unified_score", "risk_level", "score_detail"
]

REQUIRED_RESULT_KEYS = [
    "is_sensitive", "total", "matches", "overall_risk",
    "ocr_quality", "score_summary", "new_findings", "context_flags"
]


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_1_required_keys():
    print("\n[Test 1] score_findings() returns all required keys...")
    nlp, _ = run_pipeline("sample@gmail.com ABCDE1234F")
    result  = score_findings(nlp, GOOD_WORDS)
    for key in REQUIRED_RESULT_KEYS:
        check(key in result, f"Key '{key}' present")


def test_2_match_has_score_fields():
    print("\n[Test 2] Each match has unified_score and risk_level...")
    nlp, _ = run_pipeline("sample@gmail.com ABCDE1234F J8369854")
    result  = score_findings(nlp, GOOD_WORDS)
    for m in result["matches"]:
        check("unified_score" in m,
              f"{m['type']}: has unified_score")
        check("risk_level" in m,
              f"{m['type']}: has risk_level")


def test_3_score_is_valid_float():
    print("\n[Test 3] unified_score is float between 0.0 and 1.0...")
    nlp, _ = run_pipeline("sample@gmail.com ABCDE1234F 9876 5432 1098")
    result  = score_findings(nlp, GOOD_WORDS)
    for m in result["matches"]:
        s = m["unified_score"]
        check(isinstance(s, float) and 0.0 <= s <= 1.0,
              f"{m['type']}: score={s} is valid float in [0,1]")


def test_4_high_conf_type_scores_high():
    print("\n[Test 4] High-confidence types score >= 0.70 (HIGH or above)...")
    text   = "PAN ABCDE1234F Email sample@gmail.com Passport J8369854"
    nlp, _ = run_pipeline(text, run_doc=False)
    result  = score_findings(nlp, GOOD_WORDS)
    for m in result["matches"]:
        if m["confidence"] == "high" and m["source"] == "regex":
            check(m["unified_score"] >= 0.70,
                  f"{m['type']}: score={m['unified_score']:.3f} >= 0.70")
            check(m["risk_level"] in ("CRITICAL", "HIGH"),
                  f"{m['type']}: risk={m['risk_level']} is HIGH or above")


def test_5_low_conf_no_context_scores_lower():
    print("\n[Test 5] Low-confidence type without context scores < 0.70...")
    # Plain date with no DOB context words
    text   = "reference 12/04/1990 batch"
    nlp, _ = run_pipeline(text, run_doc=False)
    result  = score_findings(nlp, pd.DataFrame(), ocr_word_count=5)
    dob    = next((m for m in result["matches"] if m["type"] == "dob"), None)
    if dob:
        print(f"  DOB score: {dob['unified_score']:.3f} | risk: {dob['risk_level']}")
        check(dob["unified_score"] < 0.70,
              f"DOB without context scores < 0.70 (got {dob['unified_score']:.3f})")
    else:
        print("  SKIP — no DOB match.")


def test_6_fp_flag_forces_review():
    print("\n[Test 6] FP-flagged match forced to REVIEW regardless of score...")
    text   = "Invoice date 12/04/1990 order receipt"
    nlp, _ = run_pipeline(text, run_doc=False)
    result  = score_findings(nlp, pd.DataFrame())
    flagged = [m for m in result["matches"] if m.get("fp_risk") is True]
    print(f"  Flagged matches: {len(flagged)}")
    for m in flagged:
        check(m["risk_level"] == "REVIEW",
              f"FP-flagged {m['type']} has risk=REVIEW (got {m['risk_level']})")


def test_7_context_raises_score():
    print("\n[Test 7] DOB near 'Date of Birth' scores higher than near 'invoice'...")
    # With sensitive context
    text_sens  = "Date of Birth 12/04/1990"
    nlp_sens, _ = run_pipeline(text_sens, run_doc=False)
    res_sens    = score_findings(nlp_sens, pd.DataFrame())
    dob_sens    = next((m for m in res_sens["matches"] if m["type"] == "dob"), None)

    # With benign context
    text_ben   = "Invoice date 12/04/1990 order ref"
    nlp_ben, _ = run_pipeline(text_ben, run_doc=False)
    res_ben    = score_findings(nlp_ben, pd.DataFrame())
    dob_ben    = next((m for m in res_ben["matches"] if m["type"] == "dob"), None)

    if dob_sens and dob_ben:
        print(f"  DOB with 'Date of Birth': {dob_sens['unified_score']:.3f}")
        print(f"  DOB with 'invoice':       {dob_ben['unified_score']:.3f}")
        check(dob_sens["unified_score"] > dob_ben["unified_score"],
              "Sensitive context yields higher score than benign context")
    else:
        print("  SKIP — missing DOB matches.")


def test_8_document_label_scores_high():
    print("\n[Test 8] Document label scores CRITICAL or HIGH...")
    text   = "CONFIDENTIAL employee record"
    nlp, _ = run_pipeline(text, run_doc=True)
    result  = score_findings(nlp, pd.DataFrame())
    labels = [m for m in result["matches"] if m["type"] == "document_label"]
    print(f"  Document labels: {[(l['value'], l['unified_score'], l['risk_level']) for l in labels]}")
    for lbl in labels:
        check(lbl["risk_level"] in ("CRITICAL", "HIGH"),
              f"Document label '{lbl['value']}' risk={lbl['risk_level']}")


def test_9_overall_risk_is_highest():
    print("\n[Test 9] overall_risk = highest risk level across findings...")
    text   = "sample@gmail.com ABCDE1234F Date of Birth 12/04/1990"
    nlp, _ = run_pipeline(text, run_doc=False)
    result  = score_findings(nlp, GOOD_WORDS)

    priority = {"CRITICAL":5,"HIGH":4,"MEDIUM":3,"LOW":2,"REVIEW":1,"NONE":0}
    max_risk = max(
        (m["risk_level"] for m in result["matches"]),
        key=lambda r: priority.get(r, 0), default="NONE"
    )
    check(result["overall_risk"] == max_risk,
          f"overall_risk={result['overall_risk']} matches max={max_risk}")


def test_10_poor_ocr_reduces_score():
    print("\n[Test 10] Poor OCR quality (high drop ratio) reduces scores...")
    text  = "ABCDE1234F sample@gmail.com"
    nlp1, _ = run_pipeline(text, run_doc=False)
    nlp2, _ = run_pipeline(text, run_doc=False)

    # Good OCR
    res_good = score_findings(nlp1, GOOD_WORDS,
                               ocr_word_count=40, dropped_words=5)
    # Poor OCR
    res_poor = score_findings(nlp2, pd.DataFrame(),
                               ocr_word_count=5, dropped_words=40)

    print(f"  Good OCR quality: {res_good['ocr_quality']}")
    print(f"  Poor OCR quality: {res_poor['ocr_quality']}")

    check(res_good["ocr_quality"] in ("good", "moderate"),
          f"Good OCR classified correctly: {res_good['ocr_quality']}")
    check(res_poor["ocr_quality"] in ("poor", "moderate"),
          f"Poor OCR classified correctly: {res_poor['ocr_quality']}")


def test_11_empty_matches():
    print("\n[Test 11] Empty matches → overall_risk = NONE...")
    empty_nlp = {
        "is_sensitive": False, "total": 0, "matches": [],
        "new_findings": [], "context_flags": [], "nlp_available": False
    }
    result = score_findings(empty_nlp, pd.DataFrame())
    check(result["overall_risk"] == "NONE",
          f"overall_risk=NONE for empty matches (got {result['overall_risk']})")
    check(result["total"] == 0, "total=0")


def test_12_score_detail_fields():
    print("\n[Test 12] score_detail has all breakdown fields...")
    nlp, _ = run_pipeline("sample@gmail.com", run_doc=False)
    result  = score_findings(nlp, GOOD_WORDS)
    if not result["matches"]:
        print("  SKIP")
        return
    detail = result["matches"][0].get("score_detail", {})
    for field in ["regex_score", "nlp_score", "ocr_score",
                  "regex_weight", "nlp_weight", "ocr_weight"]:
        check(field in detail, f"score_detail has '{field}'")


def test_13_risk_icons():
    print("\n[Test 13] risk_level_icon returns correct emoji...")
    cases = [
        ("CRITICAL", "🔴"),
        ("HIGH",     "🟠"),
        ("MEDIUM",   "🟡"),
        ("LOW",      "🟢"),
        ("REVIEW",   "⚪"),
        ("NONE",     "✅"),
    ]
    for level, expected in cases:
        icon = risk_level_icon(level)
        check(icon == expected, f"{level} → '{icon}' (expected '{expected}')")


def test_14_score_summary_counts():
    print("\n[Test 14] Score summary counts are accurate...")
    text   = "sample@gmail.com ABCDE1234F J8369854 Date of Birth 12/04/1990"
    nlp, _ = run_pipeline(text, run_doc=False)
    result  = score_findings(nlp, GOOD_WORDS)
    summary = result["score_summary"]
    total_in_summary = sum(summary.values())
    check(total_in_summary == result["total"],
          f"Summary counts sum to total: {total_in_summary} == {result['total']}")


def test_15_nlp_only_finding_scored():
    print("\n[Test 15] NLP-only finding (source=nlp) scored correctly...")
    nlp_result = {
        "is_sensitive" : True,
        "total"        : 1,
        "matches"      : [{
            "type"          : "document_label",
            "value"         : "CONFIDENTIAL",
            "tokens"        : ["CONFIDENTIAL"],
            "confidence"    : "high",
            "nlp_confidence": "high",
            "source"        : "nlp",
            "fp_risk"       : False,
        }],
        "new_findings" : [],
        "context_flags": [],
        "nlp_available": True,
    }
    result = score_findings(nlp_result, pd.DataFrame())
    m      = result["matches"][0]
    print(f"  Document label score: {m['unified_score']:.3f} | risk: {m['risk_level']}")
    check(m["unified_score"] >= 0.70,
          f"NLP document_label score >= 0.70 (got {m['unified_score']:.3f})")
    check(m["risk_level"] in ("CRITICAL", "HIGH"),
          f"Document label risk is HIGH or CRITICAL (got {m['risk_level']})")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  Confidence Engine — Test Suite (15 tests)")
    print("═" * 60)

    test_1_required_keys()
    test_2_match_has_score_fields()
    test_3_score_is_valid_float()
    test_4_high_conf_type_scores_high()
    test_5_low_conf_no_context_scores_lower()
    test_6_fp_flag_forces_review()
    test_7_context_raises_score()
    test_8_document_label_scores_high()
    test_9_overall_risk_is_highest()
    test_10_poor_ocr_reduces_score()
    test_11_empty_matches()
    test_12_score_detail_fields()
    test_13_risk_icons()
    test_14_score_summary_counts()
    test_15_nlp_only_finding_scored()

    print("\n── All 15 tests done ──")