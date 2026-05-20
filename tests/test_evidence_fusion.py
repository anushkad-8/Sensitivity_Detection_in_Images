"""
tests/test_evidence_fusion.py
──────────────────────────────
Unit tests for the refactored _merge_vision_result evidence-fusion logic.

Tests four decision cases:
    Case 1 — OCR evidence present + vision agrees  → vision BOOSTS, not adds
    Case 2 — OCR failed, non-fallback model, high vision conf → capped MEDIUM
    Case 3 — OCR failed, heuristic fallback → capped REVIEW with fp_risk=True
    Case 4 — vision found nothing (doc_type=unknown) → result unchanged

Also tests:
    True-positive preservation: real passport with OCR evidence scores HIGH+
    False-positive scenario: no OCR + heuristic-only → REVIEW, not HIGH
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import _merge_vision_result, _score_to_risk_level


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_ocr_finding(unified_score=0.72, risk_level="HIGH", ftype="email"):
    return {
        "type"         : ftype,
        "unified_score": unified_score,
        "risk_level"   : risk_level,
        "fp_risk"      : False,
        "score_detail" : {},
    }


def _make_scored_result(matches=None, ocr_word_count=20, ocr_raw_word_count=None):
    matches = matches or []
    if ocr_raw_word_count is None:
        ocr_raw_word_count = ocr_word_count
    return {
        "matches"            : matches,
        "total"              : len(matches),
        "is_sensitive"       : bool(matches),
        "overall_risk"       : "HIGH" if matches else "NONE",
        "score_summary"      : {},
        "ocr_quality"        : "good" if ocr_word_count >= 10 else "poor",
        "_ocr_word_count"    : ocr_word_count,
        "_ocr_raw_word_count": ocr_raw_word_count,
    }


def _make_vision_result(
    doc_type="passport",
    confidence=0.72,
    ocr_failure=False,
    fallback=True,
    sensitivity="HIGH",
):
    finding = {
        "type"             : "document_type",
        "value"            : doc_type,
        "tokens"           : [],
        "confidence"       : "high" if confidence >= 0.75 else "medium",
        "source"           : "vision",
        "vision_confidence": confidence,
        "sensitivity_level": sensitivity,
        "ocr_failure_risk" : ocr_failure,
        "fp_risk"          : False,
        "vision_is_fallback": fallback,
    }
    return {
        "document_type"    : doc_type,
        "type_confidence"  : confidence,
        "sensitivity_level": sensitivity,
        "ocr_failure_risk" : ocr_failure,
        "vision_findings"  : [finding] if doc_type != "unknown" or ocr_failure else [],
        "model_used"       : "opencv_heuristics" if fallback else "clip_vit_b32_cpu",
        "fallback"         : fallback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CASE 1 — OCR evidence present, vision corroborates → BOOST only
# ─────────────────────────────────────────────────────────────────────────────

def test_case1_vision_boosts_ocr_score():
    """Vision should BOOST existing scores, not inject a new finding."""
    ocr_finding = _make_ocr_finding(unified_score=0.72, risk_level="HIGH")
    scored = _make_scored_result(matches=[ocr_finding], ocr_word_count=20)
    vision = _make_vision_result(doc_type="passport", confidence=0.72, fallback=False)

    result = _merge_vision_result(scored, vision)

    # No new independent vision finding added
    assert len(result["matches"]) == 1, \
        f"Expected 1 match (OCR only), got {len(result['matches'])}"

    # OCR score is boosted
    boosted_score = result["matches"][0]["unified_score"]
    assert boosted_score > 0.72, \
        f"Expected score > 0.72 after boost, got {boosted_score}"

    # Vision boost metadata attached
    assert "vision_boost" in result["matches"][0]["score_detail"]
    print(f"[PASS] Case 1: vision boost applied, score={boosted_score}")


def test_case1_low_vision_conf_no_boost():
    """Low-confidence vision should not boost OCR scores."""
    ocr_finding = _make_ocr_finding(unified_score=0.72, risk_level="HIGH")
    scored = _make_scored_result(matches=[ocr_finding], ocr_word_count=20)
    vision = _make_vision_result(doc_type="passport", confidence=0.40, fallback=False)

    result = _merge_vision_result(scored, vision)

    assert result["matches"][0]["unified_score"] == 0.72, \
        "Low-confidence vision should not boost score"
    print("[PASS] Case 1b: low-confidence vision → no boost")


# ─────────────────────────────────────────────────────────────────────────────
# CASE 2 — OCR failed, non-fallback model, high vision conf → capped MEDIUM
# ─────────────────────────────────────────────────────────────────────────────

def test_case2_ocr_failure_with_real_model_gets_medium():
    """Confirmed OCR failure + real vision model → MEDIUM finding, not HIGH."""
    scored = _make_scored_result(matches=[], ocr_word_count=2)
    vision = _make_vision_result(
        doc_type="passport", confidence=0.72,
        ocr_failure=True, fallback=False, sensitivity="HIGH"
    )

    result = _merge_vision_result(scored, vision)

    assert len(result["matches"]) == 1
    m = result["matches"][0]
    assert m["risk_level"] in ("MEDIUM", "LOW", "REVIEW"), \
        f"Expected ≤MEDIUM for vision-only finding, got {m['risk_level']}"
    assert m["unified_score"] <= 0.45, \
        f"Expected score ≤0.45 (MEDIUM cap), got {m['unified_score']}"
    assert m.get("vision_solo") is True
    assert m.get("fp_risk") is False
    assert result["is_sensitive"] is True   # analyst should still review
    print(f"[PASS] Case 2: OCR failure + real model → {m['risk_level']} (score={m['unified_score']})")


# ─────────────────────────────────────────────────────────────────────────────
# CASE 3 — OCR failed, heuristic fallback → REVIEW with fp_risk=True
# ─────────────────────────────────────────────────────────────────────────────

def test_case3_heuristic_only_no_ocr_gets_review():
    """The original bug scenario: heuristic-only passport detection → REVIEW, not HIGH."""
    # No filtered words, no raw words — but fallback=True means vision can't escalate
    scored = _make_scored_result(matches=[], ocr_word_count=0, ocr_raw_word_count=0)
    vision = _make_vision_result(
        doc_type="passport", confidence=0.65,
        ocr_failure=False, fallback=True, sensitivity="HIGH"
    )

    result = _merge_vision_result(scored, vision)

    assert len(result["matches"]) == 1
    m = result["matches"][0]
    assert m["risk_level"] == "REVIEW", \
        f"Heuristic-only vision should be REVIEW, got {m['risk_level']}"
    assert m["fp_risk"] is True, \
        "Heuristic-only vision finding must have fp_risk=True"
    assert m["unified_score"] <= 0.28, \
        f"Score should be capped at 0.28, got {m['unified_score']}"
    assert result["is_sensitive"] is False, \
        "No OCR + heuristic vision → is_sensitive must be False (not a confirmed alert)"
    assert result["overall_risk"] == "NONE", \
        f"Expected NONE, got {result['overall_risk']}"
    print(f"[PASS] Case 3 (the bug): heuristic passport → REVIEW fp_risk=True (was HIGH before fix)")


def test_case3_matches_are_still_stored_for_audit():
    """REVIEW findings should still be in matches list for audit trail."""
    scored = _make_scored_result(matches=[], ocr_word_count=0, ocr_raw_word_count=0)
    vision = _make_vision_result(doc_type="id_card", confidence=0.60, fallback=True)

    result = _merge_vision_result(scored, vision)

    # Match exists in list even though is_sensitive=False
    assert len(result["matches"]) == 1, "REVIEW finding should still be in matches"
    assert result["matches"][0]["fp_risk"] is True
    print("[PASS] Case 3b: REVIEW findings stored for audit trail")


# ─────────────────────────────────────────────────────────────────────────────
# CASE 4 — vision found nothing → result unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_case4_unknown_doc_type_no_ocr_failure_unchanged():
    """When vision finds nothing, the result must be completely unmodified."""
    ocr_finding = _make_ocr_finding(unified_score=0.55, risk_level="MEDIUM")
    scored = _make_scored_result(matches=[ocr_finding], ocr_word_count=15)
    original_overall = scored["overall_risk"]
    vision = _make_vision_result(doc_type="unknown", confidence=0.20, ocr_failure=False)
    # Override: no findings emitted for "unknown" without ocr_failure
    vision["vision_findings"] = []

    result = _merge_vision_result(scored, vision)

    assert len(result["matches"]) == 1
    assert result["matches"][0]["unified_score"] == 0.55, "Score must not change"
    print("[PASS] Case 4: unknown vision result → result unchanged")


# ─────────────────────────────────────────────────────────────────────────────
# TRUE POSITIVE PRESERVATION
# ─────────────────────────────────────────────────────────────────────────────

def test_true_positive_real_passport_stays_high():
    """
    A genuine passport scan with strong OCR evidence should still score HIGH or CRITICAL.
    Vision boost must help — not hurt — the true positive case.
    """
    ocr_finding = _make_ocr_finding(unified_score=0.78, risk_level="HIGH", ftype="passport_no")
    scored = _make_scored_result(matches=[ocr_finding], ocr_word_count=30)
    vision = _make_vision_result(
        doc_type="passport", confidence=0.75,
        ocr_failure=False, fallback=False, sensitivity="HIGH"
    )

    result = _merge_vision_result(scored, vision)

    final_score = result["matches"][0]["unified_score"]
    final_risk  = result["matches"][0]["risk_level"]
    assert final_score >= 0.78, \
        f"True positive score should not drop: {final_score}"
    assert final_risk in ("HIGH", "CRITICAL"), \
        f"True positive must remain HIGH or CRITICAL, got {final_risk}"
    assert result["is_sensitive"] is True
    print(f"[PASS] True positive: passport OCR+vision → {final_risk} (score={final_score})")


def test_true_positive_blurry_id_ocr_failure_flagged():
    """
    Blurry ID card where OCR completely failed + real model + confirmed ocr_failure_risk
    → should still surface as MEDIUM (analyst review), not silently pass.
    """
    scored = _make_scored_result(matches=[], ocr_word_count=1)
    vision = _make_vision_result(
        doc_type="id_card", confidence=0.68,
        ocr_failure=True, fallback=False, sensitivity="HIGH"
    )

    result = _merge_vision_result(scored, vision)

    assert len(result["matches"]) == 1
    m = result["matches"][0]
    assert m["risk_level"] in ("MEDIUM", "LOW"), \
        f"Blurry ID with OCR failure should be MEDIUM or LOW for review, got {m['risk_level']}"
    assert m.get("vision_evidence_gate") == "ocr_failure_confirmed"
    print(f"[PASS] True positive: blurry ID OCR failure → {m['risk_level']} (not silently passed)")


# ─────────────────────────────────────────────────────────────────────────────
# RISK LEVEL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def test_case3_noisy_ocr_non_sensitive_image():
    """
    The exact scenario from test10_nonsensitive2.jpg:
    - OCR found 23 raw words but kept only 2 after confidence filtering (91% drop)
    - Vision heuristic says passport + ocr_failure_risk=True
    - No OCR/NLP evidence at all
    - Result: should be REVIEW / is_sensitive=False, NOT LOW/SENSITIVE
    
    This is a NOISY OCR case (Tesseract found text, just low quality)
    not a TRUE OCR failure (Tesseract found no text on a structured doc).
    """
    # 2 filtered words, but 23 raw words — Tesseract DID find text, just noisy
    scored = _make_scored_result(matches=[], ocr_word_count=2, ocr_raw_word_count=23)
    vision = _make_vision_result(
        doc_type="passport", confidence=0.77,
        ocr_failure=True, fallback=True, sensitivity="HIGH"
    )

    result = _merge_vision_result(scored, vision)

    assert result["is_sensitive"] is False, \
        f"Noisy OCR non-sensitive image must be is_sensitive=False, got {result['is_sensitive']}"
    assert result["overall_risk"] == "NONE", \
        f"Expected NONE overall risk, got {result['overall_risk']}"
    # Finding may still be in the list as REVIEW for audit
    if result["matches"]:
        assert result["matches"][0]["risk_level"] == "REVIEW"
        assert result["matches"][0]["fp_risk"] is True
    print("[PASS] Noisy OCR non-sensitive: 2/23 filtered words → REVIEW/NONE (not LOW/SENSITIVE)")


def test_true_ocr_failure_structured_doc_still_flags():
    """
    True OCR failure: Tesseract found 0-2 raw words on an image that looks
    like a structured document (e.g. blurry ID card photo). With a real
    vision model (not heuristic), this should still surface as a capped finding.
    """
    # 1 filtered word, 2 raw words — Tesseract genuinely found almost nothing
    scored = _make_scored_result(matches=[], ocr_word_count=1, ocr_raw_word_count=2)
    vision = _make_vision_result(
        doc_type="id_card", confidence=0.65,
        ocr_failure=True, fallback=False, sensitivity="HIGH"
    )

    result = _merge_vision_result(scored, vision)

    assert len(result["matches"]) == 1
    m = result["matches"][0]
    # Should be LOW or MEDIUM — not silently passing, not raising false HIGH
    assert m["risk_level"] in ("LOW", "MEDIUM"), \
        f"True OCR failure on structured doc should be LOW-MEDIUM, got {m['risk_level']}"
    assert m["unified_score"] <= 0.45
    print(f"[PASS] True OCR failure on structured doc → {m['risk_level']} (flagged for review)")


# ─────────────────────────────────────────────────────────────────────────────
# OCR ENGINE CRASH FIX — non-string text column
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_filter_words_handles_non_string_text():
    """
    Regression: Tesseract returns int/float in 'text' column when output has
    only 1 layout row. This crashed with:
        AttributeError: Can only use .str accessor with string values!
    """
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import pandas as pd
    from modules.ocr_engine import _filter_words

    # Simulate Tesseract returning numeric text values (the crash scenario)
    df = pd.DataFrame([
        {"text": 5,      "left": 0, "top": 0, "width": 10, "height": 10, "conf": 95},
        {"text": None,   "left": 0, "top": 0, "width": 10, "height": 10, "conf": 90},
        {"text": "hello","left": 0, "top": 0, "width": 10, "height": 10, "conf": 85},
        {"text": "",     "left": 0, "top": 0, "width": 10, "height": 10, "conf": 70},
    ])

    try:
        result = _filter_words(df, confidence_threshold=60)
        # "hello" and "5" (coerced from int) should survive; None and "" are dropped
        # The key assertion is no crash occurred — the crash was the bug
        assert len(result) >= 1, f"Expected at least 1 word, got {len(result)}"
        words = list(result["word"])
        assert "hello" in words, f"'hello' should be in results: {words}"
        # int 5 coerces to "5" — that's acceptable, not a crash
        print(f"[PASS] OCR _filter_words: handles non-string text column without crash. words={words}")
    except AttributeError as e:
        raise AssertionError(f"Still crashing with AttributeError: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RISK LEVEL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def test_score_to_risk_level_thresholds():
    assert _score_to_risk_level(0.90) == "CRITICAL"
    assert _score_to_risk_level(0.75) == "HIGH"
    assert _score_to_risk_level(0.55) == "MEDIUM"
    assert _score_to_risk_level(0.35) == "LOW"
    assert _score_to_risk_level(0.10) == "REVIEW"
    print("[PASS] _score_to_risk_level thresholds correct")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_case1_vision_boosts_ocr_score,
        test_case1_low_vision_conf_no_boost,
        test_case2_ocr_failure_with_real_model_gets_medium,
        test_case3_heuristic_only_no_ocr_gets_review,
        test_case3_matches_are_still_stored_for_audit,
        test_case3_noisy_ocr_non_sensitive_image,
        test_case4_unknown_doc_type_no_ocr_failure_unchanged,
        test_true_positive_real_passport_stays_high,
        test_true_positive_blurry_id_ocr_failure_flagged,
        test_true_ocr_failure_structured_doc_still_flags,
        test_ocr_filter_words_handles_non_string_text,
        test_score_to_risk_level_thresholds,
    ]

    passed = 0
    failed = 0
    print("\n" + "═" * 60)
    print("  Evidence-Fusion Logic — Test Suite")
    print("═" * 60)
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")
            failed += 1

    print("─" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("═" * 60)
    sys.exit(0 if failed == 0 else 1)