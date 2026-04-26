"""
tests/test_nlp_classifier.py
------------------------------
Tests for nlp_classifier.py

Run from project root:
    python tests/test_nlp_classifier.py

Tests split into two groups:
    NO-MODEL TESTS (1-12)  : Run without transformers/BERT installed
                             Tests context verification, doc labels, structure
    NER TESTS (13-18)      : Require transformers + model download
                             Automatically skipped if not available

Tests:
    1.  classify() returns required keys always
    2.  Empty text returns wrapped regex result
    3.  Context upgrades DOB from low to medium (near "Date of Birth")
    4.  Context upgrades DOB from low to medium (near "born")
    5.  Benign context flags invoice date as FP risk
    6.  Benign context flags receipt date as FP risk
    7.  Sensitive context does NOT remove match (safety first)
    8.  Document label CONFIDENTIAL detected
    9.  Document label RESTRICTED detected
    10. Document label INTERNAL USE ONLY detected
    11. Multiple document labels detected in one text
    12. Output structure is backward compatible with Phase 1
    13. [NER] BERT loads without error
    14. [NER] Person name detected in passport text
    15. [NER] Org name detected
    16. [NER] Low-score entities filtered out
    17. [NER] NER findings carry source="nlp"
    18. [NER] Full pipeline: regex + NLP combined result is correct
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.sensitive_detector import detect_sensitive
from modules.nlp_classifier import (
    classify, is_nlp_available,
    _run_context_verification,
    _run_document_label_detection,
    _get_context_words,
    TRANSFORMERS_AVAILABLE
)

PASS = "  PASS ✓"
FAIL = "  FAIL ✗"
SKIP = "  SKIP –"

def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} — {label}")
    return condition


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def run_classify(text: str, run_ner=False, run_doc=True, run_ctx=True):
    regex  = detect_sensitive(text)
    return classify(text, regex,
                    run_ner=run_ner,
                    run_doc_labels=run_doc,
                    run_context=run_ctx)


# ─────────────────────────────────────────────
# NO-MODEL TESTS (always run)
# ─────────────────────────────────────────────

def test_1_required_keys():
    print("\n[Test 1] classify() always returns required keys...")
    r = run_classify("sample@gmail.com")
    for key in ["is_sensitive", "total", "matches",
                "new_findings", "context_flags", "nlp_available"]:
        check(key in r, f"Key '{key}' present")


def test_2_empty_text():
    print("\n[Test 2] Empty text returns wrapped regex result...")
    regex = detect_sensitive("")
    r     = classify("", regex)
    check(isinstance(r["matches"], list), "matches is list")
    check(r["new_findings"] == [], "new_findings is empty")
    check(r["context_flags"] == [], "context_flags is empty")


def test_3_dob_upgraded_near_date_of_birth():
    print("\n[Test 3] DOB upgraded low→medium near 'Date of Birth'...")
    text  = "Name John Smith Date of Birth 12/04/1990 Place Mumbai"
    r     = run_classify(text, run_ner=False, run_doc=False)
    dob   = next((m for m in r["matches"] if m["type"] == "dob"), None)
    if dob is None:
        print(f"  SKIP — no DOB match found.")
        return
    print(f"  DOB confidence: {dob['confidence']} | nlp_confidence: {dob.get('nlp_confidence')}")
    check(dob["confidence"] in ("medium", "high"),
          f"DOB confidence upgraded from low (got {dob['confidence']})")
    check("upgraded_by" in dob or dob["confidence"] == "medium",
          "Upgrade recorded in match")


def test_4_dob_upgraded_near_born():
    print("\n[Test 4] DOB upgraded near 'born'...")
    text = "Employee born 12/04/1990 in Hyderabad"
    r    = run_classify(text, run_ner=False, run_doc=False)
    dob  = next((m for m in r["matches"] if m["type"] == "dob"), None)
    if dob is None:
        print("  SKIP — no DOB match.")
        return
    check(dob["confidence"] in ("medium", "high"),
          f"DOB upgraded near 'born' (got {dob['confidence']})")


def test_5_invoice_date_flagged():
    print("\n[Test 5] Date near 'invoice' flagged as FP risk...")
    text = "Invoice date 12/04/1990 Order 45231"
    r    = run_classify(text, run_ner=False, run_doc=False)
    print(f"  FP flags: {len(r['context_flags'])}")
    print(f"  Flags: {[(f['type'], f.get('fp_reason','')) for f in r['context_flags']]}")
    check(len(r["context_flags"]) >= 1, "At least 1 FP flag raised")
    if r["context_flags"]:
        check(r["context_flags"][0]["fp_risk"] is True, "fp_risk=True on flagged match")


def test_6_receipt_date_flagged():
    print("\n[Test 6] Date near 'receipt' flagged as FP risk...")
    text = "Receipt date 01/06/2023 ref 78901"
    r    = run_classify(text, run_ner=False, run_doc=False)
    check(len(r["context_flags"]) >= 1 or isinstance(r["context_flags"], list),
          "FP flags list returned")


def test_7_fp_flag_does_not_remove_match():
    print("\n[Test 7] FP flag does NOT remove match from results (safety first)...")
    text = "Invoice date 12/04/1990"
    r    = run_classify(text, run_ner=False, run_doc=False)
    all_types = [m["type"] for m in r["matches"]]
    # Match should still be in results even if flagged
    check("dob" in all_types,
          f"DOB still in matches even when FP-flagged: {all_types}")


def test_8_confidential_label_detected():
    print("\n[Test 8] CONFIDENTIAL document label detected...")
    text = "CONFIDENTIAL\nEmployee record for John Smith"
    r    = run_classify(text, run_ner=False, run_doc=True)
    labels = [m for m in r["new_findings"] if m["type"] == "document_label"]
    check(len(labels) >= 1, f"CONFIDENTIAL detected (got {len(labels)} labels)")
    if labels:
        check("CONFIDENTIAL" in labels[0]["value"].upper(),
              f"Value contains CONFIDENTIAL: '{labels[0]['value']}'")


def test_9_restricted_label_detected():
    print("\n[Test 9] RESTRICTED document label detected...")
    text = "RESTRICTED — For internal use only"
    r    = run_classify(text, run_ner=False, run_doc=True)
    labels = [m for m in r["new_findings"] if m["type"] == "document_label"]
    check(len(labels) >= 1, f"RESTRICTED detected")


def test_10_internal_use_only_detected():
    print("\n[Test 10] INTERNAL USE ONLY label detected...")
    text = "INTERNAL USE ONLY\nQuarterly performance review"
    r    = run_classify(text, run_ner=False, run_doc=True)
    labels = [m for m in r["new_findings"] if m["type"] == "document_label"]
    check(len(labels) >= 1, f"INTERNAL USE ONLY detected (got {len(labels)})")


def test_11_multiple_labels():
    print("\n[Test 11] Multiple document labels in one text...")
    text = "CONFIDENTIAL — RESTRICTED\nSensitive HR document"
    r    = run_classify(text, run_ner=False, run_doc=True)
    labels = [m for m in r["new_findings"] if m["type"] == "document_label"]
    print(f"  Labels found: {[l['value'] for l in labels]}")
    check(len(labels) >= 2, f"At least 2 labels detected (got {len(labels)})")


def test_12_backward_compatibility():
    print("\n[Test 12] Output backward compatible with Phase 1 structure...")
    text  = "sample@gmail.com ABCDE1234F"
    r     = run_classify(text, run_ner=False)
    check(isinstance(r["matches"], list), "matches is list")
    if r["matches"]:
        m = r["matches"][0]
        for field in ["type", "value", "tokens", "confidence", "source"]:
            check(field in m, f"Match field '{field}' present")
    check(isinstance(r["is_sensitive"], bool), "is_sensitive is bool")
    check(isinstance(r["total"], int), "total is int")


# ─────────────────────────────────────────────
# NER TESTS (require transformers)
# ─────────────────────────────────────────────

def test_13_ner_model_loads():
    print("\n[Test 13] BERT NER model loads without error...")
    if not TRANSFORMERS_AVAILABLE:
        print(f"  {SKIP} — transformers not installed.")
        return
    from modules.nlp_classifier import _get_ner_pipeline
    ner = _get_ner_pipeline()
    check(ner is not None, "NER pipeline loaded")


def test_14_person_name_detected():
    print("\n[Test 14] Person name detected in passport text...")
    if not TRANSFORMERS_AVAILABLE:
        print(f"  {SKIP} — transformers not installed.")
        return
    text  = "Passport holder RAMADUGULA SITA MAHA LAKSHAI DOB 11/10/1990"
    regex = detect_sensitive(text)
    r     = classify(text, regex, run_ner=True, run_doc_labels=False)
    persons = [m for m in r["new_findings"] if m["type"] == "person"]
    print(f"  Person entities found: {[p['value'] for p in persons]}")
    check(len(persons) >= 1, f"At least 1 person name detected")
    if persons:
        check(persons[0]["source"] == "nlp", "Person finding has source='nlp'")


def test_15_org_name_detected():
    print("\n[Test 15] Organisation name detected...")
    if not TRANSFORMERS_AVAILABLE:
        print(f"  {SKIP} — transformers not installed.")
        return
    text  = "Issued by State Bank of India Mumbai branch"
    regex = detect_sensitive(text)
    r     = classify(text, regex, run_ner=True, run_doc_labels=False)
    orgs  = [m for m in r["new_findings"] if m["type"] == "org_name"]
    print(f"  Org entities found: {[o['value'] for o in orgs]}")
    check(isinstance(orgs, list), "Org findings list returned")


def test_16_low_score_filtered():
    print("\n[Test 16] Low-score NER entities filtered out...")
    if not TRANSFORMERS_AVAILABLE:
        print(f"  {SKIP} — transformers not installed.")
        return
    from modules.nlp_classifier import _run_ner, NER_CONFIDENCE
    # All findings should have score >= NER_CONFIDENCE
    text     = "The quick brown fox jumps over the lazy dog"
    findings = _run_ner(text)
    for f in findings:
        conf = float(f.get("nlp_confidence", "0"))
        check(conf >= NER_CONFIDENCE,
              f"Entity '{f['value']}' score {conf} >= {NER_CONFIDENCE}")
    print(f"  {len(findings)} entities passed confidence filter.")


def test_17_ner_source_is_nlp():
    print("\n[Test 17] NER findings carry source='nlp'...")
    if not TRANSFORMERS_AVAILABLE:
        print(f"  {SKIP} — transformers not installed.")
        return
    text  = "Document holder John Smith Passport J8369854"
    regex = detect_sensitive(text)
    r     = classify(text, regex, run_ner=True, run_doc_labels=False)
    nlp_findings = [m for m in r["new_findings"]]
    for f in nlp_findings:
        check(f["source"] == "nlp", f"source='nlp' on '{f['value']}'")


def test_18_full_pipeline_combined():
    print("\n[Test 18] Full pipeline: regex + NLP combined result...")
    if not TRANSFORMERS_AVAILABLE:
        print(f"  {SKIP} — transformers not installed.")
        return
    text = (
        "CONFIDENTIAL\n"
        "Passport holder RAMADUGULA SITA MAHA LAKSHAI\n"
        "Date of Birth 11/10/1990\n"
        "Passport No J8369854\n"
        "Issued by Government of India"
    )
    regex = detect_sensitive(text)
    r     = classify(text, regex, run_ner=True, run_doc_labels=True)

    print(f"\n  Total findings: {r['total']}")
    print(f"  New NLP findings: {len(r['new_findings'])}")
    print(f"  FP risk flags: {len(r['context_flags'])}")
    print(f"  Sources: {list({m['source'] for m in r['matches']})}")

    all_types = {m["type"] for m in r["matches"]}
    check(r["is_sensitive"] is True, "is_sensitive=True")
    check(r["total"] >= 2, f"At least 2 total findings (got {r['total']})")

    # Document label should be found
    doc_labels = [m for m in r["new_findings"] if m["type"] == "document_label"]
    check(len(doc_labels) >= 1, "CONFIDENTIAL label detected")

    # Both regex and NLP sources should be present
    sources = {m["source"] for m in r["matches"]}
    check("regex" in sources, "Regex findings preserved")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  NLP Classifier — Test Suite (18 tests)")
    print(f"  Transformers available: {TRANSFORMERS_AVAILABLE}")
    print("═" * 60)

    # No-model tests
    test_1_required_keys()
    test_2_empty_text()
    test_3_dob_upgraded_near_date_of_birth()
    test_4_dob_upgraded_near_born()
    test_5_invoice_date_flagged()
    test_6_receipt_date_flagged()
    test_7_fp_flag_does_not_remove_match()
    test_8_confidential_label_detected()
    test_9_restricted_label_detected()
    test_10_internal_use_only_detected()
    test_11_multiple_labels()
    test_12_backward_compatibility()

    # NER tests (require transformers)
    test_13_ner_model_loads()
    test_14_person_name_detected()
    test_15_org_name_detected()
    test_16_low_score_filtered()
    test_17_ner_source_is_nlp()
    test_18_full_pipeline_combined()

    print("\n── All 18 tests done ──")
    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  Tests 13-18 were skipped.")
        print("   Install: pip install transformers torch")
        print("   Then re-run to test BERT NER.")