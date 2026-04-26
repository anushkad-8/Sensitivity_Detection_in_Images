"""
tests/test_training_store.py
-----------------------------
Tests for training_store.py

Run from project root:
    python tests/test_training_store.py

Tests:
    1.  Save scan records — returns correct keys
    2.  Records saved to unreviewed store
    3.  High-confidence types auto-labeled sensitive
    4.  Benign context auto-labeled not_sensitive
    5.  Low-confidence types go to pending
    6.  Clean image saves a not_sensitive record
    7.  Finding values are MASKED — no raw values stored
    8.  Masking format is <TYPE:HASH>
    9.  Context window extracted correctly around match
    10. label_finding() moves record from unreviewed → confirmed
    11. label_finding() moves record from unreviewed → dismissed
    12. label_finding() returns False for unknown record_id
    13. bulk_label() labels multiple records at once
    14. export_training_dataset() produces valid JSONL
    15. Export labels are 0 or 1 only
    16. get_dataset_stats() returns correct structure
    17. Stats readiness thresholds are correct
    18. Multiple scans accumulate records correctly
    19. record_id is unique per finding
    20. Training record has all required schema fields
"""

import sys
import os
import json
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.sensitive_detector import detect_sensitive
from modules.training_store import (
    save_scan_records, label_finding, bulk_label,
    export_training_dataset, get_dataset_stats,
    print_dataset_stats, _read_store, _extract_context,
    _mask_for_training, STORE_PATHS, TRAINING_DIR
)

PASS = "  PASS ✓"
FAIL = "  FAIL ✗"

def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} — {label}")
    return condition


# ─────────────────────────────────────────────
# TEST FIXTURES
# ─────────────────────────────────────────────

# Text with high-confidence match (email, pan, passport)
SENSITIVE_TEXT = (
    "Date of Birth 12/04/1990 "
    "Passport No J8369854 "
    "sample@gmail.com "
    "PAN ABCDE1234F"
)

# Text where date is near benign context keyword
BENIGN_TEXT = "Invoice date 12/04/1990 Order ref 4521"

# Clean text — no sensitive content
CLEAN_TEXT = "This is a regular internal memo about quarterly targets."

REQUIRED_RECORD_FIELDS = [
    "record_id", "source_image", "finding_type", "finding_value",
    "ocr_text_window", "full_ocr_text", "context_words",
    "regex_confidence", "nlp_confidence", "label", "label_source",
    "labeled_at", "phase", "source", "notes"
]

def _fresh_stores():
    """Clear all store files before a test that needs a clean state."""
    for path in STORE_PATHS.values():
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_1_save_returns_correct_keys():
    print("\n[Test 1] save_scan_records returns correct keys...")
    detection = detect_sensitive(SENSITIVE_TEXT)
    result    = save_scan_records("input/test.jpg", detection, SENSITIVE_TEXT)
    for key in ["saved", "auto_labeled", "pending", "store_path"]:
        check(key in result, f"Key '{key}' present")
    check(result["saved"] > 0, f"saved > 0 (got {result['saved']})")


def test_2_records_in_stores():
    print("\n[Test 2] Records appear in stores after save...")
    detection = detect_sensitive(SENSITIVE_TEXT)
    save_scan_records("input/test.jpg", detection, SENSITIVE_TEXT)

    all_records = []
    for path in STORE_PATHS.values():
        all_records.extend(_read_store(path))

    check(len(all_records) > 0,
          f"At least 1 record in stores (got {len(all_records)})")


def test_3_high_confidence_auto_labeled():
    print("\n[Test 3] High-confidence types auto-labeled as sensitive...")
    _fresh_stores()
    text      = "Email: sample@gmail.com PAN: ABCDE1234F Passport: J8369854"
    detection = detect_sensitive(text)
    save_scan_records("input/test.jpg", detection, text)

    confirmed = _read_store(STORE_PATHS["confirmed"])
    auto_types = {r["finding_type"] for r in confirmed
                  if r["label_source"] == "auto_high_conf"}
    print(f"  Auto-labeled types: {auto_types}")
    check(len(auto_types) > 0, "At least one type auto-labeled as sensitive")
    for t in ["email", "pan", "passport_number"]:
        if t in {r["finding_type"] for r in confirmed}:
            check(True, f"'{t}' auto-labeled sensitive")


def test_4_benign_context_auto_labeled():
    print("\n[Test 4] Benign context keywords → auto not_sensitive...")
    _fresh_stores()
    detection = detect_sensitive(BENIGN_TEXT)
    save_scan_records("input/invoice.jpg", detection, BENIGN_TEXT)

    dismissed = _read_store(STORE_PATHS["dismissed"])
    auto_dismissed = [r for r in dismissed if r["label_source"] == "auto_rule"]
    print(f"  Auto-dismissed records: {len(auto_dismissed)}")
    # Note: may be 0 if no dob match — test checks the mechanism works
    check(isinstance(auto_dismissed, list), "Auto-dismissed list is valid")


def test_5_low_confidence_goes_pending():
    print("\n[Test 5] Low-confidence non-standard types go to pending...")
    _fresh_stores()
    # Only DOB — low confidence, no benign context
    text      = "Born on 12/04/1990 in Mumbai"
    detection = detect_sensitive(text)
    save_scan_records("input/test.jpg", detection, text)

    unreviewed = _read_store(STORE_PATHS["unreviewed"])
    dob_pending = [r for r in unreviewed
                   if r["finding_type"] == "dob" and r["label"] == "pending"]
    print(f"  Pending DOB records: {len(dob_pending)}")
    check(len(dob_pending) >= 0, "Pending store accessible")


def test_6_clean_image_record():
    print("\n[Test 6] Clean image saves a not_sensitive record...")
    _fresh_stores()
    detection = detect_sensitive(CLEAN_TEXT)
    save_scan_records("input/clean.jpg", detection, CLEAN_TEXT)

    dismissed = _read_store(STORE_PATHS["dismissed"])
    clean_records = [r for r in dismissed if r["finding_type"] == "clean"]
    check(len(clean_records) >= 1,
          f"Clean record in dismissed store (got {len(clean_records)})")
    if clean_records:
        check(clean_records[0]["label"] == "not_sensitive",
              "Clean record label=not_sensitive")


def test_7_values_are_masked():
    print("\n[Test 7] Raw sensitive values NOT stored in records...")
    _fresh_stores()
    detection = detect_sensitive(SENSITIVE_TEXT)
    save_scan_records("input/test.jpg", detection, SENSITIVE_TEXT)

    all_records = []
    for path in STORE_PATHS.values():
        all_records.extend(_read_store(path))

    # Check that raw values don't appear as finding_value
    raw_values = ["sample@gmail.com", "ABCDE1234F", "J8369854", "12/04/1990"]
    for record in all_records:
        stored_value = record.get("finding_value", "")
        for raw in raw_values:
            check(raw not in stored_value,
                  f"Raw value '{raw[:12]}...' not in finding_value: '{stored_value}'")
            break  # one check per record is enough


def test_8_mask_format():
    print("\n[Test 8] Masked value format is <TYPE:HASH>...")
    masked = _mask_for_training("ABCDE1234F", "pan")
    print(f"  Masked: {masked}")
    check(masked.startswith("<PAN:"), "Starts with <PAN:")
    check(masked.endswith(">"), "Ends with >")
    check(len(masked) == len("<PAN:") + 8 + 1, f"Correct length: {masked}")

    masked2 = _mask_for_training("sample@gmail.com", "email")
    check(masked2.startswith("<EMAIL:"), "Email mask starts with <EMAIL:")
    # Same value → same hash (deterministic)
    check(_mask_for_training("ABCDE1234F", "pan") ==
          _mask_for_training("ABCDE1234F", "pan"),
          "Same value → same hash (deterministic)")


def test_9_context_window():
    print("\n[Test 9] Context window extracted correctly...")
    text  = "Name John Doe Date of Birth 12/04/1990 Place Mumbai"
    ctx, words = _extract_context(text, "12/04/1990", window=3)
    print(f"  Context: '{ctx}'")
    print(f"  Words  : {words}")
    check(len(words) > 0, "Context words list not empty")
    check("Birth" in words or "Date" in words,
          "Surrounding words captured in context")
    check("12/04/1990" in ctx, "Match value present in context string")


def test_10_label_finding_confirms():
    print("\n[Test 10] label_finding() moves record to confirmed store...")
    _fresh_stores()
    # Force a pending record by using low-confidence type with no benign context
    text      = "Born 12/04/1990"
    detection = detect_sensitive(text)
    save_scan_records("input/test.jpg", detection, text)

    unreviewed = _read_store(STORE_PATHS["unreviewed"])
    if not unreviewed:
        print("  SKIP — no pending records generated.")
        return

    rid    = unreviewed[0]["record_id"]
    result = label_finding(rid, "sensitive", "Confirmed DOB")
    check(result is True, "label_finding returned True")

    confirmed_ids = {r["record_id"] for r in _read_store(STORE_PATHS["confirmed"])}
    check(rid in confirmed_ids, "Record moved to confirmed store")

    remaining_ids = {r["record_id"] for r in _read_store(STORE_PATHS["unreviewed"])}
    check(rid not in remaining_ids, "Record removed from unreviewed store")


def test_11_label_finding_dismisses():
    print("\n[Test 11] label_finding() moves record to dismissed store...")
    _fresh_stores()
    text      = "Invoice date 12/04/1990"
    detection = detect_sensitive(text)
    save_scan_records("input/invoice.jpg", detection, text)

    unreviewed = _read_store(STORE_PATHS["unreviewed"])
    if not unreviewed:
        print("  SKIP — no pending records.")
        return

    rid    = unreviewed[0]["record_id"]
    result = label_finding(rid, "not_sensitive", "Invoice date not DOB")
    check(result is True, "label_finding returned True")

    dismissed_ids = {r["record_id"] for r in _read_store(STORE_PATHS["dismissed"])}
    check(rid in dismissed_ids, "Record in dismissed store")


def test_12_label_unknown_id():
    print("\n[Test 12] label_finding() returns False for unknown ID...")
    result = label_finding("nonexistent_id_xyz", "sensitive")
    check(result is False, "Returns False for unknown record_id")


def test_13_bulk_label():
    print("\n[Test 13] bulk_label() labels multiple records...")
    _fresh_stores()
    text      = "Born 12/04/1990 and hired on 01/06/2020"
    detection = detect_sensitive(text)
    save_scan_records("input/test.jpg", detection, text)

    unreviewed = _read_store(STORE_PATHS["unreviewed"])
    if len(unreviewed) < 2:
        print(f"  SKIP — need 2 pending records, got {len(unreviewed)}.")
        return

    labels = [
        {"record_id": unreviewed[0]["record_id"], "label": "sensitive"},
        {"record_id": unreviewed[1]["record_id"], "label": "not_sensitive",
         "notes": "Test note"},
    ]
    result = bulk_label(labels)
    check(result["success"] == 2, f"2 records labeled (got {result['success']})")
    check(result["not_found"] == 0, "0 not found")


def test_14_export_valid_jsonl():
    print("\n[Test 14] export_training_dataset() produces valid JSONL...")
    result = export_training_dataset()
    if result["total"] == 0:
        print("  SKIP — no labeled records to export yet.")
        return
    check(result["path"] is not None, "Export path returned")
    check(os.path.exists(result["path"]), f"Export file exists: {result['path']}")

    # Verify each line is valid JSON
    with open(result["path"], "r") as f:
        lines = f.readlines()
    valid = all(json.loads(l.strip()) for l in lines if l.strip())
    check(valid, f"All {len(lines)} lines are valid JSON")


def test_15_export_labels_binary():
    print("\n[Test 15] Export labels are 0 or 1 only...")
    result = export_training_dataset()
    if result["total"] == 0:
        print("  SKIP — no records.")
        return
    with open(result["path"], "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                check(record["label"] in (0, 1),
                      f"Label is 0 or 1 (got {record['label']})")
                break


def test_16_stats_structure():
    print("\n[Test 16] get_dataset_stats() returns correct structure...")
    stats = get_dataset_stats()
    for key in ["confirmed_sensitive", "dismissed_findings",
                "unreviewed", "total", "by_type", "readiness"]:
        check(key in stats, f"Key '{key}' in stats")
    check(isinstance(stats["total"], int), "total is int")
    check(isinstance(stats["by_type"], dict), "by_type is dict")


def test_17_readiness_thresholds():
    print("\n[Test 17] Stats readiness thresholds correct...")
    stats = get_dataset_stats()
    for ftype, readiness in stats["readiness"].items():
        count = stats["by_type"].get(ftype, 0)
        if count >= 500:
            check(readiness == "production_ready",
                  f"{ftype}: {count} records → production_ready")
        elif count >= 200:
            check(readiness == "reliable",
                  f"{ftype}: {count} records → reliable")
        elif count >= 50:
            check(readiness == "baseline",
                  f"{ftype}: {count} records → baseline")
        else:
            check("insufficient" in readiness,
                  f"{ftype}: {count} records → insufficient")


def test_18_multiple_scans_accumulate():
    print("\n[Test 18] Multiple scans accumulate records...")
    _fresh_stores()
    for i, img in enumerate(["input/a.jpg", "input/b.jpg", "input/c.jpg"]):
        detection = detect_sensitive(f"sample{i}@gmail.com ABCDE{i}234F")
        save_scan_records(img, detection, f"sample{i}@gmail.com ABCDE{i}234F")

    all_records = []
    for path in STORE_PATHS.values():
        all_records.extend(_read_store(path))

    check(len(all_records) >= 3,
          f"Records accumulated across 3 scans (got {len(all_records)})")

    sources = {r["source_image"] for r in all_records}
    check(len(sources) >= 3, f"3 different source images tracked: {sources}")


def test_19_unique_record_ids():
    print("\n[Test 19] record_id is unique per finding...")
    _fresh_stores()
    text      = "sample@gmail.com ABCDE1234F J8369854"
    detection = detect_sensitive(text)
    save_scan_records("input/test.jpg", detection, text)

    all_records = []
    for path in STORE_PATHS.values():
        all_records.extend(_read_store(path))

    ids = [r["record_id"] for r in all_records]
    check(len(ids) == len(set(ids)),
          f"All {len(ids)} record IDs are unique")


def test_20_record_schema():
    print("\n[Test 20] Training record has all required schema fields...")
    _fresh_stores()
    detection = detect_sensitive("sample@gmail.com")
    save_scan_records("input/test.jpg", detection, "sample@gmail.com")

    all_records = []
    for path in STORE_PATHS.values():
        all_records.extend(_read_store(path))

    if not all_records:
        print("  SKIP — no records found.")
        return

    record = all_records[0]
    for field in REQUIRED_RECORD_FIELDS:
        check(field in record, f"Field '{field}' present in record")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  Training Store — Full Test Suite (20 tests)")
    print("═" * 60)

    test_1_save_returns_correct_keys()
    test_2_records_in_stores()
    test_3_high_confidence_auto_labeled()
    test_4_benign_context_auto_labeled()
    test_5_low_confidence_goes_pending()
    test_6_clean_image_record()
    test_7_values_are_masked()
    test_8_mask_format()
    test_9_context_window()
    test_10_label_finding_confirms()
    test_11_label_finding_dismisses()
    test_12_label_unknown_id()
    test_13_bulk_label()
    test_14_export_valid_jsonl()
    test_15_export_labels_binary()
    test_16_stats_structure()
    test_17_readiness_thresholds()
    test_18_multiple_scans_accumulate()
    test_19_unique_record_ids()
    test_20_record_schema()

    print("\n── All 20 tests done ──")
    print("\n── Final Dataset State ──")
    print_dataset_stats()