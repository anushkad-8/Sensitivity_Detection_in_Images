"""
tests/test_reporter.py
------------------------
Tests for Module 5: reporter.py

Run from project root:
    python tests/test_reporter.py

Tests:
    1.  Report generates without crashing
    2.  Returns required keys: report_path, audit_log_path, report_id, summary
    3.  Report file exists on disk
    4.  Audit log file exists and has at least one entry
    5.  Audit log entry has no sensitive values (value_full must not appear)
    6.  Audit log entry has all required metadata fields
    7.  Report ID is 32-char hex string
    8.  Masked values hide sensitive content (no full value in mask)
    9.  Encryption works — .enc file is not readable as plain JSON
    10. Decryption recovers original findings correctly
    11. Clean image (no findings) produces correct clean report
    12. Notes generated for low-confidence findings
    13. Finding types summary is accurate
    14. read_audit_log() returns list of dicts
    15. Multiple scans produce separate report files
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.sensitive_detector import detect_sensitive
from modules.reporter import (
    generate_report, decrypt_report, read_audit_log,
    REPORTS_DIR, AUDIT_LOG, ENCRYPTION_AVAILABLE, _mask_value
)

PASS = "  PASS ✓"
FAIL = "  FAIL ✗"

def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} — {label}")
    return condition


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

SAMPLE_TEXT_SENSITIVE = (
    "sample@gmail.com +91 99999 99999 "
    "ABCDE1234F 9876 5432 1098 "
    "DOB 12/04/1990 J8369854"
)

SAMPLE_TEXT_CLEAN = "Hello, this is a regular document with no sensitive info."

FAKE_IMAGE_PATH     = "input/test_reporter_sample.jpg"
FAKE_ANNOTATED_PATH = "output/test_reporter_annotated.jpg"


def make_detection(text: str) -> dict:
    return detect_sensitive(text)


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_1_generates_without_crash():
    print("\n[Test 1] Report generates without crashing...")
    try:
        d = make_detection(SAMPLE_TEXT_SENSITIVE)
        r = generate_report(FAKE_IMAGE_PATH, d, ocr_word_count=79,
                            annotated_path=FAKE_ANNOTATED_PATH, encrypt=True)
        print(f"  PASS ✓ — Report generated. ID: {r['report_id'][:12]}...")
        return r
    except Exception as e:
        print(f"  FAIL ✗ — Exception: {e}")
        return None


def test_2_return_keys(result: dict):
    print("\n[Test 2] Return dict has required keys...")
    if result is None:
        print("  SKIP — no result.")
        return
    for key in ["report_path", "audit_log_path", "report_id", "summary"]:
        check(key in result, f"Key '{key}' present")


def test_3_report_file_exists(result: dict):
    print("\n[Test 3] Report file exists on disk...")
    if result is None:
        print("  SKIP")
        return
    check(os.path.exists(result["report_path"]),
          f"File exists: {result['report_path']}")


def test_4_audit_log_exists():
    print("\n[Test 4] Audit log exists with at least one entry...")
    check(os.path.exists(AUDIT_LOG), f"Audit log exists: {AUDIT_LOG}")
    entries = read_audit_log()
    check(len(entries) >= 1, f"At least 1 entry in audit log (got {len(entries)})")


def test_5_audit_log_no_sensitive_values():
    print("\n[Test 5] Audit log contains NO sensitive values...")
    entries = read_audit_log()
    if not entries:
        print("  SKIP — no audit entries.")
        return
    last = entries[-1]
    # "finding_types" should only have type names and counts, no values
    entry_str = json.dumps(last)
    # These would be present if values leaked into the audit log
    leaked = any(bad in entry_str for bad in [
        "sample@gmail.com", "99999 99999", "ABCDE1234F", "9876 5432"
    ])
    check(not leaked, "No sensitive values found in audit log entry")
    check("finding_types" in last, "'finding_types' key present (type counts only)")


def test_6_audit_log_fields():
    print("\n[Test 6] Audit log entry has all required metadata fields...")
    entries = read_audit_log()
    if not entries:
        print("  SKIP")
        return
    entry = entries[-1]
    for field in ["report_id", "scanned_at", "image_file", "is_sensitive",
                  "total_findings", "finding_types", "ocr_word_count",
                  "report_file", "phase"]:
        check(field in entry, f"Field '{field}' in audit entry")


def test_7_report_id_format(result: dict):
    print("\n[Test 7] Report ID is 32-char hex string...")
    if result is None:
        print("  SKIP")
        return
    rid = result["report_id"]
    check(len(rid) == 32, f"Length is 32 (got {len(rid)})")
    check(all(c in "0123456789abcdef" for c in rid), "All chars are hex")


def test_8_masking():
    print("\n[Test 8] Masked values hide sensitive content...")
    cases = [
        ("sample@gmail.com",  "email",      "sample@gmail.com"),
        ("ABCDE1234F",        "pan",         "ABCDE1234F"),
        ("9876 5432 1098",    "aadhaar",     "9876 5432 1098"),
        ("4111 1111 1111 1111","bank_card",  "4111 1111 1111 1111"),
        ("J8369854",          "passport_number", "J8369854"),
        ("+91 99999 99999",   "phone",       "+91 99999 99999"),
    ]
    for value, ptype, original in cases:
        masked = _mask_value(value, ptype)
        check(masked != original, f"{ptype}: masked != original (got '{masked}')")
        check("***" in masked, f"{ptype}: contains '***' (got '{masked}')")


def test_9_encrypted_file_not_plain_json(result: dict):
    print("\n[Test 9] Encrypted .enc file is NOT readable as plain JSON...")
    if result is None or not ENCRYPTION_AVAILABLE:
        print("  SKIP — no result or encryption unavailable.")
        return
    path = result["report_path"]
    if not path.endswith(".enc"):
        print("  SKIP — file is plain JSON (encryption disabled).")
        return
    try:
        with open(path, "rb") as f:
            raw = f.read()
        json.loads(raw.decode("utf-8"))
        print(f"  FAIL ✗ — File is readable as plain JSON! Encryption not working.")
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"  PASS ✓ — File is binary/encrypted, cannot be read as plain JSON")


def test_10_decryption_works(result: dict):
    print("\n[Test 10] Decryption recovers original findings...")
    if result is None or not ENCRYPTION_AVAILABLE:
        print("  SKIP")
        return
    path = result["report_path"]
    if not path.endswith(".enc"):
        print("  SKIP — plain JSON, decryption not applicable.")
        return
    try:
        decrypted = decrypt_report(path)
        check("findings" in decrypted, "'findings' key in decrypted report")
        check(len(decrypted["findings"]) > 0, "At least 1 finding in decrypted report")
        # Verify full value is present (not masked) in encrypted report
        values = [f["value_full"] for f in decrypted["findings"]]
        check("sample@gmail.com" in values, "Full email value recovered after decryption")
        print(f"  Decrypted {len(decrypted['findings'])} findings successfully.")
    except Exception as e:
        print(f"  FAIL ✗ — Decryption error: {e}")


def test_11_clean_image_report():
    print("\n[Test 11] Clean image (no findings) produces correct report...")
    d = make_detection(SAMPLE_TEXT_CLEAN)
    r = generate_report("input/clean_doc.jpg", d, ocr_word_count=10, encrypt=False)
    try:
        with open(r["report_path"], "r") as f:
            report = json.load(f)
        check(report["is_sensitive"] is False, "is_sensitive=False")
        check(report["total_findings"] == 0, "total_findings=0")
        check(report["findings"] == [], "findings=[]")
        check(len(report["notes"]) > 0, "Notes present for clean scan")
    except Exception as e:
        print(f"  FAIL ✗ — {e}")


def test_12_notes_for_low_confidence():
    print("\n[Test 12] Notes generated for low-confidence findings...")
    d = make_detection("DOB: 12/04/1990")   # dob is low confidence
    r = generate_report("input/test_notes.jpg", d, ocr_word_count=5, encrypt=False)
    try:
        with open(r["report_path"], "r") as f:
            report = json.load(f)
        notes_text = " ".join(report.get("notes", []))
        check("low" in notes_text.lower() or "confidence" in notes_text.lower(),
              "Low confidence note present")
    except Exception as e:
        print(f"  FAIL ✗ — {e}")


def test_13_finding_types_summary():
    print("\n[Test 13] finding_types summary is accurate...")
    d = make_detection("sample@gmail.com another@test.com ABCDE1234F")
    r = generate_report("input/test_types.jpg", d, ocr_word_count=5, encrypt=False)
    try:
        with open(r["report_path"], "r") as f:
            report = json.load(f)
        ft = report["finding_types"]
        check("email" in ft, "'email' in finding_types")
        check(ft.get("email", 0) >= 1, f"email count >= 1 (got {ft.get('email')})")
        check("pan" in ft, "'pan' in finding_types")
    except Exception as e:
        print(f"  FAIL ✗ — {e}")


def test_14_read_audit_log():
    print("\n[Test 14] read_audit_log() returns list of dicts...")
    entries = read_audit_log()
    check(isinstance(entries, list), "Returns a list")
    if entries:
        check(isinstance(entries[0], dict), "First entry is a dict")
        check("report_id" in entries[0], "First entry has 'report_id'")


def test_15_multiple_scans_separate_files():
    print("\n[Test 15] Multiple scans produce separate report files...")
    d = make_detection(SAMPLE_TEXT_SENSITIVE)
    r1 = generate_report("input/img_a.jpg", d, ocr_word_count=40, encrypt=False)
    r2 = generate_report("input/img_b.jpg", d, ocr_word_count=40, encrypt=False)
    check(r1["report_path"] != r2["report_path"],
          f"Different report files: {os.path.basename(r1['report_path'])} vs "
          f"{os.path.basename(r2['report_path'])}")
    check(r1["report_id"] != r2["report_id"], "Different report IDs")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  Reporter — Module 5 Tests (15 tests)")
    print("═" * 60)

    result = test_1_generates_without_crash()
    test_2_return_keys(result)
    test_3_report_file_exists(result)
    test_4_audit_log_exists()
    test_5_audit_log_no_sensitive_values()
    test_6_audit_log_fields()
    test_7_report_id_format(result)
    test_8_masking()
    test_9_encrypted_file_not_plain_json(result)
    test_10_decryption_works(result)
    test_11_clean_image_report()
    test_12_notes_for_low_confidence()
    test_13_finding_types_summary()
    test_14_read_audit_log()
    test_15_multiple_scans_separate_files()

    print("\n── All 15 tests done ──")