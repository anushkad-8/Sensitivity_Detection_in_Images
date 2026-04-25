"""
tests/test_sensitive_detector.py
---------------------------------
Tests for Module 3: sensitive_detector.py (upgraded with Indian document patterns)

Run from project root:
    python tests/test_sensitive_detector.py

CORE TESTS (1-8)   : Original patterns — email, phone, aadhaar, pan, dob, card
NEW TESTS  (9-15)  : Indian document patterns — passport, voter, gst, ifsc, dl, swift, micr
SYSTEM TESTS (16-18): Overlap fix, real passport OCR simulation, structure validation
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.sensitive_detector import detect_sensitive

PASS = "  PASS ✓"
FAIL = "  FAIL ✗"
SKIP = "  SKIP –"

def check(condition: bool, label: str) -> bool:
    print(f"{PASS if condition else FAIL} — {label}")
    return condition

def run(name: str, text: str, expect_type: str = None, expect_count: int = None):
    print(f"\n{name}")
    result = detect_sensitive(text)
    if expect_count is not None:
        check(result["total"] == expect_count,
              f"Expected {expect_count} match(es), got {result['total']}")
    if expect_type is not None:
        found = [m["type"] for m in result["matches"]]
        check(expect_type in found, f"'{expect_type}' detected. Found: {found}")
    return result


# ─────────────────────────────────────────────
# CORE PATTERN TESTS
# ─────────────────────────────────────────────

def test_1_empty():
    print("\n[Test 1] Empty string → is_sensitive=False")
    r = detect_sensitive("")
    check(r["is_sensitive"] is False, "is_sensitive=False")
    check(r["total"] == 0, "total=0")
    check(r["matches"] == [], "matches=[]")

def test_2_email():
    run("[Test 2] Email", "Contact: sample@gmail.com", "email", 1)

def test_3_phone_country_code():
    run("[Test 3] Phone +91", "Call: +91 99999 99999", "phone", 1)

def test_4_phone_plain():
    run("[Test 4] Phone plain 10-digit", "Mobile: 9876543210", "phone", 1)

def test_5_aadhaar_space():
    run("[Test 5] Aadhaar space-separated", "UID: 9876 5432 1098", "aadhaar", 1)

def test_6_pan():
    run("[Test 6] PAN number", "PAN: ABCDE1234F", "pan", 1)

def test_7_dob_numeric():
    run("[Test 7] DOB DD/MM/YYYY", "DOB: 12/04/1990", "dob", 1)

def test_8_dob_month():
    run("[Test 8] DOB month name", "Born: 12 April 1990", "dob", 1)


# ─────────────────────────────────────────────
# NEW INDIAN DOCUMENT PATTERN TESTS
# ─────────────────────────────────────────────

def test_9_passport_number():
    print("\n[Test 9] Passport number — Indian format (1 letter + 7 digits)")
    r = run("", "Passport No: J8369854", "passport_number", 1)
    vals = [m["value"] for m in r["matches"] if m["type"] == "passport_number"]
    check("J8369854" in vals, f"Correct value 'J8369854' detected. Got: {vals}")

def test_10_voter_id():
    run("[Test 10] Voter ID (EPIC)", "Voter ID: ABC1234567", "voter_id", 1)

def test_11_gst_number():
    print("\n[Test 11] GST number")
    r = run("", "GST: 27ABCDE1234F1Z5", "gst_number", 1)
    vals = [m["value"] for m in r["matches"] if m["type"] == "gst_number"]
    check(len(vals) > 0, f"GST detected: {vals}")

def test_12_ifsc_code():
    print("\n[Test 12] IFSC code")
    r = run("", "IFSC: SBIN0001234", "ifsc_code", 1)
    vals = [m["value"] for m in r["matches"] if m["type"] == "ifsc_code"]
    check("SBIN0001234" in vals, f"IFSC value correct: {vals}")

def test_13_swift_bic():
    run("[Test 13] SWIFT/BIC code", "Wire to: SBININBB", "swift_bic", 1)

def test_14_micr_code():
    run("[Test 14] MICR code", "MICR: 400002009", "micr_code", 1)

def test_15_mrz_line():
    print("\n[Test 15] MRZ line (passport machine readable zone)")
    mrz = "P<INDRAMADUGULA<<SITA<MAHA<LAKSHAI<<<<<<<<<<<"
    r = run("", mrz, "mrz_line")
    check(r["is_sensitive"], "MRZ line detected as sensitive")


# ─────────────────────────────────────────────
# SYSTEM / INTEGRATION TESTS
# ─────────────────────────────────────────────

def test_16_overlap_fix():
    """Critical: 16-digit card must NOT also match as Aadhaar."""
    print("\n[Test 16] Overlap fix — card digits must not match as Aadhaar")
    text = "Card: 4111 1111 1111 1111 and Aadhaar: 9876 5432 1098"
    r = detect_sensitive(text)
    aadhaar = [m["value"] for m in r["matches"] if m["type"] == "aadhaar"]
    card    = [m["value"] for m in r["matches"] if m["type"] == "bank_card"]
    check(len(card) == 1,    f"Exactly 1 card match: {card}")
    check("9876 5432 1098" in aadhaar, f"Real Aadhaar detected: {aadhaar}")
    check(not any("4111" in v for v in aadhaar),
          f"Card digits NOT in aadhaar matches: {aadhaar}")

def test_17_passport_simulation():
    """Simulate realistic OCR output from an Indian passport scan."""
    print("\n[Test 17] Passport OCR simulation — all key fields")
    text = (
        "REPUBLIC OF INDIA Type P Country Code IND "
        "No J8369854 "
        "RAMADUGULA SITA MAHA LAKSHAI "
        "Date of Birth 11/10/1990 "
        "Place of Birth HYDERABAD "
        "Date of Issue 11/10/2011 "
        "Date of Expiry 10/10/2021 "
        "P<INDRAMADUGULA<<SITA<MAHA<LAKSHAI<<<<<<<<<< "
        "J8369854<6IND9010112F2110101<<<<<<<<<<<<<<<4"
    )
    r = detect_sensitive(text)
    types = {m["type"] for m in r["matches"]}
    print(f"  Types detected: {sorted(types)}")
    check("passport_number" in types, "Passport number detected")
    check("dob" in types, "Date of birth detected")
    check("mrz_line" in types, "MRZ zone detected")
    check(r["is_sensitive"], "Overall: is_sensitive=True")

def test_18_result_structure():
    """Validate every required key exists in match objects."""
    print("\n[Test 18] Result structure — all required fields present")
    r = detect_sensitive("sample@gmail.com ABCDE1234F J8369854")
    check("is_sensitive" in r, "Key 'is_sensitive'")
    check("total" in r,        "Key 'total'")
    check("matches" in r,      "Key 'matches'")
    for m in r["matches"]:
        for key in ["type", "value", "tokens", "confidence", "source"]:
            check(key in m, f"Match key '{key}' in {m.get('type','?')}")
        check(m["source"] == "regex", f"source='regex' in {m.get('type','?')}")
        check(m["confidence"] in ["high","medium","low"],
              f"Valid confidence in {m.get('type','?')}: {m.get('confidence')}")

def test_19_deduplication():
    print("\n[Test 19] Same value twice → reported once")
    r = detect_sensitive("sample@gmail.com and again sample@gmail.com")
    emails = [m for m in r["matches"] if m["type"] == "email"]
    check(len(emails) == 1, f"Only 1 email match (got {len(emails)})")

def test_20_all_types_coverage():
    """One text with all supported types — verify full coverage."""
    print("\n[Test 20] Full coverage — all supported pattern types")
    text = (
        "Email: sample@gmail.com "
        "Phone: +91 99999 99999 "
        "Aadhaar: 9876 5432 1098 "
        "PAN: ABCDE1234F "
        "DOB: 12/04/1990 "
        "Card: 5500 0000 0000 0004 "
        "Passport: J8369854 "
        "Voter: ABC1234567 "
        "GST: 27ABCDE1234F1Z5 "
        "IFSC: SBIN0001234 "
        "SWIFT: SBININBB "
        "MICR: 400002009"
    )
    r = detect_sensitive(text)
    found_types = {m["type"] for m in r["matches"]}
    expected = {
        "email", "phone", "aadhaar", "pan", "dob",
        "bank_card", "passport_number", "voter_id",
        "gst_number", "ifsc_code", "swift_bic", "micr_code"
    }
    print(f"  Found:    {sorted(found_types)}")
    print(f"  Expected: {sorted(expected)}")
    missing = expected - found_types
    if missing:
        print(f"  WARN — Not detected: {missing}")
    else:
        print(f"  PASS ✓ — All {len(expected)} types detected")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  Sensitive Detector — Full Test Suite (20 tests)")
    print("═" * 60)

    # Core
    test_1_empty()
    test_2_email()
    test_3_phone_country_code()
    test_4_phone_plain()
    test_5_aadhaar_space()
    test_6_pan()
    test_7_dob_numeric()
    test_8_dob_month()

    # New Indian document patterns
    test_9_passport_number()
    test_10_voter_id()
    test_11_gst_number()
    test_12_ifsc_code()
    test_13_swift_bic()
    test_14_micr_code()
    test_15_mrz_line()

    # System tests
    test_16_overlap_fix()
    test_17_passport_simulation()
    test_18_result_structure()
    test_19_deduplication()
    test_20_all_types_coverage()

    print("\n── All 20 tests done ──")