"""
Module 3: sensitive_detector.py
--------------------------------
DLP Rule Engine — Regex-based detection of sensitive information from OCR text.

Architecture position:
    Text Extraction (raw string) → DLP Rule Engine → Regex Patterns → OCR Flag

PHASE 1 PATTERN COVERAGE (Indian document focus):
    Core identifiers   : Email, Phone, PAN, Aadhaar, DOB, Bank/Card
    Indian documents   : Passport No., Voter ID, Driving Licence, GST, IFSC
    Structured zones   : MRZ (Machine Readable Zone on passports/visas)
    Financial          : SWIFT/BIC code, MICR code

OVERLAP PROTECTION (span-aware matching):
    bank_card runs first → claims its character spans
    aadhaar    runs with exclusion → skips spans already claimed by bank_card
    passport   runs with exclusion → skips spans already claimed by bank_card
    This prevents a 16-digit card being partially re-matched as Aadhaar/passport.

CONFIDENCE LEVELS:
    high   → very specific format, near-zero false positives (PAN, passport, GST)
    medium → moderately specific, small false positive risk (Aadhaar, card, IFSC)
    low    → broad pattern, context-dependent (DOB, phone, driving licence)

PHASE 2 READINESS:
    Every match carries source="regex" so NLP matches (source="nlp") can be
    merged into the same list without any structural changes.
"""

import re
import json


# ─────────────────────────────────────────────
# CONFIDENCE LEVELS
# ─────────────────────────────────────────────

CONFIDENCE = {
    # Core
    "email"           : "high",
    "phone"           : "high",
    "pan"             : "high",
    "aadhaar"         : "medium",
    "bank_card"       : "medium",
    "dob"             : "low",
    # Indian documents
    "passport_number" : "high",
    "voter_id"        : "medium",
    "driving_licence" : "low",
    "gst_number"      : "high",
    "ifsc_code"       : "high",
    # Structured zones
    "mrz_line"        : "high",
    # Financial
    "swift_bic"       : "high",
    "micr_code"       : "medium",
}


# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

PATTERNS = {

    # ── Email ─────────────────────────────────────────────────────────────────
    # local@domain.tld — handles dots, hyphens, plus signs in local part
    # Example: sample@gmail.com | john.doe+tag@company.org
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        re.IGNORECASE
    ),

    # ── Indian Phone ──────────────────────────────────────────────────────────
    # Handles: +91 99999 99999 | +91-9876543210 | 9876543210 | 09876543210
    # Starts with 6-9 (valid Indian mobile prefix)
    "phone": re.compile(
        r"(?:\+91[\s\-]?)?(?:\(?0?\)?[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}",
        re.IGNORECASE
    ),

    # ── Aadhaar ───────────────────────────────────────────────────────────────
    # 12 digits in groups of 4, separated by space/dot/dash or nothing
    # First digit: 2-9 (valid Aadhaar constraint — never starts with 0 or 1)
    # Example: 9876 5432 1098 | 9876.5432.1098 | 987654321098
    "aadhaar": re.compile(
        r"\b[2-9]\d{3}[\s.\-]?\d{4}[\s.\-]?\d{4}\b"
    ),

    # ── PAN ───────────────────────────────────────────────────────────────────
    # Exact format: 5 uppercase letters + 4 digits + 1 uppercase letter
    # Example: ABCDE1234F | RAMDU1234F
    # High specificity — format is extremely precise, near-zero false positives
    "pan": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    ),

    # ── Date of Birth ─────────────────────────────────────────────────────────
    # DD/MM/YYYY | DD-MM-YYYY | DD.MM.YYYY
    # YYYY/MM/DD | YYYY-MM-DD | YYYY.MM.DD
    # DD Month YYYY (e.g. 12 April 1990)
    # ⚠️ Low confidence — date formats also appear on invoices, expiry dates etc.
    #    NLP phase will add context ("DOB", "born on", "date of birth") to upgrade.
    "dob": re.compile(
        r"\b(?:"
        r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}"
        r"|"
        r"\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"
        r"|"
        r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}"
        r")\b",
        re.IGNORECASE
    ),

    # ── Bank / Credit Card ────────────────────────────────────────────────────
    # 16 digits in groups of 4, separated by space/dash or nothing
    # Covers: Visa, Mastercard, RuPay (all 16-digit)
    # Note: Amex (15-digit) and Diners (14-digit) added separately below
    # Example: 4111 1111 1111 1111 | 4111-1111-1111-1111 | 4111111111111111
    "bank_card": re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),

    # ── Indian Passport Number ────────────────────────────────────────────────
    # Format: 1 uppercase letter + 7 digits
    # Example: J8369854 | A1234567 | Z9876543
    # ⚠️ Must not overlap with bank card span — span-aware runner used
    # Your test passport had: J8369854 — extracted by OCR, now detectable
    "passport_number": re.compile(
        r"\b[A-Z]\d{7}\b"
    ),

    # ── Voter ID (EPIC Number) ────────────────────────────────────────────────
    # Format: 3 uppercase letters + 7 digits (10 chars total)
    # Example: ABC1234567 | XYZ9876543
    # Issued by Election Commission of India
    "voter_id": re.compile(
        r"\b[A-Z]{3}[0-9]{7}\b"
    ),

    # ── Driving Licence (Indian) ──────────────────────────────────────────────
    # Format: 2-letter state code + 2-digit district + 4-digit year + 7 digits
    # Example: MH1220110012345 | DL0420190123456
    # ⚠️ Low confidence — format varies slightly across states
    #    Space/dash separators also common: MH-12-2011-0012345
    "driving_licence": re.compile(
        r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?\d{4}[\s\-]?\d{7}\b",
        re.IGNORECASE
    ),

    # ── GST Number ────────────────────────────────────────────────────────────
    # Format: 2-digit state code + PAN (10 chars) + 1 digit + Z + 1 alphanumeric
    # Total: 15 characters
    # Example: 27ABCDE1234F1Z5 | 07RAMDU1234F1Z3
    # High specificity — embeds PAN format within it
    "gst_number": re.compile(
        r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b"
    ),

    # ── IFSC Code ─────────────────────────────────────────────────────────────
    # Format: 4 uppercase letters (bank code) + 0 + 6 alphanumeric (branch)
    # Example: SBIN0001234 | HDFC0000123 | ICIC0001234
    # The 5th character is always 0 — used to identify it as IFSC
    "ifsc_code": re.compile(
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
    ),

    # ── MRZ Line (Machine Readable Zone) ─────────────────────────────────────
    # Found at bottom of passports, visas, national IDs
    # Two lines of exactly 44 characters using A-Z, 0-9, and < (filler)
    # Example line 1: P<INDRAMADUGULA<<SITA<MAHA<LAKSHAI<<<<<<<<<<<
    # Example line 2: J8369854<6IND9010112F2110101<<<<<<<<<<<<<<<<<<4
    # ⚠️ OCR often garbles MRZ zones — this catches partial matches too
    # Minimum 20 chars of valid MRZ chars to avoid false positives
    "mrz_line": re.compile(
        r"[A-Z0-9<]{20,44}"
    ),

    # ── SWIFT / BIC Code ─────────────────────────────────────────────────────
    # Format: 4-letter bank code + 2-letter country + 2 char location + 3 char branch (optional)
    # Example: SBININBB | HDFCINBB | ICICINBBCTS
    # Used in international wire transfers — highly sensitive financial identifier
    "swift_bic": re.compile(
        r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"
    ),

    # ── MICR Code ─────────────────────────────────────────────────────────────
    # Format: 9 digits (city + bank + branch)
    # Example: 400002009 | 110002003
    # Found at the bottom of Indian cheques
    # ⚠️ Medium confidence — 9-digit sequences can appear in other contexts
    "micr_code": re.compile(
        r"\b\d{9}\b"
    ),
}


# ─────────────────────────────────────────────
# OVERLAP PRIORITY MAP
# ─────────────────────────────────────────────
# Defines which patterns must run with span exclusion.
# Format: { pattern_type : [list of pattern_types whose spans to exclude] }
#
# Why:
#   bank_card (16 digits) contains aadhaar (12 digits) as a substring.
#   bank_card (16 digits) starts with a pattern that could match passport_number.
#   We run higher-specificity patterns first and protect their spans.

SPAN_PROTECTED = {
    "aadhaar"        : ["bank_card"],
    "passport_number": ["bank_card"],
    "voter_id"       : ["bank_card", "pan"],  # 10-char voter ID could overlap PAN
    "micr_code"      : ["bank_card"],          # 9-digit MICR could be inside card
}

# Patterns that run standard (no span conflict risk)
STANDARD_PATTERNS = [
    "email", "phone", "pan", "dob",
    "gst_number", "ifsc_code", "swift_bic",
    "driving_licence",
]

# Patterns that must run first and claim spans
SPAN_CLAIMERS = ["bank_card"]

# Patterns that run with span exclusion
SPAN_AWARE_PATTERNS = ["aadhaar", "passport_number", "voter_id", "micr_code", "mrz_line"]


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def detect_sensitive(text: str) -> dict:
    """
    Run all regex patterns against extracted OCR text.

    Execution order:
        1. Standard patterns     — no overlap risk
        2. Span claimers         — run first, record their spans
        3. Span-aware patterns   — run with exclusion of claimed spans

    Returns:
        {
            "is_sensitive" : bool
            "total"        : int
            "matches"      : list of {type, value, tokens, confidence, source}
        }
    """
    if not text or not text.strip():
        print("[Detector] Warning: Empty text received. Nothing to scan.")
        return _empty_result()

    print(f"\n[Detector] Scanning {len(text)} chars across {len(PATTERNS)} pattern types...")

    all_matches  = []
    claimed_spans = {}   # { pattern_type : set of (start, end) tuples }

    # ── Step 1: Standard patterns ─────────────────────────────────────────────
    for ptype in STANDARD_PATTERNS:
        all_matches.extend(_run_pattern(ptype, text))

    # ── Step 2: Span claimers (bank_card) ────────────────────────────────────
    for ptype in SPAN_CLAIMERS:
        results, spans = _run_pattern_with_spans(ptype, text)
        all_matches.extend(results)
        claimed_spans[ptype] = spans

    # ── Step 3: Span-aware patterns ───────────────────────────────────────────
    for ptype in SPAN_AWARE_PATTERNS:
        excluded = set()
        for blocker in SPAN_PROTECTED.get(ptype, []):
            excluded |= claimed_spans.get(blocker, set())
        results = _run_pattern_span_aware(ptype, text, excluded)
        all_matches.extend(results)

    is_sensitive = len(all_matches) > 0
    _print_summary(all_matches)

    return {
        "is_sensitive": is_sensitive,
        "total"       : len(all_matches),
        "matches"     : all_matches,
    }


# ─────────────────────────────────────────────
# PATTERN RUNNERS
# ─────────────────────────────────────────────

def _run_pattern(pattern_type: str, text: str) -> list:
    """Standard runner — no span tracking. Used for non-overlapping types."""
    pattern = PATTERNS[pattern_type]
    results = []
    seen    = set()

    for m in pattern.finditer(text):
        value = m.group().strip()
        if not value or value in seen:
            continue
        seen.add(value)
        results.append(_make_match(pattern_type, value))

    return results


def _run_pattern_with_spans(pattern_type: str, text: str) -> tuple:
    """
    Run pattern and record all matched character spans.
    Used by span claimers (bank_card) so downstream patterns can avoid overlapping.

    Returns:
        (results list, set of (start, end) character span tuples)
    """
    pattern = PATTERNS[pattern_type]
    results = []
    spans   = set()
    seen    = set()

    for m in pattern.finditer(text):
        value = m.group().strip()
        if not value or value in seen:
            continue
        seen.add(value)
        spans.add((m.start(), m.end()))
        results.append(_make_match(pattern_type, value))

    return results, spans


def _run_pattern_span_aware(pattern_type: str, text: str, excluded_spans: set) -> list:
    """
    Run pattern but skip matches that overlap with any excluded character span.

    This prevents:
        - Aadhaar matching the first 12 digits of a 16-digit credit card
        - Passport number matching at the start of a card number
        - MICR code matching inside a credit card number

    Overlap check: match_start < excluded_end AND match_end > excluded_start
    This is the standard interval overlap formula — catches partial overlaps too.
    """
    pattern = PATTERNS[pattern_type]
    results = []
    seen    = set()

    for m in pattern.finditer(text):
        value = m.group().strip()
        if not value or value in seen:
            continue

        ms, me = m.start(), m.end()
        overlaps = any(ms < ex_end and me > ex_start
                       for ex_start, ex_end in excluded_spans)

        if overlaps:
            print(f"[Detector] ↷ Skipped '{value}' ({pattern_type}) — overlaps with higher-priority match.")
            continue

        seen.add(value)
        results.append(_make_match(pattern_type, value))

    return results


def _make_match(pattern_type: str, value: str) -> dict:
    """Build a structured match dict and log the detection."""
    tokens = value.split()
    conf   = CONFIDENCE.get(pattern_type, "low")
    print(f"[Detector] ✓ {pattern_type.upper():<18} → '{value}'  [{conf}]")
    return {
        "type"      : pattern_type,
        "value"     : value,
        "tokens"    : tokens,
        "confidence": conf,
        "source"    : "regex",   # Phase 2 NLP will use source="nlp"
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _empty_result() -> dict:
    return {"is_sensitive": False, "total": 0, "matches": []}


def _print_summary(matches: list) -> None:
    print(f"\n[Detector] ── Detection Summary {'─' * 30}")
    if not matches:
        print("[Detector]  ✅  No sensitive information detected.")
    else:
        print(f"[Detector]  ⚠️   SENSITIVE CONTENT FOUND — {len(matches)} match(es)")
        # Group by type for clean display
        by_type = {}
        for m in matches:
            by_type.setdefault(m["type"], []).append(m["value"])
        for dtype, values in by_type.items():
            for v in values:
                print(f"[Detector]     → {dtype:<18} : {v}")
    print(f"[Detector] {'─' * 52}")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test 1: Core patterns ──
    print("═" * 60)
    print("TEST 1 — Core patterns + overlap fix")
    print("═" * 60)
    core_text = (
        "sample@gmail.com "
        "+91 99999 99999 "
        "9876 5432 1098 "           # Aadhaar
        "ABCDE1234F "               # PAN
        "DOB 12/04/1990 "
        "4111 1111 1111 1111"       # Credit card — must NOT match as Aadhaar
    )
    r1 = detect_sensitive(core_text)
    aadhaar_vals = [m["value"] for m in r1["matches"] if m["type"] == "aadhaar"]
    card_vals    = [m["value"] for m in r1["matches"] if m["type"] == "bank_card"]
    print(f"\nOverlap fix: aadhaar={aadhaar_vals} | bank_card={card_vals}")
    assert not any("4111" in v for v in aadhaar_vals), "BUG: Aadhaar matched inside card!"
    print("✅ Overlap fix confirmed.\n")

    # ── Test 2: Indian document patterns ──
    print("═" * 60)
    print("TEST 2 — Indian document patterns (passport, voter, GST, IFSC, DL)")
    print("═" * 60)
    doc_text = (
        "Passport No: J8369854 "            # Passport number
        "Voter ID: ABC1234567 "             # Voter ID
        "GST: 27ABCDE1234F1Z5 "            # GST number
        "IFSC: SBIN0001234 "               # IFSC code
        "DL: MH12 2011 0012345 "           # Driving licence
        "SWIFT: SBININBB "                 # SWIFT/BIC
        "MICR: 400002009"                  # MICR code
    )
    r2 = detect_sensitive(doc_text)
    print(json.dumps(r2, indent=2))

    # ── Test 3: Simulated passport OCR output ──
    print("═" * 60)
    print("TEST 3 — Simulated passport OCR (mirrors your test3_passport.jpg)")
    print("═" * 60)
    passport_text = (
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
    r3 = detect_sensitive(passport_text)
    types_found = list({m["type"] for m in r3["matches"]})
    print(f"\nTypes detected on passport: {types_found}")