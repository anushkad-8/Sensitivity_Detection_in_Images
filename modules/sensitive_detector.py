"""
Module 3: sensitive_detector.py
--------------------------------
DLP Rule Engine — Regex-based detection of sensitive information from OCR text.

Architecture position:
    ocr_engine.py → text_for_detection → detect_sensitive() → matches + flags

CHANGES (Anti-Obfuscation Update):
    1. detect_sensitive() now accepts obfuscation_flags parameter from ocr_engine.
       Callers should pass ocr_result["obfuscation_flags"] here.
    2. Return dict includes obfuscation_flags so downstream decision engine
       can escalate images with obfuscation + sensitive content simultaneously.
    3. Auto-escalation logic added: if obfuscation detected AND regex matched,
       the combination is flagged as HIGH RISK (rare coincidence in clean docs).
    4. Input text should be ocr_result["text_for_detection"] — the cleanest
       version (normalized + base64/hex decoded). Using raw text also works
       but produces more false negatives.

CALLER IN main.py SHOULD DO:
    detection = detect_sensitive(
        ocr_result["text_for_detection"],
        obfuscation_flags=ocr_result["obfuscation_flags"]
    )

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
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        re.IGNORECASE
    ),

    # ── Indian Phone ──────────────────────────────────────────────────────────
    "phone": re.compile(
        r"(?:\+91[\s\-]?)?(?:\(?0?\)?[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}",
        re.IGNORECASE
    ),

    # ── Aadhaar ───────────────────────────────────────────────────────────────
    # 12 digits in groups of 4, first digit 2-9
    # NOTE: After normalization + reassembly, fragmented Aadhaar numbers are
    # rejoined, so "9183 0074 6619" matches even if OCR originally split them.
    "aadhaar": re.compile(
        r"\b[2-9]\d{3}[\s.\-]?\d{4}[\s.\-]?\d{4}\b"
    ),

    # ── PAN ───────────────────────────────────────────────────────────────────
    "pan": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    ),

    # ── Date of Birth ─────────────────────────────────────────────────────────
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
    "bank_card": re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),

    # ── Indian Passport Number ────────────────────────────────────────────────
    "passport_number": re.compile(
        r"\b[A-Z]\d{7}\b"
    ),

    # ── Voter ID (EPIC Number) ────────────────────────────────────────────────
    "voter_id": re.compile(
        r"\b[A-Z]{3}[0-9]{7}\b"
    ),

    # ── Driving Licence (Indian) ──────────────────────────────────────────────
    "driving_licence": re.compile(
        r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?\d{4}[\s\-]?\d{7}\b",
        re.IGNORECASE
    ),

    # ── GST Number ────────────────────────────────────────────────────────────
    "gst_number": re.compile(
        r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b"
    ),

    # ── IFSC Code ─────────────────────────────────────────────────────────────
    "ifsc_code": re.compile(
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
    ),

    # ── MRZ Line ─────────────────────────────────────────────────────────────
    "mrz_line": re.compile(
        r"[A-Z0-9<]{20,44}"
    ),

    # ── SWIFT / BIC Code ─────────────────────────────────────────────────────
    "swift_bic": re.compile(
        r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"
    ),

    # ── MICR Code ─────────────────────────────────────────────────────────────
    "micr_code": re.compile(
        r"\b\d{9}\b"
    ),
}


# ─────────────────────────────────────────────
# OVERLAP PRIORITY MAP
# ─────────────────────────────────────────────

SPAN_PROTECTED = {
    "aadhaar"        : ["bank_card"],
    "passport_number": ["bank_card"],
    "voter_id"       : ["bank_card", "pan"],
    "micr_code"      : ["bank_card"],
}

STANDARD_PATTERNS  = ["email", "phone", "pan", "dob", "gst_number", "ifsc_code",
                       "swift_bic", "driving_licence"]
SPAN_CLAIMERS      = ["bank_card"]
SPAN_AWARE_PATTERNS = ["aadhaar", "passport_number", "voter_id", "micr_code", "mrz_line"]


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def detect_sensitive(text: str, obfuscation_flags: dict = None) -> dict:
    """
    Run all regex patterns against OCR text.

    Args:
        text              : Use ocr_result["text_for_detection"] — normalized and
                            base64/hex decoded. Fallback to text_normalized or raw text.
        obfuscation_flags : Dict from ocr_result["obfuscation_flags"]. Pass this to
                            include obfuscation context in the result and enable
                            auto-escalation when both obfuscation AND sensitive content
                            are found simultaneously.

    Returns:
        {
            "is_sensitive"       : bool
            "total"              : int
            "matches"            : list of match dicts
            "obfuscation_flags"  : dict  — passed through from ocr_engine
            "escalate"           : bool  — True if obfuscation + sensitive content found
            "risk_level"         : str   — "high" | "medium" | "low" | "none"
        }
    """
    obfuscation_flags = obfuscation_flags or {}

    if not text or not text.strip():
        print("[Detector] Warning: Empty text received. Nothing to scan.")
        return _empty_result(obfuscation_flags)

    print(f"\n[Detector] Scanning {len(text)} chars across {len(PATTERNS)} pattern types...")

    all_matches   = []
    claimed_spans = {}

    # ── Step 1: Standard patterns ─────────────────────────────────────────────
    for ptype in STANDARD_PATTERNS:
        all_matches.extend(_run_pattern(ptype, text))

    # ── Step 2: Span claimers ────────────────────────────────────────────────
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

    # ── Escalation logic ──────────────────────────────────────────────────────
    # Obfuscation + sensitive content together is a strong signal of intentional
    # data exfiltration — not accidental. Escalate for human review.
    obf_detected = (
        obfuscation_flags.get("encrypted_content")
        or obfuscation_flags.get("hash_patterns_found")
        or obfuscation_flags.get("encoded_tokens_found")
    )
    escalate = bool(obfuscation_flags.get("escalate")) or (obf_detected and is_sensitive)

    risk_level = _compute_risk_level(all_matches, obfuscation_flags, escalate)

    if escalate and is_sensitive:
        print(f"\n[Detector] 🚨 HIGH RISK: Sensitive content + obfuscation technique detected.")
        print(f"[Detector]    Encrypted: {obfuscation_flags.get('encrypted_content')}")
        print(f"[Detector]    Hashes   : {obfuscation_flags.get('hash_patterns_found')}")
        print(f"[Detector]    Encoded  : {obfuscation_flags.get('encoded_tokens_found')}")

    _print_summary(all_matches, risk_level, escalate)

    return {
        "is_sensitive"      : is_sensitive,
        "total"             : len(all_matches),
        "matches"           : all_matches,
        "obfuscation_flags" : obfuscation_flags,
        "escalate"          : escalate,
        "risk_level"        : risk_level,
    }


# ─────────────────────────────────────────────
# RISK LEVEL COMPUTATION
# ─────────────────────────────────────────────

def _compute_risk_level(matches: list, obfuscation_flags: dict, escalate: bool) -> str:
    """
    Determine overall risk level for the image.

    HIGH:
        - Escalate flag set (obfuscation + sensitive content)
        - Any high-confidence match found
        - Encrypted or hashed content with any sensitive match

    MEDIUM:
        - Medium-confidence matches only
        - Encoded tokens found even without other matches

    LOW:
        - Low-confidence matches only

    NONE:
        - No matches, no obfuscation
    """
    if not matches and not obfuscation_flags.get("escalate"):
        return "none"

    if escalate:
        return "high"

    confidences = [m["confidence"] for m in matches]
    if "high" in confidences:
        return "high"
    if "medium" in confidences or obfuscation_flags.get("encoded_tokens_found"):
        return "medium"
    if confidences:
        return "low"
    return "none"


# ─────────────────────────────────────────────
# PATTERN RUNNERS (unchanged from original)
# ─────────────────────────────────────────────

def _run_pattern(pattern_type: str, text: str) -> list:
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
            print(f"[Detector] ↷ Skipped '{value}' ({pattern_type}) — overlaps higher-priority match.")
            continue
        seen.add(value)
        results.append(_make_match(pattern_type, value))
    return results


def _make_match(pattern_type: str, value: str) -> dict:
    tokens = value.split()
    conf   = CONFIDENCE.get(pattern_type, "low")
    print(f"[Detector] ✓ {pattern_type.upper():<18} → '{value}'  [{conf}]")
    return {
        "type"      : pattern_type,
        "value"     : value,
        "tokens"    : tokens,
        "confidence": conf,
        "source"    : "regex",
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _empty_result(obfuscation_flags: dict) -> dict:
    escalate = bool(obfuscation_flags.get("escalate"))
    return {
        "is_sensitive"      : False,
        "total"             : 0,
        "matches"           : [],
        "obfuscation_flags" : obfuscation_flags,
        "escalate"          : escalate,
        "risk_level"        : "high" if escalate else "none",
    }


def _print_summary(matches: list, risk_level: str, escalate: bool) -> None:
    print(f"\n[Detector] ── Detection Summary {'─' * 30}")
    risk_icon = {"high": "🚨", "medium": "⚠️ ", "low": "ℹ️ ", "none": "✅"}.get(risk_level, "?")
    print(f"[Detector]  {risk_icon}  Risk level: {risk_level.upper()}")
    if escalate:
        print(f"[Detector]  ⬆️   ESCALATION FLAG SET — human review required")
    if not matches:
        print("[Detector]  ✅  No regex sensitive patterns detected.")
    else:
        print(f"[Detector]  ⚠️   SENSITIVE CONTENT — {len(matches)} match(es)")
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

    print("═" * 60)
    print("TEST 1 — Standard patterns (unchanged behaviour)")
    print("═" * 60)
    core_text = (
        "sample@gmail.com "
        "+91 99999 99999 "
        "9876 5432 1098 "
        "ABCDE1234F "
        "DOB 12/04/1990 "
        "4111 1111 1111 1111"
    )
    r1 = detect_sensitive(core_text)
    print(f"Risk level: {r1['risk_level']} | Matches: {r1['total']}\n")

    print("═" * 60)
    print("TEST 2 — With obfuscation flags (escalation)")
    print("═" * 60)
    flags = {
        "encoded_tokens_found": True,
        "encrypted_content"   : False,
        "hash_patterns_found" : True,
        "escalate"            : True,
        "low_conf_ratio"      : 0.7,
    }
    r2 = detect_sensitive("ABCDE1234F John Doe", obfuscation_flags=flags)
    print(f"Risk level: {r2['risk_level']} | Escalate: {r2['escalate']}\n")

    print("═" * 60)
    print("TEST 3 — Obfuscation only, no regex match")
    print("═" * 60)
    flags_only = {
        "encrypted_content": True,
        "escalate"         : True,
    }
    r3 = detect_sensitive("some text without patterns", obfuscation_flags=flags_only)
    print(f"Risk level: {r3['risk_level']} | Escalate: {r3['escalate']}")
    print(f"is_sensitive: {r3['is_sensitive']} (False because no regex match)")