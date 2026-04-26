"""
Module 5: reporter.py
----------------------
Structured audit report generator for DLP detection results.

Architecture position:
    annotator.py (annotated image output)
            +
    sensitive_detector.py (detection results)
            ↓
    reporter.py → encrypted JSON report + plain audit log

Barclays Requirement (from mentor feedback):
    "Data storage design must include:
     - Secure, structured metadata storage
     - Masking and encryption of sensitive outputs"

What this module produces per scan:
    1. Encrypted JSON report  → output/reports/<name>_report.enc
       Contains full findings with masked values, encrypted with Fernet (AES-128)
       Key stored separately at output/reports/.dlp_key (chmod restricted on Linux)

    2. Plain audit log entry  → output/reports/audit_log.jsonl
       One JSON line per scan — no sensitive values, safe to store unencrypted
       Contains: filename, timestamp, is_sensitive, total findings, types found

    3. Console summary        → printed to stdout during pipeline run

ENCRYPTION APPROACH:
    - Uses Python cryptography library (Fernet = AES-128-CBC + HMAC-SHA256)
    - Key generated once and reused across scans (stored in .dlp_key file)
    - Sensitive VALUES are masked in the audit log (shown as ***MASKED***)
    - Full values only in encrypted report (decryptable by key holder)
    - This satisfies Barclays' "masking + encryption of sensitive outputs"

PHASE 2 READINESS:
    - Report format includes "source" field per finding (regex / nlp / vision)
    - NLP and vision findings will slot into the same structure automatically
    - "confidence_score" field reserved for Phase 2 unified confidence engine

"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional

# Try to import cryptography — graceful fallback if not installed
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    print("[Reporter] WARNING: 'cryptography' package not installed.")
    print("[Reporter] Run: pip install cryptography")
    print("[Reporter] Reports will be saved as plain JSON (NOT encrypted).")


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

REPORTS_DIR = os.path.join("output", "reports")
KEY_FILE    = os.path.join(REPORTS_DIR, ".dlp_key")
AUDIT_LOG   = os.path.join(REPORTS_DIR, "audit_log.jsonl")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def generate_report(
    image_path       : str,
    detection_result : dict,
    ocr_word_count   : int,
    annotated_path   : Optional[str] = None,
    encrypt          : bool = True
) -> dict:
    """
    Generate a full structured report for a single image scan.

    Args:
        image_path       : Path to the original input image.
        detection_result : Output dict from sensitive_detector.detect_sensitive().
        ocr_word_count   : Number of words extracted by OCR (after confidence filter).
        annotated_path   : Path to the annotated/redacted output image (if any).
        encrypt          : If True, encrypt the report JSON. Default True.
                           Set False only for debugging — never in production.

    Returns:
        {
            "report_path"    : path to encrypted (or plain) report file
            "audit_log_path" : path to audit log file
            "report_id"      : unique ID for this scan
            "summary"        : plain-text summary string for console
        }
    """

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Build report payload ──────────────────────────────────────────────────
    report_id  = _generate_report_id(image_path)
    timestamp  = datetime.now(timezone.utc).isoformat()
    image_name = os.path.basename(image_path)

    report = {
        "report_id"      : report_id,
        "schema_version" : "1.0",                  # bump when structure changes
        "generated_at"   : timestamp,
        "image_file"     : image_name,
        "image_path"     : os.path.abspath(image_path),
        "annotated_path" : annotated_path or "N/A",
        "ocr_word_count" : ocr_word_count,
        "is_sensitive"   : detection_result["is_sensitive"],
        "total_findings" : detection_result["total"],
        "findings"       : _build_findings(detection_result["matches"]),
        "finding_types"  : _summarise_types(detection_result["matches"]),
        "phase"          : _detect_phase(detection_result["matches"]),
        "notes"          : _build_notes(detection_result),
    }

    # ── Save encrypted or plain report ───────────────────────────────────────
    report_path = _save_report(report, image_name, encrypt)

    # ── Append to audit log (no sensitive values) ─────────────────────────────
    _append_audit_log(report, image_name, report_path, timestamp)

    # ── Build console summary ─────────────────────────────────────────────────
    summary = _build_summary(report, report_path)
    print(summary)

    return {
        "report_path"    : report_path,
        "audit_log_path" : AUDIT_LOG,
        "report_id"      : report_id,
        "summary"        : summary,
    }


# ─────────────────────────────────────────────
# REPORT BUILDING
# ─────────────────────────────────────────────

def _build_findings(matches: list) -> list:
    """
    Build structured finding records from detection matches.

    Each finding includes:
        type          : pattern type (email, pan, passport_number etc.)
        value_masked  : value with middle chars replaced by *** (audit-safe)
        value_full    : complete matched value (only in encrypted report)
        confidence    : high / medium / low
        source        : regex / nlp / vision
        tokens        : individual word tokens (for annotator reference)

    Masking examples:
        sample@gmail.com  →  sa***@***.com
        J8369854          →  J8***854
        9876 5432 1098    →  98***098
        ABCDE1234F        →  AB***4F
    """
    findings = []
    for match in matches:
        findings.append({
            "type"         : match["type"],
            "value_masked" : _mask_value(match["value"], match["type"]),
            "value_full"   : match["value"],    # only safe because report is encrypted
            "confidence"   : match["confidence"],
            "source"       : match["source"],
            "tokens"       : match["tokens"],
        })
    return findings


def _mask_value(value: str, pattern_type: str) -> str:
    """
    Mask sensitive value for audit log and console display.
    Preserves enough characters to identify the type without exposing the value.

    Strategy:
        Short values  (<= 6 chars) : show first + last char only → A***F
        Medium values (7-12 chars) : show first 2 + last 2      → AB***4F
        Long values   (>12 chars)  : show first 3 + last 3      → sam***com
        Email                      : mask local part and domain separately
        Aadhaar / card             : show last 4 digits only     → *** *** 1098
    """
    value = value.strip()

    # Special handling for email
    if pattern_type == "email" and "@" in value:
        local, domain = value.split("@", 1)
        masked_local  = local[:2] + "***" if len(local) > 2 else "***"
        domain_parts  = domain.split(".")
        masked_domain = "***." + domain_parts[-1] if domain_parts else "***"
        return f"{masked_local}@{masked_domain}"

    # Aadhaar and card — show last 4 digits only
    if pattern_type in ("aadhaar", "bank_card"):
        digits = value.replace(" ", "").replace("-", "").replace(".", "")
        return f"**** **** {digits[-4:]}"

    # General masking by length
    n = len(value)
    if n <= 6:
        return value[0] + "***" + value[-1]
    elif n <= 12:
        return value[:2] + "***" + value[-2:]
    else:
        return value[:3] + "***" + value[-3:]


def _summarise_types(matches: list) -> dict:
    """
    Return count of each finding type.
    Example: {"email": 1, "pan": 2, "dob": 1}
    Used in audit log for quick filtering without exposing values.
    """
    summary = {}
    for m in matches:
        summary[m["type"]] = summary.get(m["type"], 0) + 1
    return summary


def _detect_phase(matches: list) -> str:
    """Return highest pipeline phase represented in the finding sources."""
    sources = {m.get("source", "regex") for m in matches}
    if "vision" in sources:
        return "phase_3_vision"
    if "nlp" in sources:
        return "phase_2_nlp"
    return "phase_1_regex"


def _build_notes(detection_result: dict) -> list:
    """
    Generate human-readable notes about the scan quality and findings.
    These are advisory notes for the analyst reviewing the report.
    """
    notes = []

    if not detection_result["is_sensitive"]:
        notes.append("No sensitive content detected. Image appears clean.")
        return notes

    # Check for low-confidence findings
    low_conf = [m for m in detection_result["matches"] if m["confidence"] == "low"]
    if low_conf:
        types = list({m["type"] for m in low_conf})
        notes.append(
            f"Low-confidence findings present ({', '.join(types)}). "
            "Recommend NLP phase review to confirm or dismiss."
        )

    # Check for MRZ presence
    mrz = [m for m in detection_result["matches"] if m["type"] == "mrz_line"]
    if mrz:
        notes.append(
            "MRZ zone detected — image likely contains a passport or travel document."
        )

    # Check for high-confidence findings
    high_conf = [m for m in detection_result["matches"] if m["confidence"] == "high"]
    if high_conf:
        types = list({m["type"] for m in high_conf})
        notes.append(
            f"High-confidence findings: {', '.join(types)}. "
            "Immediate review recommended."
        )

    return notes


# ─────────────────────────────────────────────
# ENCRYPTION
# ─────────────────────────────────────────────

def _get_or_create_key() -> bytes:
    """
    Load existing encryption key or generate a new one.

    Key is stored at output/reports/.dlp_key
    On Linux/Mac: file permissions set to 600 (owner read/write only).
    On Windows: ACL restriction is not applied — warn the user.

    IMPORTANT: This key must be backed up securely.
    If lost, encrypted reports cannot be decrypted.
    """
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()

    # Generate new key
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)

    # Restrict permissions on Unix systems
    if os.name != "nt":
        os.chmod(KEY_FILE, 0o600)
        print(f"[Reporter] New encryption key created: {KEY_FILE} (permissions: 600)")
    else:
        print(f"[Reporter] New encryption key created: {KEY_FILE}")
        print(f"[Reporter] ⚠️  Windows: Manually restrict access to this file.")

    return key


def _encrypt_report(report_json: str) -> bytes:
    """Encrypt report JSON string using Fernet (AES-128-CBC + HMAC-SHA256)."""
    key   = _get_or_create_key()
    f     = Fernet(key)
    token = f.encrypt(report_json.encode("utf-8"))
    return token


def decrypt_report(encrypted_path: str, key_path: str = None) -> dict:
    """
    Decrypt an encrypted report file and return the JSON dict.

    Args:
        encrypted_path : Path to the .enc report file.
        key_path       : Path to the key file. Defaults to standard key location.

    Returns:
        Decrypted report as a Python dict.

    Usage:
        from modules.reporter import decrypt_report
        report = decrypt_report("output/reports/myimage_report.enc")
        print(report["findings"])
    """
    if not ENCRYPTION_AVAILABLE:
        raise RuntimeError("cryptography package not installed.")

    key_path = key_path or KEY_FILE
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Key file not found: {key_path}")
    if not os.path.exists(encrypted_path):
        raise FileNotFoundError(f"Report file not found: {encrypted_path}")

    with open(key_path, "rb") as f:
        key = f.read()
    with open(encrypted_path, "rb") as f:
        token = f.read()

    fernet      = Fernet(key)
    decrypted   = fernet.decrypt(token)
    return json.loads(decrypted.decode("utf-8"))


# ─────────────────────────────────────────────
# FILE I/O
# ─────────────────────────────────────────────

def _save_report(report: dict, image_name: str, encrypt: bool) -> str:
    """
    Save report to disk — encrypted (.enc) or plain (.json).

    Filename: output/reports/<image_stem>_<report_id[:8]>_report.enc
    Short report_id prefix in filename allows finding reports for a specific image
    without having to decrypt them all.
    """
    stem      = os.path.splitext(image_name)[0]
    short_id  = report["report_id"][:8]
    ext       = ".enc" if (encrypt and ENCRYPTION_AVAILABLE) else ".json"
    filename  = f"{stem}_{short_id}_report{ext}"
    filepath  = os.path.join(REPORTS_DIR, filename)

    report_json = json.dumps(report, indent=2, ensure_ascii=False)

    if encrypt and ENCRYPTION_AVAILABLE:
        encrypted = _encrypt_report(report_json)
        with open(filepath, "wb") as f:
            f.write(encrypted)
        print(f"[Reporter] ✅ Encrypted report saved: {filepath}")
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_json)
        if encrypt and not ENCRYPTION_AVAILABLE:
            print(f"[Reporter] ⚠️  Plain JSON saved (encryption unavailable): {filepath}")
        else:
            print(f"[Reporter] Report saved (plain): {filepath}")

    return filepath


def _append_audit_log(report: dict, image_name: str, report_path: str, timestamp: str) -> None:
    """
    Append one line to the audit log (JSONL format — one JSON object per line).

    Audit log contains NO sensitive values — only metadata.
    Safe to store unencrypted, back up to SIEM, or share with compliance team.

    Fields:
        report_id      : links this log entry to its encrypted report
        scanned_at     : UTC timestamp
        image_file     : filename only (no path)
        is_sensitive   : bool
        total_findings : int
        finding_types  : dict of {type: count} — no values
        report_file    : path to encrypted report for this scan
        phase          : which pipeline phase produced these results
    """
    audit_entry = {
        "report_id"     : report["report_id"],
        "scanned_at"    : timestamp,
        "image_file"    : image_name,
        "is_sensitive"  : report["is_sensitive"],
        "total_findings": report["total_findings"],
        "finding_types" : report["finding_types"],
        "ocr_word_count": report["ocr_word_count"],
        "report_file"   : os.path.basename(report_path),
        "phase"         : report["phase"],
    }

    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_entry) + "\n")

    print(f"[Reporter] ✅ Audit log updated: {AUDIT_LOG}")


# ─────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────

def _build_summary(report: dict, report_path: str) -> str:
    """Build a clean console summary block for main.py to print."""
    lines = [
        f"\n{'═' * 56}",
        f"  DLP SCAN REPORT",
        f"{'─' * 56}",
        f"  Image       : {report['image_file']}",
        f"  Scanned at  : {report['generated_at']}",
        f"  Report ID   : {report['report_id']}",
        f"  OCR words   : {report['ocr_word_count']}",
        f"{'─' * 56}",
    ]

    if report["is_sensitive"]:
        lines.append(f"  STATUS      : ⚠️  SENSITIVE — {report['total_findings']} finding(s)")
        lines.append(f"{'─' * 56}")
        for finding in report["findings"]:
            conf_marker = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(
                finding["confidence"], "⚪"
            )
            lines.append(
                f"  {conf_marker} {finding['type']:<18} "
                f"{finding['value_masked']:<25} "
                f"[{finding['confidence']}] [{finding['source']}]"
            )
    else:
        lines.append(f"  STATUS      : ✅ CLEAN — No sensitive content detected")

    if report["notes"]:
        lines.append(f"{'─' * 56}")
        lines.append(f"  NOTES:")
        for note in report["notes"]:
            lines.append(f"    • {note}")

    lines.append(f"{'─' * 56}")
    lines.append(f"  Report file : {report_path}")
    lines.append(f"  Audit log   : {AUDIT_LOG}")
    lines.append(f"{'═' * 56}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _generate_report_id(image_path: str) -> str:
    """
    Generate a unique report ID from image path + current timestamp.
    Format: first 32 chars of SHA-256 hash.
    Deterministic within same second but unique across scans.
    """
    seed = f"{os.path.abspath(image_path)}-{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


def read_audit_log() -> list:
    """
    Read and return all entries from the audit log as a list of dicts.
    Useful for building dashboards or compliance reports.
    """
    if not os.path.exists(AUDIT_LOG):
        return []
    entries = []
    with open(AUDIT_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from modules.sensitive_detector import detect_sensitive

    # Simulate what main.py will pass in
    sample_text = (
        "sample@gmail.com +91 99999 99999 "
        "ABCDE1234F 9876 5432 1098 "
        "DOB 12/04/1990 J8369854"
    )

    print("── Simulating full pipeline output ──")
    detection = detect_sensitive(sample_text)

    result = generate_report(
        image_path       = "input/test_sample.jpg",
        detection_result = detection,
        ocr_word_count   = 79,
        annotated_path   = "output/test_sample_annotated.jpg",
        encrypt          = True
    )

    print(f"\n── Report generated ──")
    print(f"Report path    : {result['report_path']}")
    print(f"Audit log path : {result['audit_log_path']}")
    print(f"Report ID      : {result['report_id']}")

    # Test decryption
    if ENCRYPTION_AVAILABLE and result["report_path"].endswith(".enc"):
        print(f"\n── Testing decryption ──")
        try:
            decrypted = decrypt_report(result["report_path"])
            print(f"Decryption: ✅ SUCCESS")
            print(f"Findings in decrypted report: {len(decrypted['findings'])}")
            for f in decrypted["findings"]:
                print(f"  → {f['type']:<18} masked: {f['value_masked']:<25} full: {f['value_full']}")
        except Exception as e:
            print(f"Decryption: ❌ FAILED — {e}")

    print(f"\n── Audit log entries ──")
    for entry in read_audit_log():
        print(json.dumps(entry, indent=2))
