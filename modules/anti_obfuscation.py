"""
Module 2b: anti_obfuscation.py
-------------------------------
Detects deliberate OCR-evasion techniques embedded in extracted text.

Called by ocr_engine.py AFTER post-processing, BEFORE returning results.

WHAT THIS MODULE HANDLES:
    1. Base64 encoding  — decode and recover plaintext
    2. Hex encoding     — decode and recover plaintext
    3. Encryption       — DETECT ONLY. Flags for human review.
    4. Hashing          — DETECT ONLY. Flags for human review.

CHANGES (v3 — EC Private Key + full PEM coverage):
    - ENCRYPTION_MARKERS now includes EC PRIVATE KEY, DSA, OPENSSH,
      ENCRYPTED PRIVATE KEY, PKCS8, and CERTIFICATE REQUEST.
    - Patterns use \s* to tolerate OCR-introduced spacing gaps inside headers.
    - has_encrypted is True for ANY private key or certificate marker.
    - escalate now triggers on has_encrypted as well (was missing before).
"""

import re
import base64
import binascii


# ─────────────────────────────────────────────
# HASH PATTERNS
# ─────────────────────────────────────────────

HASH_PATTERNS = [
    ("hash_bcrypt", re.compile(r'^\$2[aby]\$\d{2}\$.{53}$')),
    ("hash_sha512", re.compile(r'^[a-f0-9]{128}$', re.IGNORECASE)),
    ("hash_sha256", re.compile(r'^[a-f0-9]{64}$',  re.IGNORECASE)),
    ("hash_sha1",   re.compile(r'^[a-f0-9]{40}$',  re.IGNORECASE)),
    ("hash_md5",    re.compile(r'^[a-f0-9]{32}$',  re.IGNORECASE)),
]


# ─────────────────────────────────────────────
# ENCRYPTION / KEY MARKERS
#
# FIX v3: Added EC, DSA, OpenSSH, PKCS8, EncryptedPKCS8, CertRequest.
# \s* between keywords tolerates OCR spacing gaps such as:
#   "BEGIN  EC  PRIVATE  KEY" instead of "BEGIN EC PRIVATE KEY"
# ─────────────────────────────────────────────

ENCRYPTION_MARKERS = [
    # ── PGP ──────────────────────────────────────────────────────────────────
    ("pgp_block",
     re.compile(r'BEGIN\s+PGP\s+(MESSAGE|ENCRYPTED\s+MESSAGE|SIGNED\s+MESSAGE)', re.IGNORECASE)),
    ("pgp_public_key",
     re.compile(r'BEGIN\s+PGP\s+PUBLIC\s+KEY', re.IGNORECASE)),
    ("pgp_private_key",
     re.compile(r'BEGIN\s+PGP\s+PRIVATE\s+KEY', re.IGNORECASE)),

    # ── Certificates ─────────────────────────────────────────────────────────
    ("cert_request",
     re.compile(r'BEGIN\s+CERTIFICATE\s+REQUEST', re.IGNORECASE)),
    ("ssl_cert",
     re.compile(r'BEGIN\s+CERTIFICATE', re.IGNORECASE)),          # after cert_request

    # ── RSA ───────────────────────────────────────────────────────────────────
    ("rsa_key",
     re.compile(r'BEGIN\s+RSA\s*(PRIVATE|PUBLIC)\s*KEY', re.IGNORECASE)),

    # ── FIX v3: EC private key (most common modern key — was completely missing) ──
    ("ec_private_key",
     re.compile(r'BEGIN\s*EC\s*PRIVATE\s*KEY', re.IGNORECASE)),

    # ── FIX v3: PKCS#8 generic PRIVATE KEY (wraps EC/RSA/DSA without explicit label) ──
    ("private_key_pkcs8",
     re.compile(r'BEGIN\s+PRIVATE\s+KEY', re.IGNORECASE)),

    # ── FIX v3: PKCS#8 password-protected encrypted private key ──────────────
    ("encrypted_private_key",
     re.compile(r'BEGIN\s+ENCRYPTED\s+PRIVATE\s+KEY', re.IGNORECASE)),

    # ── FIX v3: DSA private key ───────────────────────────────────────────────
    ("dsa_private_key",
     re.compile(r'BEGIN\s+DSA\s+PRIVATE\s+KEY', re.IGNORECASE)),

    # ── FIX v3: OpenSSH private key (Ed25519 / ECDSA modern format) ──────────
    ("openssh_private_key",
     re.compile(r'BEGIN\s+OPENSSH\s+PRIVATE\s+KEY', re.IGNORECASE)),
]

# Flat set of all technique names that count as "encrypted" for the flags dict
_ENCRYPTED_TECHNIQUES = {
    "pgp_block", "pgp_public_key", "pgp_private_key",
    "ssl_cert", "cert_request",
    "rsa_key", "ec_private_key", "private_key_pkcs8",
    "encrypted_private_key", "dsa_private_key", "openssh_private_key",
}


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def detect_all(text: str) -> dict:
    """
    Run all obfuscation detection methods against normalized OCR text.

    Args:
        text : Normalized text string from normalize_ocr_text() in ocr_engine.py.

    Returns:
        {
            "findings"     : list  — detailed finding dicts
            "has_encoded"  : bool  — base64 or hex tokens found and decoded
            "has_encrypted": bool  — PEM key / encrypted block found
            "has_hashes"   : bool  — hash patterns found
            "decoded_text" : str   — text with encoded tokens replaced by decoded values
            "escalate"     : bool  — True if ANY finding requires human review
        }
    """
    if not text or not text.strip():
        return _empty_result(text)

    print(f"\n[AntiObf] Scanning {len(text)} chars for obfuscation techniques...")

    findings     = []
    decoded_text = text

    enc_findings, decoded_text = _detect_encoding(text, decoded_text)
    hash_findings              = _detect_hashes(text)
    crypt_findings             = _detect_encryption_markers(text)

    findings.extend(enc_findings)
    findings.extend(hash_findings)
    findings.extend(crypt_findings)

    has_encoded   = any(f["technique"] in ("base64_encoded", "hex_encoded") for f in findings)
    has_encrypted = any(f["technique"] in _ENCRYPTED_TECHNIQUES for f in findings)
    has_hashes    = any(
        f["technique"].startswith("hash_") or f["technique"] == "partial_hash"
        for f in findings
    )

    # FIX v3: escalate on encrypted too — private keys must always escalate
    escalate = has_encrypted or has_hashes or has_encoded

    _print_summary(findings)

    return {
        "findings"     : findings,
        "has_encoded"  : has_encoded,
        "has_encrypted": has_encrypted,
        "has_hashes"   : has_hashes,
        "decoded_text" : decoded_text,
        "escalate"     : escalate,
    }


# ─────────────────────────────────────────────
# DETECTOR 1 — ENCODING (BASE64 + HEX)
# ─────────────────────────────────────────────

def _detect_encoding(text: str, decoded_text: str) -> tuple:
    findings = []
    tokens   = text.split()

    for token in tokens:
        clean = token.strip(".,;:\"'()[]{}")

        # BASE64: 20+ chars, valid base64 charset, optional = padding
        if len(clean) >= 20 and re.match(r'^[A-Za-z0-9+/]+=*$', clean):
            finding = _try_decode_base64(clean)
            if finding:
                findings.append(finding)
                if finding["decoded"]:
                    decoded_text = decoded_text.replace(token, finding["decoded"])
                    print(f"[AntiObf] Base64 decoded: '{clean[:12]}...' → '{finding['decoded'][:30]}'")
                continue

        # HEX: 8+ chars, even length, valid hex charset, not a known hash length
        if (len(clean) >= 8
                and len(clean) % 2 == 0
                and re.match(r'^[0-9a-fA-F]+$', clean)
                and not _is_likely_hash(clean)):
            finding = _try_decode_hex(clean)
            if finding:
                findings.append(finding)
                if finding["decoded"]:
                    decoded_text = decoded_text.replace(token, finding["decoded"])
                    print(f"[AntiObf] Hex decoded: '{clean[:12]}...' → '{finding['decoded'][:30]}'")

    return findings, decoded_text


def _try_decode_base64(token: str) -> dict | None:
    try:
        padded = token + "=" * (4 - len(token) % 4) if len(token) % 4 else token
        raw    = base64.b64decode(padded)
        try:
            decoded = raw.decode("utf-8")
            ratio   = sum(1 for c in decoded if c.isprintable()) / max(len(decoded), 1)
            if ratio > 0.85 and len(decoded.strip()) > 4:
                return {
                    "token"      : token,
                    "technique"  : "base64_encoded",
                    "decoded"    : decoded.strip(),
                    "recoverable": True,
                    "risk"       : "high",
                    "action"     : "regex_rescan",
                }
            return None
        except UnicodeDecodeError:
            return None
    except Exception:
        return None


def _try_decode_hex(token: str) -> dict | None:
    try:
        raw = binascii.unhexlify(token)
        try:
            decoded = raw.decode("utf-8")
            ratio   = sum(1 for c in decoded if c.isprintable()) / max(len(decoded), 1)
            if ratio > 0.85 and len(decoded.strip()) > 3:
                return {
                    "token"      : token,
                    "technique"  : "hex_encoded",
                    "decoded"    : decoded.strip(),
                    "recoverable": True,
                    "risk"       : "high",
                    "action"     : "regex_rescan",
                }
            return None
        except UnicodeDecodeError:
            return None
    except Exception:
        return None


# ─────────────────────────────────────────────
# DETECTOR 2 — HASHING
# ─────────────────────────────────────────────

def _detect_hashes(text: str) -> list:
    findings = []
    tokens   = text.split()

    # Strict full-length hash detection (token-by-token)
    for token in tokens:
        clean = token.strip(".,;:\"'()[]{}")
        if len(clean) < 32:
            continue
        for name, pattern in HASH_PATTERNS:
            if pattern.match(clean):
                print(f"[AntiObf] Hash detected: {name.upper()} → '{clean[:12]}...'")
                findings.append({
                    "token"      : clean,
                    "technique"  : name,
                    "decoded"    : None,
                    "recoverable": False,
                    "risk"       : "medium",
                    "action"     : "human_review",
                    "note"       : f"One-way hash ({name}) — plaintext unrecoverable.",
                })
                break

    # Partial hash detection — for OCR-broken or truncated hashes
    hex_candidates = re.findall(r'[a-f0-9]{20,}', text, re.IGNORECASE)
    for candidate in hex_candidates:
        length = len(candidate)
        if length in (32, 40, 64, 128):
            continue  # Already handled above
        if any(candidate in f["token"] for f in findings):
            continue
        print(f"[AntiObf] Partial hash detected → '{candidate[:12]}...' ({length} chars)")
        findings.append({
            "token"      : candidate,
            "technique"  : "partial_hash",
            "decoded"    : None,
            "recoverable": False,
            "risk"       : "medium",
            "action"     : "human_review",
            "note"       : "Possible truncated or OCR-broken hash value.",
        })

    return findings


# ─────────────────────────────────────────────
# DETECTOR 3 — ENCRYPTION / KEY MARKERS
# ─────────────────────────────────────────────

def _detect_encryption_markers(text: str) -> list:
    """
    Scan full text for PEM headers, PGP blocks, certificates.

    FIX v3: All key types now covered. Patterns use \s* for OCR tolerance.
    Order matters — more specific patterns (cert_request) placed before
    broader ones (ssl_cert) to avoid double-matching.
    """
    findings = []

    for name, pattern in ENCRYPTION_MARKERS:
        match = pattern.search(text)
        if match:
            print(f"[AntiObf] 🔑 Key/cert marker found: {name.upper()}")
            findings.append({
                "token"      : match.group()[:80],
                "technique"  : name,
                "decoded"    : None,
                "recoverable": False,
                "risk"       : "high",
                "action"     : "human_review",
                "note"       : (
                    f"Cryptographic key or certificate detected ({name}). "
                    "Immediate human review required."
                ),
            })

    return findings


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _is_likely_hash(token: str) -> bool:
    return len(token) in (32, 40, 64, 128)


def _empty_result(text: str) -> dict:
    return {
        "findings"     : [],
        "has_encoded"  : False,
        "has_encrypted": False,
        "has_hashes"   : False,
        "decoded_text" : text or "",
        "escalate"     : False,
    }


def _print_summary(findings: list) -> None:
    print(f"\n[AntiObf] ── Obfuscation Scan Summary ─────────────")
    if not findings:
        print("[AntiObf]  ✅ No obfuscation techniques detected.")
    else:
        print(f"[AntiObf]  ⚠️  {len(findings)} finding(s):")
        for f in findings:
            icon   = "🔑" if f["technique"] in _ENCRYPTED_TECHNIQUES else "⚠️ "
            status = "RECOVERED" if f.get("recoverable") else "HUMAN REVIEW REQUIRED"
            print(f"[AntiObf]    {icon} {f['technique']:<28} | {status:<22} | risk: {f['risk']}")
    print(f"[AntiObf] ─────────────────────────────────────")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import hashlib

    tests = [
        ("EC Private Key header (your test case)",
         "-----BEGIN EC PRIVATE KEY----- MHcCAQEEIGNDB1AYI5yJ -----END EC PRIVATE KEY-----"),
        ("RSA Private Key",
         "-----BEGIN RSA PRIVATE KEY----- abc123 -----END RSA PRIVATE KEY-----"),
        ("Generic PKCS8 PRIVATE KEY",
         "-----BEGIN PRIVATE KEY----- abc123 -----END PRIVATE KEY-----"),
        ("OpenSSH key",
         "-----BEGIN OPENSSH PRIVATE KEY----- b3BlbnNzaC -----END OPENSSH PRIVATE KEY-----"),
        ("PGP block",
         "-----BEGIN PGP MESSAGE----- hQEMA..."),
        ("Base64 encoded Aadhaar",
         f"Record: {base64.b64encode('9183 0074 6619'.encode()).decode()} admitted"),
        ("MD5 hash",
         f"Hash: {hashlib.md5('918300746619'.encode()).hexdigest()}"),
        ("SHA-256 hash",
         f"Ref: {hashlib.sha256('918300746619'.encode()).hexdigest()}"),
        ("No obfuscation",
         "Name: Rahul Sharma DOB: 01/01/1990"),
    ]

    for title, text in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {title}")
        print(f"Input: {text[:80]}")
        result = detect_all(text)
        print(f"  has_encrypted : {result['has_encrypted']}")
        print(f"  has_encoded   : {result['has_encoded']}")
        print(f"  has_hashes    : {result['has_hashes']}")
        print(f"  escalate      : {result['escalate']}")
        print(f"  findings      : {[f['technique'] for f in result['findings']]}")