# """
# Module 2b: anti_obfuscation.py
# -------------------------------
# Detects deliberate OCR-evasion techniques embedded in extracted text.

# Called by ocr_engine.py AFTER post-processing, BEFORE returning results.

# Architecture position:
#     ocr_engine._filter_words() → normalize_ocr_text()
#                                → anti_obfuscation.detect_all()   ← THIS MODULE
#                                → run_ocr() returns result dict

# WHAT THIS MODULE HANDLES:
#     1. Base64 encoding  — decode and recover plaintext (e.g. base64-encoded Aadhaar)
#     2. Hex encoding     — decode and recover plaintext (e.g. hex-encoded PAN number)
#     3. Encryption       — DETECT ONLY (AES, PGP, generic ciphertext). Cannot decrypt.
#                           Flags for human review.
#     4. Hashing          — DETECT ONLY (MD5, SHA-1, SHA-256, SHA-512, bcrypt).
#                           One-way function — plaintext is unrecoverable by design.
#                           Flags for human review.

# WHY DETECTION-ONLY FOR ENCRYPTION AND HASHING:
#     Encryption requires the secret key — we never have it in a DLP pipeline.
#     Hashing is a one-way function — mathematically irreversible.
#     The correct enterprise response for both is: FLAG → ESCALATE → HUMAN REVIEW.
#     Attempting to "crack" hashes or brute-force keys is out of scope for DLP.

# OUTPUT FORMAT:
#     {
#         "findings": [
#             {
#                 "token"      : str   — the original token from OCR text
#                 "technique"  : str   — "base64_encoded" | "hex_encoded" |
#                                        "encrypted" | "hash_md5" | "hash_sha1" etc.
#                 "decoded"    : str   — recovered plaintext (encoding only), else None
#                 "recoverable": bool  — True if plaintext recovered, False for enc/hash
#                 "risk"       : str   — "high" | "medium"
#                 "action"     : str   — "regex_rescan" | "human_review"
#             }
#         ],
#         "has_encoded"    : bool  — base64 or hex tokens found and decoded
#         "has_encrypted"  : bool  — possible encrypted content found
#         "has_hashes"     : bool  — hash patterns found
#         "decoded_text"   : str   — full normalized text with encoded tokens replaced
#                                    by their decoded equivalents. Feed THIS to regex detector.
#         "escalate"       : bool  — True if any non-recoverable content found
#     }
# """

# import re
# import base64
# import binascii
# import unicodedata


# # ─────────────────────────────────────────────
# # HASH PATTERNS
# # ─────────────────────────────────────────────
# # Ordered from most specific (bcrypt) to least specific (md5).
# # Most specific patterns run first to avoid false classification.

# HASH_PATTERNS = [
#     # bcrypt: $2a$12$<53 chars> — extremely specific format
#     ("hash_bcrypt", re.compile(r'^\$2[aby]\$\d{2}\$.{53}$')),

#     # SHA-512: 128 hex chars
#     ("hash_sha512", re.compile(r'^[a-f0-9]{128}$', re.IGNORECASE)),

#     # SHA-256: 64 hex chars
#     ("hash_sha256", re.compile(r'^[a-f0-9]{64}$', re.IGNORECASE)),

#     # SHA-1: 40 hex chars
#     # ⚠️ Also matches Git commit hashes, API key fragments — flagging these is
#     #    acceptable since all such tokens warrant review in a DLP context.
#     ("hash_sha1", re.compile(r'^[a-f0-9]{40}$', re.IGNORECASE)),

#     # MD5: 32 hex chars
#     # ⚠️ Also matches session tokens, UUIDs without hyphens — same note as SHA-1.
#     ("hash_md5", re.compile(r'^[a-f0-9]{32}$', re.IGNORECASE)),
# ]


# # ─────────────────────────────────────────────
# # ENCRYPTION MARKERS
# # ─────────────────────────────────────────────
# # These are text patterns that indicate encrypted content blocks.
# # OCR sometimes extracts PGP/GPG headers from document scans.

# ENCRYPTION_MARKERS = [
#     ("pgp_block",      re.compile(r'BEGIN\s+PGP\s+(MESSAGE|ENCRYPTED\s+MESSAGE|SIGNED\s+MESSAGE)', re.IGNORECASE)),
#     ("pgp_public_key", re.compile(r'BEGIN\s+PGP\s+PUBLIC\s+KEY', re.IGNORECASE)),
#     ("ssl_cert",       re.compile(r'BEGIN\s+CERTIFICATE', re.IGNORECASE)),
#     ("rsa_key",        re.compile(r'BEGIN\s+(RSA\s+)?(PRIVATE|PUBLIC)\s+KEY', re.IGNORECASE)),
# ]


# # ─────────────────────────────────────────────
# # MAIN ENTRY POINT
# # ─────────────────────────────────────────────

# def detect_all(text: str) -> dict:
#     """
#     Run all obfuscation detection methods against normalized OCR text.

#     Args:
#         text : Normalized text string from normalize_ocr_text() in ocr_engine.py.
#                Must be called AFTER normalization (homoglyphs already cleaned).

#     Returns:
#         Structured result dict — see module docstring for full format.
#     """
#     if not text or not text.strip():
#         return _empty_result(text)

#     print(f"\n[AntiObf] Scanning {len(text)} chars for obfuscation techniques...")

#     findings      = []
#     decoded_text  = text   # We'll substitute decoded values into this

#     # ── Run all detectors ────────────────────────────────────────────────────
#     enc_findings, decoded_text  = _detect_encoding(text, decoded_text)
#     hash_findings               = _detect_hashes(text)
#     crypt_findings              = _detect_encryption_markers(text)

#     findings.extend(enc_findings)
#     findings.extend(hash_findings)
#     findings.extend(crypt_findings)

#     has_encoded   = any(f["technique"] in ("base64_encoded", "hex_encoded") for f in findings)
#     has_encrypted = any(f["technique"] in ("encrypted", "pgp_block", "pgp_public_key",
#                                            "ssl_cert", "rsa_key") for f in findings)
#     has_hashes    = any(f["technique"].startswith("hash_") for f in findings)
#     escalate      = has_encrypted or has_hashes

#     _print_summary(findings)

#     return {
#         "findings"     : findings,
#         "has_encoded"  : has_encoded,
#         "has_encrypted": has_encrypted,
#         "has_hashes"   : has_hashes,
#         "decoded_text" : decoded_text,
#         "escalate"     : escalate,
#     }


# # ─────────────────────────────────────────────
# # DETECTOR 1 — ENCODING (BASE64 + HEX)
# # ─────────────────────────────────────────────

# def _detect_encoding(text: str, decoded_text: str) -> tuple:
#     """
#     Scan each token for base64 or hex encoding.
#     When decoding succeeds and yields printable text, substitute the decoded
#     value back into decoded_text so regex patterns in Module 3 can match it.

#     Returns:
#         (list of finding dicts, updated decoded_text string)
#     """
#     findings = []
#     tokens   = text.split()

#     for token in tokens:
#         clean = token.strip(".,;:\"'()[]{}")

#         # ── Base64 ────────────────────────────────────────────────────────────
#         # Criteria: 20+ chars, valid base64 charset, optional = padding
#         if len(clean) >= 20 and re.match(r'^[A-Za-z0-9+/]+=*$', clean):
#             finding = _try_decode_base64(clean)
#             if finding:
#                 findings.append(finding)
#                 if finding["recoverable"] and finding["decoded"]:
#                     # Replace the encoded token with decoded plaintext in the working text
#                     decoded_text = decoded_text.replace(token, finding["decoded"])
#                     print(f"[AntiObf] Base64 decoded: '{clean[:12]}...' → '{finding['decoded'][:30]}'")
#                 continue  # Don't also check as hex if base64 matched

#         # ── Hex ───────────────────────────────────────────────────────────────
#         # Criteria: 8+ chars, valid hex charset, even length (hex always pairs)
#         if (len(clean) >= 8
#                 and len(clean) % 2 == 0
#                 and re.match(r'^[0-9a-fA-F]+$', clean)
#                 and not _is_likely_hash(clean)):          # Don't double-count hashes
#             finding = _try_decode_hex(clean)
#             if finding:
#                 findings.append(finding)
#                 if finding["recoverable"] and finding["decoded"]:
#                     decoded_text = decoded_text.replace(token, finding["decoded"])
#                     print(f"[AntiObf] Hex decoded: '{clean[:12]}...' → '{finding['decoded'][:30]}'")

#     return findings, decoded_text


# def _try_decode_base64(token: str) -> dict | None:
#     """
#     Attempt base64 decoding. Distinguish between:
#         - Recoverable: decodes to readable UTF-8 text
#         - Encrypted:   decodes but yields non-printable binary (likely AES ciphertext)
#         - False positive: decoding fails entirely
#     """
#     try:
#         # Pad to valid base64 length if needed
#         padded = token + "=" * (4 - len(token) % 4) if len(token) % 4 else token
#         raw    = base64.b64decode(padded)

#         try:
#             decoded_str = raw.decode("utf-8")
#             # Check that decoded string contains actual readable content
#             printable_ratio = sum(1 for c in decoded_str if c.isprintable()) / max(len(decoded_str), 1)
#             if printable_ratio > 0.85 and len(decoded_str.strip()) > 4:
#                 return {
#                     "token"      : token,
#                     "technique"  : "base64_encoded",
#                     "decoded"    : decoded_str.strip(),
#                     "recoverable": True,
#                     "risk"       : "high",
#                     "action"     : "regex_rescan",
#                 }
#             else:
#                 # Decoded but unreadable → likely AES/binary ciphertext inside base64
#                 return {
#                     "token"      : token,
#                     "technique"  : "encrypted",
#                     "decoded"    : None,
#                     "recoverable": False,
#                     "risk"       : "high",
#                     "action"     : "human_review",
#                     "note"       : "Base64 wraps non-UTF8 binary — likely AES/binary ciphertext",
#                 }
#         except UnicodeDecodeError:
#             # Raw bytes are not UTF-8 → encrypted binary content
#             return {
#                 "token"      : token,
#                 "technique"  : "encrypted",
#                 "decoded"    : None,
#                 "recoverable": False,
#                 "risk"       : "high",
#                 "action"     : "human_review",
#                 "note"       : "Base64 wraps non-UTF8 bytes — possible AES encrypted block",
#             }

#     except Exception:
#         return None


# def _try_decode_hex(token: str) -> dict | None:
#     """
#     Attempt hex decoding. Same recoverable/encrypted distinction as base64.
#     """
#     try:
#         raw = binascii.unhexlify(token)
#         try:
#             decoded_str = raw.decode("utf-8")
#             printable_ratio = sum(1 for c in decoded_str if c.isprintable()) / max(len(decoded_str), 1)
#             if printable_ratio > 0.85 and len(decoded_str.strip()) > 3:
#                 return {
#                     "token"      : token,
#                     "technique"  : "hex_encoded",
#                     "decoded"    : decoded_str.strip(),
#                     "recoverable": True,
#                     "risk"       : "high",
#                     "action"     : "regex_rescan",
#                 }
#             else:
#                 return {
#                     "token"      : token,
#                     "technique"  : "encrypted",
#                     "decoded"    : None,
#                     "recoverable": False,
#                     "risk"       : "high",
#                     "action"     : "human_review",
#                     "note"       : "Hex wraps non-UTF8 binary — possible encrypted block",
#                 }
#         except UnicodeDecodeError:
#             return {
#                 "token"      : token,
#                 "technique"  : "encrypted",
#                 "decoded"    : None,
#                 "recoverable": False,
#                 "risk"       : "high",
#                 "action"     : "human_review",
#                 "note"       : "Hex wraps non-UTF8 bytes — possible encrypted block",
#             }
#     except Exception:
#         return None


# # ─────────────────────────────────────────────
# # DETECTOR 2 — HASHING
# # ─────────────────────────────────────────────

# def _detect_hashes(text: str) -> list:
#     """
#     Scan each token against known hash format patterns.

#     IMPORTANT — hashing is one-way. We can detect, never recover.
#     All hash findings carry action="human_review".

#     False positive awareness:
#         SHA-1 (40 hex) also matches Git commit hashes.
#         MD5  (32 hex) also matches session tokens, UUIDs without hyphens.
#         In a DLP context, flagging these is CORRECT — all such tokens
#         warrant human inspection regardless of their origin.
#     """
#     findings = []
#     tokens   = text.split()

#     for token in tokens:
#         clean = token.strip(".,;:\"'()[]{}")
#         if len(clean) < 32:
#             continue  # Shortest hash (MD5) is 32 chars — skip short tokens

#         for hash_name, pattern in HASH_PATTERNS:
#             if pattern.match(clean):
#                 print(f"[AntiObf] Hash detected: {hash_name.upper()} → '{clean[:12]}...'")
#                 findings.append({
#                     "token"      : clean,
#                     "technique"  : hash_name,
#                     "decoded"    : None,
#                     "recoverable": False,
#                     "risk"       : "medium",
#                     "action"     : "human_review",
#                     "note"       : (
#                         f"One-way hash ({hash_name}) — plaintext is unrecoverable. "
#                         "May be a hashed Aadhaar, PAN, or other sensitive identifier. "
#                         "Escalate for manual inspection."
#                     ),
#                 })
#                 break   # First match wins — don't double-flag same token

#     return findings


# # ─────────────────────────────────────────────
# # DETECTOR 3 — ENCRYPTION BLOCK MARKERS
# # ─────────────────────────────────────────────

# def _detect_encryption_markers(text: str) -> list:
#     """
#     Scan full text for PGP/SSL/RSA block headers.
#     These sometimes appear in scanned documents (printed private keys,
#     PGP-signed letters, certificates printed on paper).

#     All encryption findings carry action="human_review".
#     We cannot decrypt without the private key.
#     """
#     findings = []

#     for marker_name, pattern in ENCRYPTION_MARKERS:
#         match = pattern.search(text)
#         if match:
#             print(f"[AntiObf] Encryption marker found: {marker_name.upper()}")
#             findings.append({
#                 "token"      : match.group()[:60],
#                 "technique"  : marker_name,
#                 "decoded"    : None,
#                 "recoverable": False,
#                 "risk"       : "high",
#                 "action"     : "human_review",
#                 "note"       : (
#                     f"Encryption block header detected ({marker_name}). "
#                     "Document may contain a printed private key, certificate, "
#                     "or PGP-encrypted message. Immediate human review required."
#                 ),
#             })

#     return findings


# # ─────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────

# def _is_likely_hash(token: str) -> bool:
#     """
#     Returns True if a hex string looks like a hash (will be caught by _detect_hashes).
#     Prevents double-counting a SHA-256 as both hex-encoded AND a hash.
#     """
#     l = len(token)
#     return l in (32, 40, 64, 128)  # MD5, SHA-1, SHA-256, SHA-512


# def _empty_result(text: str) -> dict:
#     return {
#         "findings"     : [],
#         "has_encoded"  : False,
#         "has_encrypted": False,
#         "has_hashes"   : False,
#         "decoded_text" : text or "",
#         "escalate"     : False,
#     }


# def _print_summary(findings: list) -> None:
#     print(f"\n[AntiObf] ── Obfuscation Scan Summary {'─' * 25}")
#     if not findings:
#         print("[AntiObf]  ✅  No obfuscation techniques detected.")
#     else:
#         print(f"[AntiObf]  ⚠️   {len(findings)} obfuscation finding(s):")
#         for f in findings:
#             status = "RECOVERED" if f["recoverable"] else "HUMAN REVIEW REQUIRED"
#             print(f"[AntiObf]    → {f['technique']:<20} | {status:<22} | risk: {f['risk']}")
#     print(f"[AntiObf] {'─' * 55}")


# # ─────────────────────────────────────────────
# # QUICK TEST
# # ─────────────────────────────────────────────

# if __name__ == "__main__":

#     print("═" * 60)
#     print("TEST 1 — Base64 encoded Aadhaar")
#     print("═" * 60)
#     import base64 as b64
#     aadhaar_b64 = b64.b64encode("9183 0074 6619".encode()).decode()
#     text1 = f"Patient record: {aadhaar_b64} admitted 12/04/2024"
#     print(f"Input: {text1}")
#     r1 = detect_all(text1)
#     print(f"Decoded text: {r1['decoded_text']}")
#     print(f"Findings: {len(r1['findings'])} | Escalate: {r1['escalate']}\n")

#     print("═" * 60)
#     print("TEST 2 — Hex encoded PAN")
#     print("═" * 60)
#     pan_hex = "ABCDE1234F".encode().hex()
#     text2 = f"Account linked to {pan_hex} verified"
#     print(f"Input: {text2}")
#     r2 = detect_all(text2)
#     print(f"Decoded text: {r2['decoded_text']}")
#     print(f"Findings: {len(r2['findings'])} | Escalate: {r2['escalate']}\n")

#     print("═" * 60)
#     print("TEST 3 — MD5 hash of Aadhaar")
#     print("═" * 60)
#     import hashlib
#     aadhaar_md5 = hashlib.md5("918300746619".encode()).hexdigest()
#     text3 = f"User hash: {aadhaar_md5} stored in log"
#     print(f"Input: {text3}")
#     r3 = detect_all(text3)
#     print(f"Recoverable: {r3['findings'][0]['recoverable'] if r3['findings'] else 'N/A'}")
#     print(f"Action: {r3['findings'][0]['action'] if r3['findings'] else 'N/A'}")
#     print(f"Escalate: {r3['escalate']}\n")

#     print("═" * 60)
#     print("TEST 4 — SHA-256 hash")
#     print("═" * 60)
#     sha256 = hashlib.sha256("918300746619".encode()).hexdigest()
#     text4 = f"Reference: {sha256}"
#     print(f"Input: {text4}")
#     r4 = detect_all(text4)
#     print(f"Technique: {r4['findings'][0]['technique'] if r4['findings'] else 'N/A'}")
#     print(f"Escalate: {r4['escalate']}\n")

#     print("═" * 60)
#     print("TEST 5 — PGP block marker")
#     print("═" * 60)
#     text5 = "-----BEGIN PGP MESSAGE----- hQEMA..."
#     r5 = detect_all(text5)
#     print(f"Technique: {r5['findings'][0]['technique'] if r5['findings'] else 'N/A'}")
#     print(f"Escalate: {r5['escalate']}")




import re
import base64
import binascii


# ─────────────────────────────────────────────
# HASH PATTERNS
# ─────────────────────────────────────────────

HASH_PATTERNS = [
    ("hash_bcrypt", re.compile(r'^\$2[aby]\$\d{2}\$.{53}$')),
    ("hash_sha512", re.compile(r'^[a-f0-9]{128}$', re.IGNORECASE)),
    ("hash_sha256", re.compile(r'^[a-f0-9]{64}$', re.IGNORECASE)),
    ("hash_sha1", re.compile(r'^[a-f0-9]{40}$', re.IGNORECASE)),
    ("hash_md5", re.compile(r'^[a-f0-9]{32}$', re.IGNORECASE)),
]


# ─────────────────────────────────────────────
# ENCRYPTION MARKERS (STRICT ONLY)
# ─────────────────────────────────────────────

ENCRYPTION_MARKERS = [
    ("pgp_block", re.compile(r'BEGIN\s+PGP\s+(MESSAGE|ENCRYPTED\s+MESSAGE|SIGNED\s+MESSAGE)', re.IGNORECASE)),
    ("pgp_public_key", re.compile(r'BEGIN\s+PGP\s+PUBLIC\s+KEY', re.IGNORECASE)),
    ("ssl_cert", re.compile(r'BEGIN\s+CERTIFICATE', re.IGNORECASE)),
    ("rsa_key", re.compile(r'BEGIN\s+(RSA\s+)?(PRIVATE|PUBLIC)\s+KEY', re.IGNORECASE)),
]


# ─────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────

def detect_all(text: str) -> dict:
    if not text or not text.strip():
        return _empty_result(text)

    print(f"\n[AntiObf] Scanning {len(text)} chars for obfuscation techniques...")

    findings = []
    decoded_text = text

    enc_findings, decoded_text = _detect_encoding(text, decoded_text)
    hash_findings = _detect_hashes(text)
    crypt_findings = _detect_encryption_markers(text)

    findings.extend(enc_findings)
    findings.extend(hash_findings)
    findings.extend(crypt_findings)

    has_encoded = any(f["technique"] in ("base64_encoded", "hex_encoded") for f in findings)
    has_encrypted = any(f["technique"] in ("pgp_block", "pgp_public_key", "ssl_cert", "rsa_key") for f in findings)
    has_hashes = any(
    f["technique"].startswith("hash_") or f["technique"] == "partial_hash"
    for f in findings
)

    escalate = has_encrypted or has_hashes

    _print_summary(findings)

    return {
        "findings": findings,
        "has_encoded": has_encoded,
        "has_encrypted": has_encrypted,
        "has_hashes": has_hashes,
        "decoded_text": decoded_text,
        "escalate": escalate,
    }


# ─────────────────────────────────────────────
# ENCODING DETECTION (FIXED)
# ─────────────────────────────────────────────

def _detect_encoding(text: str, decoded_text: str):
    findings = []
    tokens = text.split()

    for token in tokens:
        clean = token.strip(".,;:\"'()[]{}")

        # BASE64
        if len(clean) >= 20 and re.match(r'^[A-Za-z0-9+/]+=*$', clean):
            finding = _try_decode_base64(clean)
            if finding:
                findings.append(finding)
                if finding["decoded"]:
                    decoded_text = decoded_text.replace(token, finding["decoded"])
                    print(f"[AntiObf] Base64 decoded: '{clean[:12]}...'")
                continue

        # HEX
        if (len(clean) >= 8
                and len(clean) % 2 == 0
                and re.match(r'^[0-9a-fA-F]+$', clean)
                and not _is_likely_hash(clean)):
            finding = _try_decode_hex(clean)
            if finding:
                findings.append(finding)
                if finding["decoded"]:
                    decoded_text = decoded_text.replace(token, finding["decoded"])
                    print(f"[AntiObf] Hex decoded: '{clean[:12]}...'")

    return findings, decoded_text


# ─────────────────────────────────────────────
# BASE64 (FIXED — NO FALSE ENCRYPTION)
# ─────────────────────────────────────────────

def _try_decode_base64(token: str):
    try:
        padded = token + "=" * (4 - len(token) % 4) if len(token) % 4 else token
        raw = base64.b64decode(padded)

        try:
            decoded = raw.decode("utf-8")

            printable_ratio = sum(1 for c in decoded if c.isprintable()) / max(len(decoded), 1)

            if printable_ratio > 0.85 and len(decoded.strip()) > 4:
                return {
                    "token": token,
                    "technique": "base64_encoded",
                    "decoded": decoded.strip(),
                    "recoverable": True,
                    "risk": "high",
                    "action": "regex_rescan",
                }

            return None  # ❌ DO NOT mark encrypted

        except UnicodeDecodeError:
            return None  # ❌ DO NOT mark encrypted

    except Exception:
        return None


# ─────────────────────────────────────────────
# HEX (FIXED)
# ─────────────────────────────────────────────

def _try_decode_hex(token: str):
    try:
        raw = binascii.unhexlify(token)

        try:
            decoded = raw.decode("utf-8")

            printable_ratio = sum(1 for c in decoded if c.isprintable()) / max(len(decoded), 1)

            if printable_ratio > 0.85 and len(decoded.strip()) > 3:
                return {
                    "token": token,
                    "technique": "hex_encoded",
                    "decoded": decoded.strip(),
                    "recoverable": True,
                    "risk": "high",
                    "action": "regex_rescan",
                }

            return None

        except UnicodeDecodeError:
            return None

    except Exception:
        return None


# ─────────────────────────────────────────────
# HASH DETECTION
# ─────────────────────────────────────────────

def _detect_hashes(text: str) -> list:
    findings = []
    tokens = text.split()

    # ─────────────────────────────
    # 1. STRICT HASH DETECTION
    # ─────────────────────────────
    for token in tokens:
        clean = token.strip(".,;:\"'()[]{}")

        if len(clean) < 32:
            continue

        for name, pattern in HASH_PATTERNS:
            if pattern.match(clean):
                print(f"[AntiObf] Hash detected: {name.upper()} → '{clean[:12]}...'")
                findings.append({
                    "token": clean,
                    "technique": name,
                    "decoded": None,
                    "recoverable": False,
                    "risk": "medium",
                    "action": "human_review",
                })
                break

    # ─────────────────────────────
    # 2. PARTIAL HASH DETECTION (NEW)
    # ─────────────────────────────

    # find long hex-like sequences in full text
    hex_candidates = re.findall(r'[a-f0-9]{20,}', text, re.IGNORECASE)

    for candidate in hex_candidates:
        length = len(candidate)

        # skip if already detected as full hash
        if length in (32, 40, 64, 128):
            continue

        # avoid duplicate detection
        if any(candidate in f["token"] for f in findings):
            continue

        print(f"[AntiObf] Partial hash detected → '{candidate[:12]}...' ({length} chars)")

        findings.append({
            "token": candidate,
            "technique": "partial_hash",
            "decoded": None,
            "recoverable": False,
            "risk": "medium",
            "action": "human_review",
            "note": "Possible truncated or OCR-broken hash value",
        })

    return findings

# ─────────────────────────────────────────────
# ENCRYPTION MARKERS (STRICT)
# ─────────────────────────────────────────────

def _detect_encryption_markers(text: str):
    findings = []

    for name, pattern in ENCRYPTION_MARKERS:
        match = pattern.search(text)
        if match:
            print(f"[AntiObf] Encryption marker found: {name}")
            findings.append({
                "token": match.group(),
                "technique": name,
                "decoded": None,
                "recoverable": False,
                "risk": "high",
                "action": "human_review",
            })

    return findings


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _is_likely_hash(token: str):
    return len(token) in (32, 40, 64, 128)


def _empty_result(text: str):
    return {
        "findings": [],
        "has_encoded": False,
        "has_encrypted": False,
        "has_hashes": False,
        "decoded_text": text or "",
        "escalate": False,
    }


def _print_summary(findings):
    print(f"\n[AntiObf] ── Obfuscation Scan Summary ─────────────")
    if not findings:
        print("[AntiObf]  ✅ No obfuscation techniques detected.")
    else:
        for f in findings:
            print(f"[AntiObf]  → {f['technique']}")
    print(f"[AntiObf] ─────────────────────────────────────")