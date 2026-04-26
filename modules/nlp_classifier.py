"""
modules/nlp_classifier.py
--------------------------
Phase 2 NLP Layer — contextual sensitivity classification with dual NER backend.

WHAT THIS MODULE DOES:
    Regex (Phase 1) finds patterns by shape.
    NLP (Phase 2) understands meaning and context.

    Three tasks:
        Task A — Context Verification (no model needed)
            Re-evaluates every regex match using surrounding words.
            "12/04/1990" near "Date of Birth" → upgrade confidence
            "12/04/1990" near "Invoice"       → flag as likely FP

        Task B — Named Entity Recognition (NER)
            Detects person names and org names — invisible to regex.

        Task C — Document Sensitivity Label Detection (no model needed)
            Scans for CONFIDENTIAL, RESTRICTED, INTERNAL USE ONLY, etc.

NER BACKEND — AUTO-SELECTED (Python 3.13 compatible):
    Priority 1: spaCy (en_core_web_sm)
        - Lightweight ~12MB model
        - Installs cleanly on Python 3.13
        - Install: python -m pip install spacy
                   python -m spacy download en_core_web_sm
        - Fast: ~0.1s per image on CPU

    Priority 2: HuggingFace BERT (dslim/bert-base-NER)
        - Higher accuracy, ~400MB model
        - Requires Python <=3.12 for stable torch support
        - Install: python -m pip install transformers torch
        - Slower: ~2-4s per image on CPU

    Priority 3: No NER (graceful fallback)
        - Tasks A and C still run fully
        - Regex results preserved unchanged
        - System remains fully functional

INSTALL GUIDE (Python 3.13 — use spaCy):
    python -m pip install spacy
    python -m spacy download en_core_web_sm

INSTALL GUIDE (Python <=3.12 — use BERT):
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install transformers
"""

import os
from typing import Optional


# ─────────────────────────────────────────────
# BACKEND DETECTION — priority: spaCy > BERT > None
# ─────────────────────────────────────────────

# Try spaCy first (Python 3.13 compatible, lightweight)
try:
    import spacy
    _spacy_model = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("[NLP] NER backend: spaCy (en_core_web_sm) ✓")
except ImportError:
    SPACY_AVAILABLE = False
    _spacy_model   = None
except OSError:
    SPACY_AVAILABLE = False
    _spacy_model   = None
    print("[NLP] spaCy installed but model not found.")
    print("[NLP] Run: python -m spacy download en_core_web_sm")

# Try HuggingFace transformers as fallback
if not SPACY_AVAILABLE:
    try:
        from transformers import pipeline as hf_pipeline
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        TRANSFORMERS_AVAILABLE = True
        print("[NLP] NER backend: HuggingFace BERT ✓")
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
else:
    TRANSFORMERS_AVAILABLE = False

# Final availability flag
NER_AVAILABLE = SPACY_AVAILABLE or TRANSFORMERS_AVAILABLE

if not NER_AVAILABLE:
    print("[NLP] No NER backend available.")
    print("[NLP] For Python 3.13 (recommended):")
    print("[NLP]   python -m pip install spacy")
    print("[NLP]   python -m spacy download en_core_web_sm")
    print("[NLP] Tasks A and C (context + doc labels) still run fully.")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BERT_MODEL       = "dslim/bert-base-NER"
NER_CONFIDENCE   = 0.80    # minimum BERT confidence (spaCy uses all entities)
CONTEXT_WINDOW   = 8       # words either side of match for context check

# Context keywords confirming a match is truly sensitive
SENSITIVE_CONTEXT = {
    "dob"            : {"born", "birth", "dob", "date", "birthday", "age"},
    "phone"          : {"phone", "mobile", "contact", "call", "tel", "number"},
    "bank_card"      : {"card", "credit", "debit", "account", "bank", "payment"},
    "aadhaar"        : {"aadhaar", "uid", "unique", "identification"},
    "passport_number": {"passport", "travel", "visa", "document", "no", "number"},
    "micr_code"      : {"micr", "cheque", "check", "bank", "branch"},
    "driving_licence": {"licence", "license", "driving", "dl", "vehicle"},
    "voter_id"       : {"voter", "election", "epic", "electoral"},
}

# Context keywords suggesting a false positive
BENIGN_CONTEXT = {
    "invoice", "receipt", "order", "transaction", "ref",
    "reference", "expires", "expiry", "serial", "version",
    "batch", "lot", "item", "product", "code", "tracking",
    "shipment", "delivery", "po", "purchase"
}

# Document-level sensitivity classification markers
DOCUMENT_LABELS = [
    "CONFIDENTIAL",
    "STRICTLY CONFIDENTIAL",
    "RESTRICTED",
    "INTERNAL USE ONLY",
    "INTERNAL ONLY",
    "NOT FOR DISTRIBUTION",
    "SENSITIVE",
    "CLASSIFIED",
    "FOR OFFICIAL USE ONLY",
    "PRIVATE AND CONFIDENTIAL",
    "TOP SECRET",
    "COMMERCIAL IN CONFIDENCE",
]

# spaCy entity types → DLP types
SPACY_ENTITY_MAP = {
    "PERSON": "person",    # person names
    "ORG"   : "org_name",  # organisation names
}

# BERT entity types → DLP types
BERT_ENTITY_MAP = {
    "PER": "person",
    "ORG": "org_name",
}


# ─────────────────────────────────────────────
# BERT MODEL LOADER — singleton
# ─────────────────────────────────────────────

_bert_pipeline = None

def _get_bert_pipeline():
    """Load BERT NER pipeline once and cache it."""
    global _bert_pipeline
    if _bert_pipeline is not None:
        return _bert_pipeline
    if not TRANSFORMERS_AVAILABLE:
        return None
    print(f"[NLP] Loading BERT model: {BERT_MODEL} (first run downloads ~400MB)...")
    try:
        _bert_pipeline = hf_pipeline(
            task                 = "ner",
            model                = BERT_MODEL,
            aggregation_strategy = "simple",
            device               = -1,
        )
        print(f"[NLP] BERT model loaded on CPU.")
        return _bert_pipeline
    except Exception as e:
        print(f"[NLP] BERT load error: {e}")
        return None


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def classify(
    ocr_text       : str,
    regex_result   : dict,
    run_ner        : bool = True,
    run_doc_labels : bool = True,
    run_context    : bool = True
) -> dict:
    """
    Run NLP enrichment on OCR text and regex findings.

    Args:
        ocr_text       : Full OCR text from ocr_engine.run_ocr()["text"]
        regex_result   : Output from sensitive_detector.detect_sensitive()
        run_ner        : Run NER for person/org detection (default True)
        run_doc_labels : Scan for document sensitivity labels (default True)
        run_context    : Run context verification on regex matches (default True)

    Returns:
        {
            "is_sensitive"    : bool
            "total"           : int
            "matches"         : list — all findings (regex + NLP enriched)
            "new_findings"    : list — findings added by NLP only
            "context_flags"   : list — regex matches flagged as likely FP
            "nlp_available"   : bool — whether any NER backend ran
            "ner_backend"     : str  — "spacy" | "bert" | "none"
        }
    """
    if not ocr_text or not ocr_text.strip():
        print("[NLP] Empty text — skipping NLP classification.")
        return _wrap_regex_result(regex_result, ner_backend="none")

    backend = ("spacy" if SPACY_AVAILABLE
               else "bert" if TRANSFORMERS_AVAILABLE
               else "none")

    print(f"\n[NLP] Starting NLP classification | backend={backend} | "
          f"{len(ocr_text)} chars...")

    new_findings  = []
    context_flags = []

    # Start with all existing regex matches
    enriched = list(regex_result.get("matches", []))

    # ── Task A: Context Verification ─────────────────────────────────────────
    if run_context:
        enriched, context_flags = _run_context_verification(enriched, ocr_text)

    # ── Task B: NER ───────────────────────────────────────────────────────────
    if run_ner and NER_AVAILABLE:
        if SPACY_AVAILABLE:
            new_findings.extend(_run_ner_spacy(ocr_text))
        elif TRANSFORMERS_AVAILABLE:
            new_findings.extend(_run_ner_bert(ocr_text))

    # ── Task C: Document labels ───────────────────────────────────────────────
    if run_doc_labels:
        new_findings.extend(_run_document_label_detection(ocr_text))

    all_matches  = enriched + new_findings
    is_sensitive = len(all_matches) > 0

    _print_summary(new_findings, context_flags, backend)

    return {
        "is_sensitive" : is_sensitive,
        "total"        : len(all_matches),
        "matches"      : all_matches,
        "new_findings" : new_findings,
        "context_flags": context_flags,
        "nlp_available": NER_AVAILABLE,
        "ner_backend"  : backend,
    }


# ─────────────────────────────────────────────
# TASK A — CONTEXT VERIFICATION
# ─────────────────────────────────────────────

def _run_context_verification(matches: list, ocr_text: str) -> tuple:
    """
    Check surrounding words for each regex match.
    Upgrades low/medium confidence if sensitive context found.
    Flags matches as FP risk if benign context found.
    Never removes a match — safety first.
    """
    print(f"[NLP] Task A: Context verification on {len(matches)} regex matches...")

    enriched    = []
    flagged_fps = []

    for match in matches:
        m = dict(match)
        context_words = _get_context_words(ocr_text, m["value"], CONTEXT_WINDOW)
        context_lower = {w.lower().strip(".,;:") for w in context_words}

        # Check benign context
        benign_overlap = context_lower & BENIGN_CONTEXT
        if benign_overlap:
            m["fp_risk"]        = True
            m["fp_reason"]      = f"Benign context words: {benign_overlap}"
            m["nlp_confidence"] = "low"
            flagged_fps.append(m)
            print(f"[NLP]   ⚠ FP risk: {m['type']} '{m['value']}' "
                  f"← benign: {benign_overlap}")
        else:
            m["fp_risk"] = False

        # Check sensitive context → upgrade confidence
        sensitive_kw      = SENSITIVE_CONTEXT.get(m["type"], set())
        sensitive_overlap = context_lower & sensitive_kw

        if sensitive_overlap and m["confidence"] == "low":
            m["confidence"]     = "medium"
            m["nlp_confidence"] = "medium"
            m["upgraded_by"]    = f"Context: {sensitive_overlap}"
            print(f"[NLP]   ↑ Upgraded: {m['type']} '{m['value']}' "
                  f"low→medium ← {sensitive_overlap}")
        elif sensitive_overlap and m["confidence"] == "medium":
            m["nlp_confidence"] = "high"
            m["upgraded_by"]    = f"Context: {sensitive_overlap}"
            print(f"[NLP]   ↑ Confirmed: {m['type']} '{m['value']}' "
                  f"← strong context: {sensitive_overlap}")
        else:
            m.setdefault("nlp_confidence", m["confidence"])

        enriched.append(m)

    return enriched, flagged_fps


def _get_context_words(text: str, match_value: str, window: int) -> list:
    """Extract words surrounding a match value."""
    words       = text.split()
    match_words = match_value.split()

    for i in range(len(words) - len(match_words) + 1):
        if words[i:i + len(match_words)] == match_words:
            start = max(0, i - window)
            end   = min(len(words), i + len(match_words) + window)
            return words[start:i] + words[i + len(match_words):end]

    # Fallback character-based
    idx = text.find(match_value[:6])
    if idx == -1:
        return []
    start = max(0, idx - 60)
    end   = min(len(text), idx + len(match_value) + 60)
    return text[start:end].split()


# ─────────────────────────────────────────────
# TASK B — NER (spaCy backend)
# ─────────────────────────────────────────────

def _run_ner_spacy(ocr_text: str) -> list:
    """
    Run spaCy NER to detect PERSON and ORG entities.

    spaCy en_core_web_sm:
        - Detects: PERSON, ORG, GPE, DATE, MONEY, etc.
        - We use PERSON and ORG only for DLP purposes
        - Fast: ~0.1s on CPU, 12MB model
        - Installs cleanly on Python 3.13

    Why spaCy is sufficient for Phase 2:
        For the DLP use case (finding person names and org names in
        scanned documents), spaCy's accuracy is production-ready.
        BERT gives marginally higher accuracy on edge cases but
        the difference is small for structured document text.
    """
    if _spacy_model is None:
        return []

    print(f"[NLP] Task B: Running spaCy NER...")

    try:
        # Truncate to avoid memory issues on very long OCR texts
        text_input = ocr_text[:2000] if len(ocr_text) > 2000 else ocr_text
        doc        = _spacy_model(text_input)
    except Exception as e:
        print(f"[NLP] spaCy inference error: {e}")
        return []

    findings = []
    seen     = set()

    for ent in doc.ents:
        label = ent.label_
        word  = ent.text.strip()

        if label not in SPACY_ENTITY_MAP:
            continue
        if len(word) < 2:
            continue
        if word.lower() in {"the", "a", "an", "of", "in", "on"}:
            continue
        if word in seen:
            continue

        seen.add(word)
        finding_type = SPACY_ENTITY_MAP[label]

        findings.append({
            "type"          : finding_type,
            "value"         : word,
            "tokens"        : word.split(),
            "confidence"    : "medium",
            "nlp_confidence": "spacy",
            "source"        : "nlp",
            "ner_label"     : label,
            "ner_backend"   : "spacy",
            "fp_risk"       : False,
        })
        print(f"[NLP]   ✓ {finding_type.upper():<18} → '{word}'  "
              f"[spaCy/{label}]")

    print(f"[NLP] Task B: {len(findings)} NER finding(s) via spaCy.")
    return findings


# ─────────────────────────────────────────────
# TASK B — NER (BERT backend fallback)
# ─────────────────────────────────────────────

def _run_ner_bert(ocr_text: str) -> list:
    """
    Run BERT NER (HuggingFace dslim/bert-base-NER).
    Used when spaCy is not available (Python <= 3.12 with torch installed).
    """
    ner = _get_bert_pipeline()
    if ner is None:
        return []

    print(f"[NLP] Task B: Running BERT NER...")

    try:
        text_input = ocr_text[:1000] if len(ocr_text) > 1000 else ocr_text
        entities   = ner(text_input)
    except Exception as e:
        print(f"[NLP] BERT inference error: {e}")
        return []

    findings = []
    seen     = set()

    for entity in entities:
        label = entity.get("entity_group", "")
        score = entity.get("score", 0)
        word  = entity.get("word", "").strip()

        if label not in BERT_ENTITY_MAP:
            continue
        if score < NER_CONFIDENCE:
            continue
        if len(word) < 2 or word in seen:
            continue

        seen.add(word)
        finding_type = BERT_ENTITY_MAP[label]

        findings.append({
            "type"          : finding_type,
            "value"         : word,
            "tokens"        : word.split(),
            "confidence"    : "medium" if score >= 0.90 else "low",
            "nlp_confidence": f"{score:.2f}",
            "source"        : "nlp",
            "ner_label"     : label,
            "ner_backend"   : "bert",
            "fp_risk"       : False,
        })
        print(f"[NLP]   ✓ {finding_type.upper():<18} → '{word}'  "
              f"[bert/{label} score={score:.2f}]")

    print(f"[NLP] Task B: {len(findings)} NER finding(s) via BERT.")
    return findings


# ─────────────────────────────────────────────
# TASK C — DOCUMENT LABEL DETECTION
# ─────────────────────────────────────────────

def _run_document_label_detection(ocr_text: str) -> list:
    """
    Scan for document-level sensitivity classification markers.
    No model needed — pure string matching.
    """
    print(f"[NLP] Task C: Scanning for document sensitivity labels...")

    findings   = []
    text_upper = ocr_text.upper()
    seen       = set()

    for label in DOCUMENT_LABELS:
        if label.upper() in text_upper and label not in seen:
            seen.add(label)
            idx   = text_upper.find(label.upper())
            value = ocr_text[idx:idx + len(label)]
            findings.append({
                "type"          : "document_label",
                "value"         : value,
                "tokens"        : value.split(),
                "confidence"    : "high",
                "nlp_confidence": "high",
                "source"        : "nlp",
                "fp_risk"       : False,
            })
            print(f"[NLP]   ✓ DOCUMENT_LABEL      → '{value}'  [high]")

    if not findings:
        print(f"[NLP] Task C: No document sensitivity labels found.")

    return findings


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _wrap_regex_result(regex_result: dict, ner_backend: str = "none") -> dict:
    """Wrap regex result when NLP is skipped — keeps downstream structure intact."""
    matches = list(regex_result.get("matches", []))
    for m in matches:
        m.setdefault("fp_risk", False)
        m.setdefault("nlp_confidence", m.get("confidence", "low"))
    return {
        "is_sensitive" : regex_result.get("is_sensitive", False),
        "total"        : len(matches),
        "matches"      : matches,
        "new_findings" : [],
        "context_flags": [],
        "nlp_available": False,
        "ner_backend"  : ner_backend,
    }


def _print_summary(new_findings: list, context_flags: list, backend: str) -> None:
    print(f"\n[NLP] ── NLP Summary {'─' * 35}")
    print(f"[NLP]   NER backend      : {backend}")
    print(f"[NLP]   New NLP findings : {len(new_findings)}")
    print(f"[NLP]   FP risk flags    : {len(context_flags)}")
    for f in new_findings:
        conf = f.get("nlp_confidence", f.get("confidence", "?"))
        print(f"[NLP]   + {f['type']:<18} → '{f['value']}'  [{conf}]")
    for f in context_flags:
        print(f"[NLP]   ⚠ FP risk: {f['type']:<14} → '{f['value']}'")
    print(f"[NLP] {'─' * 54}")


def is_nlp_available() -> bool:
    return NER_AVAILABLE

def get_ner_backend() -> str:
    if SPACY_AVAILABLE:     return "spacy"
    if TRANSFORMERS_AVAILABLE: return "bert"
    return "none"


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from modules.sensitive_detector import detect_sensitive

    print("═" * 60)
    print(f"  NLP Classifier — Quick Tests")
    print(f"  NER backend: {get_ner_backend()}")
    print("═" * 60)

    # Test 1: Context verification — DOB upgrade
    print("\n── Test 1: DOB upgraded near 'Date of Birth' ──")
    text1   = "Name John Smith Date of Birth 12/04/1990 Place Mumbai"
    regex1  = detect_sensitive(text1)
    result1 = classify(text1, regex1, run_ner=False, run_doc_labels=False)
    dob     = next((m for m in result1["matches"] if m["type"] == "dob"), None)
    if dob:
        print(f"DOB confidence: {dob['confidence']} | upgraded_by: {dob.get('upgraded_by','—')}")

    # Test 2: Benign context FP flag
    print("\n── Test 2: Invoice date flagged as FP ──")
    text2   = "Invoice date 12/04/1990 Order ref 45231"
    regex2  = detect_sensitive(text2)
    result2 = classify(text2, regex2, run_ner=False, run_doc_labels=False)
    print(f"FP flags: {len(result2['context_flags'])}")

    # Test 3: Document label
    print("\n── Test 3: CONFIDENTIAL label detected ──")
    text3   = "CONFIDENTIAL — HR Employee record DOB 12/04/1990"
    regex3  = detect_sensitive(text3)
    result3 = classify(text3, regex3, run_ner=False)
    labels  = [m["value"] for m in result3["new_findings"]
               if m["type"] == "document_label"]
    print(f"Labels: {labels}")

    # Test 4: Full NER (if backend available)
    if NER_AVAILABLE:
        print(f"\n── Test 4: Full NER via {get_ner_backend()} ──")
        text4 = (
            "REPUBLIC OF INDIA Passport\n"
            "Name RAMADUGULA SITA MAHA LAKSHAI\n"
            "Date of Birth 11/10/1990\n"
            "Passport No J8369854"
        )
        regex4  = detect_sensitive(text4)
        result4 = classify(text4, regex4, run_ner=True, run_doc_labels=True)
        print(f"\nTotal findings: {result4['total']}")
        print(f"NLP findings  : {len(result4['new_findings'])}")
        for f in result4["matches"]:
            src = f.get("source", "regex")
            print(f"  [{src}] {f['type']:<18} → '{f['value']}'")
    else:
        print(f"\n── Test 4: NER skipped — no backend ──")
        print("Install spaCy for Python 3.13:")
        print("  python -m pip install spacy")
        print("  python -m spacy download en_core_web_sm")