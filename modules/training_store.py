"""
modules/training_store.py
--------------------------
Labeled training data storage for future NLP fine-tuning.

PURPOSE (Barclays Feedback):
    "Are we storing sensitive and non-sensitive data so the NLP
     can train on it for future classifications?"

    Answer: Yes. Every pipeline scan produces labeled examples.
    This module captures them in a structured, fine-tuning-ready format.

HOW IT FITS IN THE PIPELINE:
    After reporter.py saves the encrypted report, training_store.py
    saves a sanitised, labeled version of each finding to disk.
    These records accumulate over time and form a growing dataset
    that can be used to fine-tune BERT on domain-specific examples.

THREE STORES:
    confirmed_sensitive.jsonl  → analyst-confirmed true positives
    dismissed_findings.jsonl   → analyst-confirmed false positives
                                 (regex fired but was wrong)
    unreviewed.jsonl           → pending analyst review
                                 (auto-saved after every scan)

LABELING FLOW:
    Every scan → records go to unreviewed.jsonl (label="pending")
    Analyst reviews → calls label_finding() to move to confirmed/dismissed
    Batch auto-label → high-confidence regex matches auto-labeled as sensitive
                       (reduces manual review burden)

TRAINING RECORD SCHEMA:
    {
        record_id       : unique ID for this training example
        source_image    : filename of the scanned image
        finding_type    : pan / email / dob / person / document_label / etc.
        finding_value   : MASKED — never stores raw sensitive value
        ocr_text_window : ±10 words around the finding (context for NLP training)
        full_ocr_text   : complete OCR output (for document-level models)
        context_words   : list of surrounding words
        regex_confidence: high / medium / low / none (if NLP-only finding)
        nlp_confidence  : high / medium / low / none (if regex-only)
        label           : sensitive / not_sensitive / pending
        label_source    : analyst / auto_high_conf / auto_rule
        labeled_at      : UTC timestamp
        phase           : which pipeline phase produced this record
        source          : regex / nlp / vision
        notes           : free text for analyst comments
    }

PRIVACY DESIGN:
    finding_value is ALWAYS masked before storage.
    The training store contains NO raw sensitive values.
    It stores context windows and labels only — safe to use for training
    without exposing the original sensitive data.
"""

import os
import json
import hashlib
import re
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

TRAINING_DIR = os.path.join("output", "training_data")

STORE_PATHS = {
    "confirmed"  : os.path.join(TRAINING_DIR, "confirmed_sensitive.jsonl"),
    "dismissed"  : os.path.join(TRAINING_DIR, "dismissed_findings.jsonl"),
    "unreviewed" : os.path.join(TRAINING_DIR, "unreviewed.jsonl"),
}

# Stats file — tracks counts per store, per type
STATS_FILE = os.path.join(TRAINING_DIR, "dataset_stats.json")


# ─────────────────────────────────────────────
# CONFIDENCE AUTO-LABEL RULES
# ─────────────────────────────────────────────
# Findings matching these rules are auto-labeled
# without requiring analyst review.
# Reduces manual workload for obvious cases.

AUTO_LABEL_SENSITIVE = {
    # Pattern types where high confidence = auto-label as sensitive
    "high_confidence_types": {
        "pan", "passport_number", "gst_number",
        "ifsc_code", "swift_bic", "email"
    },
    # NLP-detected entities always auto-labeled
    "nlp_types": {
        "person", "document_label"
    }
}

AUTO_LABEL_NOT_SENSITIVE = {
    # Pattern types where context strongly suggests false positive
    # Used when NLP context engine flags benign context
    "benign_context_keywords": {
        "invoice", "receipt", "order", "transaction",
        "ref", "reference", "expires", "expiry", "serial"
    }
}


# ─────────────────────────────────────────────
# MAIN ENTRY POINT — called by pipeline
# ─────────────────────────────────────────────

def save_scan_records(
    image_path       : str,
    detection_result : dict,
    ocr_text         : str,
    nlp_result       : Optional[dict] = None
) -> dict:
    """
    Save training records for all findings from a single scan.
    Called automatically by main.py after every pipeline run.

    Args:
        image_path       : Path to the scanned image.
        detection_result : Output from sensitive_detector.detect_sensitive().
        ocr_text         : Full OCR text string from ocr_engine.run_ocr().
        nlp_result       : Optional output from nlp_classifier (Phase 2).
                           If None, only regex findings are stored.

    Returns:
        {
            "saved"      : int — number of records saved
            "auto_labeled": int — records auto-labeled (no review needed)
            "pending"    : int — records needing analyst review
            "store_path" : str — path to unreviewed store
        }
    """
    os.makedirs(TRAINING_DIR, exist_ok=True)

    image_name  = os.path.basename(image_path)
    saved       = 0
    auto_labeled = 0
    pending     = 0

    # Combine regex + NLP matches into one list
    all_matches = list(detection_result.get("matches", []))
    if nlp_result:
        all_matches.extend(nlp_result.get("new_findings", []))

    if not all_matches:
        # Save a "clean image" record — also valuable for training
        record = _build_record(
            image_name    = image_name,
            match         = None,
            ocr_text      = ocr_text,
            is_clean_image = True
        )
        _write_record(record)
        print(f"[TrainingStore] Clean image record saved (label=not_sensitive).")
        return {"saved": 1, "auto_labeled": 1, "pending": 0,
                "store_path": STORE_PATHS["confirmed"]}

    for match in all_matches:
        record = _build_record(
            image_name = image_name,
            match      = match,
            ocr_text   = ocr_text
        )

        # Determine label
        label, label_source = _auto_label(match, ocr_text)
        record["label"]        = label
        record["label_source"] = label_source

        # Route to correct store
        _write_record(record)
        saved += 1

        if label == "pending":
            pending += 1
        else:
            auto_labeled += 1

    # Update stats
    _update_stats()

    print(f"[TrainingStore] Saved {saved} record(s) — "
          f"{auto_labeled} auto-labeled, {pending} pending review.")

    return {
        "saved"       : saved,
        "auto_labeled": auto_labeled,
        "pending"     : pending,
        "store_path"  : STORE_PATHS["unreviewed"]
    }


# ─────────────────────────────────────────────
# ANALYST LABELING
# ─────────────────────────────────────────────

def label_finding(
    record_id : str,
    label     : str,
    notes     : str = ""
) -> bool:
    """
    Analyst labels a pending finding as sensitive or not_sensitive.
    Moves the record from unreviewed to confirmed or dismissed store.

    Args:
        record_id : The record_id of the finding to label.
        label     : "sensitive" or "not_sensitive"
        notes     : Optional analyst notes (e.g. "false positive — invoice date")

    Returns:
        True if record found and moved, False if not found.

    Usage:
        from modules.training_store import label_finding
        label_finding("abc123def456", "not_sensitive", "invoice date, not DOB")
        label_finding("xyz789", "sensitive", "confirmed passport number")
    """
    if label not in ("sensitive", "not_sensitive"):
        raise ValueError(f"label must be 'sensitive' or 'not_sensitive', got '{label}'")

    # Find the record in unreviewed store
    record = _find_and_remove(record_id, STORE_PATHS["unreviewed"])

    if record is None:
        print(f"[TrainingStore] Record '{record_id}' not found in unreviewed store.")
        return False

    # Update label fields
    record["label"]        = label
    record["label_source"] = "analyst"
    record["labeled_at"]   = datetime.now(timezone.utc).isoformat()
    record["notes"]        = notes

    # Route to correct store
    target = "confirmed" if label == "sensitive" else "dismissed"
    _append_to_store(record, STORE_PATHS[target])
    _update_stats()

    print(f"[TrainingStore] '{record_id[:12]}...' labeled as '{label}' "
          f"→ moved to {target} store.")
    return True


def bulk_label(labels: list) -> dict:
    """
    Label multiple findings at once.
    Useful for batch analyst review sessions.

    Args:
        labels : List of dicts, each with:
                 {"record_id": str, "label": str, "notes": str (optional)}

    Returns:
        {"success": int, "not_found": int}

    Usage:
        bulk_label([
            {"record_id": "abc123", "label": "sensitive"},
            {"record_id": "def456", "label": "not_sensitive", "notes": "invoice"},
        ])
    """
    success   = 0
    not_found = 0

    for item in labels:
        result = label_finding(
            record_id = item["record_id"],
            label     = item["label"],
            notes     = item.get("notes", "")
        )
        if result:
            success += 1
        else:
            not_found += 1

    print(f"[TrainingStore] Bulk label: {success} labeled, {not_found} not found.")
    return {"success": success, "not_found": not_found}


# ─────────────────────────────────────────────
# DATASET EXPORT — for fine-tuning
# ─────────────────────────────────────────────

def export_training_dataset(
    output_path   : str = None,
    include_auto  : bool = True,
    min_per_type  : int = 0
) -> dict:
    """
    Export the full labeled dataset in a format ready for BERT fine-tuning.

    Combines confirmed_sensitive + dismissed_findings into one dataset.
    Optionally filters by minimum examples per type.

    Output format (one JSON line per record):
        {
            "text"        : context window text (input to model)
            "label"       : 1 (sensitive) or 0 (not_sensitive)
            "finding_type": original pattern type
            "source"      : regex / nlp / vision
        }

    Args:
        output_path  : Where to save. Defaults to training_data/export_<date>.jsonl
        include_auto : Include auto-labeled records (default True).
                       Set False to export only analyst-labeled records.
        min_per_type : Minimum examples per type to include in export.
                       Types with fewer examples are excluded.

    Returns:
        {
            "total"     : total records exported
            "sensitive" : count of sensitive examples
            "not_sensitive": count of not_sensitive examples
            "by_type"   : per-type counts
            "path"      : output file path
        }
    """
    os.makedirs(TRAINING_DIR, exist_ok=True)

    if output_path is None:
        date_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(TRAINING_DIR, f"export_{date_str}.jsonl")

    records    = []
    by_type    = {}

    # Load confirmed + dismissed
    for store_key in ("confirmed", "dismissed"):
        for record in _read_store(STORE_PATHS[store_key]):
            if not include_auto and record.get("label_source") != "analyst":
                continue
            records.append(record)
            ftype = record.get("finding_type", "unknown")
            by_type[ftype] = by_type.get(ftype, 0) + 1

    # Apply min_per_type filter
    if min_per_type > 0:
        valid_types = {t for t, c in by_type.items() if c >= min_per_type}
        records     = [r for r in records if r.get("finding_type") in valid_types]

    if not records:
        print("[TrainingStore] No labeled records to export yet.")
        return {"total": 0, "sensitive": 0, "not_sensitive": 0,
                "by_type": {}, "path": None}

    # Write export file
    sensitive     = 0
    not_sensitive = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            is_sensitive = record["label"] == "sensitive"
            export_record = {
                "text"        : record.get("ocr_text_window", ""),
                "label"       : 1 if is_sensitive else 0,
                "finding_type": record.get("finding_type", "unknown"),
                "source"      : record.get("source", "regex"),
                "record_id"   : record.get("record_id", ""),
            }
            f.write(json.dumps(export_record) + "\n")
            if is_sensitive:
                sensitive += 1
            else:
                not_sensitive += 1

    print(f"[TrainingStore] Exported {len(records)} records → {output_path}")
    print(f"[TrainingStore] Sensitive: {sensitive} | Not sensitive: {not_sensitive}")

    return {
        "total"        : len(records),
        "sensitive"    : sensitive,
        "not_sensitive": not_sensitive,
        "by_type"      : by_type,
        "path"         : output_path
    }


# ─────────────────────────────────────────────
# STATS AND REPORTING
# ─────────────────────────────────────────────

def get_dataset_stats() -> dict:
    """
    Return current dataset statistics.
    Shows how many labeled examples exist per store and per type.
    Useful for deciding when you have enough data to start fine-tuning.

    Rule of thumb for fine-tuning BERT:
        50+ examples per type = baseline fine-tuning possible
        200+ examples per type = reliable fine-tuning
        500+ examples per type = production-grade fine-tuning
    """
    stats = {
        "confirmed_sensitive" : 0,
        "dismissed_findings"  : 0,
        "unreviewed"          : 0,
        "total"               : 0,
        "by_type"             : {},
        "readiness"           : {}
    }

    for store_key, path in STORE_PATHS.items():
        records = _read_store(path)
        count   = len(records)
        stats[store_key if store_key != "confirmed" else "confirmed_sensitive"] = count

        for record in records:
            ftype = record.get("finding_type", "unknown")
            stats["by_type"][ftype] = stats["by_type"].get(ftype, 0) + 1

    stats["total"] = (stats["confirmed_sensitive"] +
                      stats["dismissed_findings"] +
                      stats["unreviewed"])

    # Fine-tuning readiness assessment
    for ftype, count in stats["by_type"].items():
        if count >= 500:
            readiness = "production_ready"
        elif count >= 200:
            readiness = "reliable"
        elif count >= 50:
            readiness = "baseline"
        else:
            readiness = f"insufficient ({count}/50 minimum)"
        stats["readiness"][ftype] = readiness

    return stats


def print_dataset_stats() -> None:
    """Print a formatted dataset statistics report to console."""
    stats = get_dataset_stats()

    print(f"\n{'═' * 55}")
    print(f"  Training Dataset Statistics")
    print(f"{'─' * 55}")
    print(f"  Confirmed sensitive  : {stats['confirmed_sensitive']}")
    print(f"  Dismissed (false+)   : {stats['dismissed_findings']}")
    print(f"  Unreviewed (pending) : {stats['unreviewed']}")
    print(f"  Total records        : {stats['total']}")

    if stats["by_type"]:
        print(f"{'─' * 55}")
        print(f"  {'Type':<22} {'Count':>6}  Readiness")
        print(f"  {'─'*22} {'─'*6}  {'─'*20}")
        for ftype, count in sorted(stats["by_type"].items()):
            readiness = stats["readiness"].get(ftype, "")
            print(f"  {ftype:<22} {count:>6}  {readiness}")

    print(f"{'═' * 55}\n")


# ─────────────────────────────────────────────
# RECORD BUILDER
# ─────────────────────────────────────────────

def _build_record(
    image_name     : str,
    match          : Optional[dict],
    ocr_text       : str,
    is_clean_image : bool = False
) -> dict:
    """
    Build a single training record from a detection match.

    Privacy rule: finding_value is always masked before storage.
    The training store never contains raw sensitive values.
    Context window and labels are what the NLP model actually needs.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if is_clean_image or match is None:
        return {
            "record_id"        : _generate_id(image_name + timestamp),
            "source_image"     : image_name,
            "finding_type"     : "clean",
            "finding_value"    : "N/A",
            "ocr_text_window"  : ocr_text[:300] if ocr_text else "",
            "full_ocr_text"    : ocr_text,
            "context_words"    : [],
            "regex_confidence" : "none",
            "nlp_confidence"   : "none",
            "label"            : "not_sensitive",
            "label_source"     : "auto_rule",
            "labeled_at"       : timestamp,
            "phase"            : "phase_1_regex",
            "source"           : "auto",
            "notes"            : "Clean image — no sensitive content detected",
        }

    # Extract context window (±10 words around the match value)
    context_window, context_words = _extract_context(
        ocr_text, match["value"], window=10
    )

    # Mask the finding value — never store raw
    masked_value = _mask_for_training(match["value"], match["type"])

    return {
        "record_id"        : _generate_id(image_name + match["value"] + timestamp),
        "source_image"     : image_name,
        "finding_type"     : match["type"],
        "finding_value"    : masked_value,          # MASKED — no raw value stored
        "ocr_text_window"  : context_window,        # context for NLP training
        "full_ocr_text"    : ocr_text,              # full document text
        "context_words"    : context_words,         # surrounding word list
        "regex_confidence" : match.get("confidence", "none"),
        "nlp_confidence"   : match.get("nlp_confidence", "none"),
        "label"            : "pending",             # overwritten by auto-labeler
        "label_source"     : "pending",
        "labeled_at"       : timestamp,
        "phase"            : "phase_3_vision" if match.get("source") == "vision"
                             else "phase_2_nlp" if match.get("source") == "nlp"
                             else "phase_1_regex",
        "source"           : match.get("source", "regex"),
        "notes"            : "",
    }


# ─────────────────────────────────────────────
# AUTO LABELER
# ─────────────────────────────────────────────

def _auto_label(match: dict, ocr_text: str) -> tuple:
    """
    Attempt to auto-label a match without analyst review.

    Rules (in priority order):
        1. NLP-detected entities → auto sensitive
        2. High-confidence regex types → auto sensitive
        3. Benign context keywords near the match → auto not_sensitive
        4. Everything else → pending (needs analyst review)

    Returns:
        (label, label_source) tuple
    """
    ptype  = match.get("type", "")
    source = match.get("source", "regex")
    conf   = match.get("confidence", "low")

    # Rule 1: NLP entity types → auto sensitive
    if ptype in AUTO_LABEL_SENSITIVE["nlp_types"]:
        return "sensitive", "auto_high_conf"

    # Rule 1b: Vision document classifications indicate image-level sensitivity.
    if source == "vision" and ptype == "document_type" and conf in ("high", "medium"):
        return "sensitive", "auto_high_conf"

    # Rule 2: High-confidence regex types → auto sensitive
    if (source == "regex" and
            conf == "high" and
            ptype in AUTO_LABEL_SENSITIVE["high_confidence_types"]):
        return "sensitive", "auto_high_conf"

    # Rule 3: Check context for benign keywords
    value = match.get("value", "")
    _, context_words = _extract_context(ocr_text, value, window=5)
    context_lower = {w.lower() for w in context_words}

    benign_overlap = context_lower & AUTO_LABEL_NOT_SENSITIVE["benign_context_keywords"]
    if benign_overlap:
        return "not_sensitive", "auto_rule"

    # Default: needs review
    return "pending", "pending"


# ─────────────────────────────────────────────
# CONTEXT WINDOW EXTRACTION
# ─────────────────────────────────────────────

def _extract_context(
    ocr_text    : str,
    match_value : str,
    window      : int = 10
) -> tuple:
    """
    Extract a window of words around a matched value in the OCR text.

    This context window is the primary INPUT to the NLP model during training.
    Surrounding words tell the model WHY something is or isn't sensitive:
        "Date of Birth 12/04/1990 Place of Birth" → context confirms sensitivity
        "Invoice date 12/04/1990 Order ref 4521"  → context suggests not sensitive

    Args:
        ocr_text    : Full OCR text string.
        match_value : The matched sensitive value to find.
        window      : Number of words to take on each side.

    Returns:
        (context_string, context_words_list)
    """
    if not ocr_text or not match_value:
        return "", []

    words      = ocr_text.split()
    match_words = match_value.split()

    # Find match start position in word list
    match_start = -1
    for i in range(len(words) - len(match_words) + 1):
        if words[i:i + len(match_words)] == match_words:
            match_start = i
            break

    if match_start == -1:
        # Match not found as exact word sequence — return surrounding text
        # This happens with OCR variations — still useful context
        start = max(0, ocr_text.find(match_value[:4]) - 50)
        end   = min(len(ocr_text), start + len(match_value) + 100)
        snippet = ocr_text[start:end]
        return snippet, snippet.split()

    match_end     = match_start + len(match_words)
    context_start = max(0, match_start - window)
    context_end   = min(len(words), match_end + window)

    context_words  = words[context_start:context_end]
    context_string = " ".join(context_words)

    return context_string, context_words


# ─────────────────────────────────────────────
# FILE I/O
# ─────────────────────────────────────────────

def _write_record(record: dict) -> None:
    """Route a record to the correct store based on its label."""
    label = record.get("label", "pending")

    if label == "sensitive":
        path = STORE_PATHS["confirmed"]
    elif label == "not_sensitive":
        path = STORE_PATHS["dismissed"]
    else:
        path = STORE_PATHS["unreviewed"]

    _append_to_store(record, path)


def _append_to_store(record: dict, path: str) -> None:
    """Append one record to a JSONL store file."""
    os.makedirs(TRAINING_DIR, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_store(path: str) -> list:
    """Read all records from a JSONL store file."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _find_and_remove(record_id: str, path: str) -> Optional[dict]:
    """
    Find a record by ID in a JSONL file and remove it.
    Used when moving records from unreviewed to confirmed/dismissed.
    Rewrites the file without the found record.
    """
    if not os.path.exists(path):
        return None

    records = _read_store(path)
    found   = None
    remaining = []

    for record in records:
        if record.get("record_id") == record_id:
            found = record
        else:
            remaining.append(record)

    if found is None:
        return None

    # Rewrite store without the found record
    with open(path, "w", encoding="utf-8") as f:
        for record in remaining:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return found


def _update_stats() -> None:
    """Refresh the stats file after any write operation."""
    stats = get_dataset_stats()
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


# ─────────────────────────────────────────────
# MASKING FOR TRAINING STORE
# ─────────────────────────────────────────────

def _mask_for_training(value: str, pattern_type: str) -> str:
    """
    Mask sensitive value before storing in training data.

    Different from reporter masking — more aggressive here
    because training data may be shared with ML engineers
    who don't need the actual values, only the context.

    Format: <TYPE:HASH> — type label + short hash of value
    This lets us identify if the same value appears multiple times
    across scans without ever storing the value itself.
    """
    value_hash = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"<{pattern_type.upper()}:{value_hash}>"


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _generate_id(seed: str) -> str:
    """Generate a unique 32-char hex record ID."""
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from modules.sensitive_detector import detect_sensitive

    print("── Test 1: Save scan records from a detection result ──")
    sample_text = (
        "Date of Birth 12/04/1990 "
        "Passport No J8369854 "
        "sample@gmail.com "
        "ABCDE1234F "
        "Invoice date 01/01/2024"
    )
    detection = detect_sensitive(sample_text)
    result    = save_scan_records(
        image_path       = "input/test_passport.jpg",
        detection_result = detection,
        ocr_text         = sample_text
    )
    print(f"Result: {result}")

    print("\n── Test 2: Dataset statistics ──")
    print_dataset_stats()

    print("\n── Test 3: Analyst labeling ──")
    unreviewed = _read_store(STORE_PATHS["unreviewed"])
    if unreviewed:
        rid = unreviewed[0]["record_id"]
        print(f"Labeling first unreviewed record: {rid[:12]}...")
        label_finding(rid, "sensitive", "Confirmed — passport DOB")
        print_dataset_stats()
    else:
        print("No unreviewed records to label.")

    print("\n── Test 4: Export training dataset ──")
    export = export_training_dataset()
    print(f"Export result: {export}")
