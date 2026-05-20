"""
modules/vision_classifier.py
----------------------------
Phase 3 Vision Layer - image-level sensitivity classification.

WHAT THIS MODULE DOES:
    OCR and regex/NLP work on extracted text. This module looks at the image
    itself so the pipeline can still flag likely sensitive documents when OCR
    is sparse, noisy, angled, handwritten, or adversarially obfuscated.

    Three tasks:
        Task A - Document Type Classification
            Classifies the image into one of:
            id_card / passport / cheque / document / screenshot / unknown

        Task B - Low-Text OCR Failure Detection
            Flags images where OCR returned few words even though the image
            contains structured document-like content.

        Task C - Sensitive Region Heatmap
            Optional CLIP patch scoring hook. If CLIP is unavailable, returns
            no heatmap and never blocks the pipeline.

BACKEND STRATEGY (Python 3.13 safe):
    Priority 1: CLIP if torch + clip are installed and load cleanly
    Priority 2: torchvision/timm hooks if torch is installed
    Priority 3: OpenCV heuristics (default on this project)

The OpenCV fallback is intentionally strong enough to be the normal backend on
Windows Python 3.13 where torch is not installed.
"""

import os
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# OPTIONAL BACKEND DETECTION - all imports are graceful
# ---------------------------------------------------------------------------

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        import clip
        CLIP_AVAILABLE = True
    except Exception:
        clip = None
        CLIP_AVAILABLE = False
else:
    clip = None
    CLIP_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        import torchvision.models as tv_models
        TORCHVISION_AVAILABLE = True
    except Exception:
        tv_models = None
        TORCHVISION_AVAILABLE = False
else:
    tv_models = None
    TORCHVISION_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        import timm
        TIMM_AVAILABLE = True
    except Exception:
        timm = None
        TIMM_AVAILABLE = False
else:
    timm = None
    TIMM_AVAILABLE = False


if CLIP_AVAILABLE:
    print("[Vision] Backend candidate: CLIP available")
elif TORCHVISION_AVAILABLE:
    print("[Vision] Backend candidate: torchvision available")
elif TIMM_AVAILABLE:
    print("[Vision] Backend candidate: timm available")
else:
    print("[Vision] No torch/CLIP/timm backend available.")
    print("[Vision] Using OpenCV heuristics; pipeline remains fully functional.")


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

VALID_DOCUMENT_TYPES = {
    "id_card", "passport", "cheque", "document", "screenshot", "unknown"
}

SENSITIVITY_BY_TYPE = {
    "id_card"   : "HIGH",
    "passport"  : "HIGH",
    "cheque"    : "CRITICAL",
    "document"  : "MEDIUM",
    "screenshot": "MEDIUM",
    "unknown"   : "LOW",
}

CONFIDENCE_LABELS = [
    (0.75, "high"),
    (0.45, "medium"),
    (0.00, "low"),
]


# ---------------------------------------------------------------------------
# SINGLETON MODEL LOADING
# ---------------------------------------------------------------------------

_model = None
_preprocess = None
_model_used = None
_model_load_attempted = False


def _get_model():
    """Load an optional vision backend once and cache it."""
    global _model, _preprocess, _model_used, _model_load_attempted

    if _model_load_attempted:
        return _model

    _model_load_attempted = True

    if CLIP_AVAILABLE:
        try:
            print("[Vision] Loading CLIP model on CPU...")
            _model, _preprocess = clip.load("ViT-B/32", device="cpu")
            _model.eval()
            _model_used = "clip_vit_b32_cpu"
            print("[Vision] CLIP loaded on CPU.")
            return _model
        except Exception as e:
            print(f"[Vision] CLIP load error: {e}")

    if TORCHVISION_AVAILABLE:
        try:
            print("[Vision] torchvision available; using OpenCV classifier for document taxonomy.")
            _model_used = "opencv_heuristics"
            return None
        except Exception as e:
            print(f"[Vision] torchvision setup error: {e}")

    if TIMM_AVAILABLE:
        try:
            print("[Vision] timm available; using OpenCV classifier for document taxonomy.")
            _model_used = "opencv_heuristics"
            return None
        except Exception as e:
            print(f"[Vision] timm setup error: {e}")

    _model_used = "opencv_heuristics"
    return None


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def classify_image(image_path: str, ocr_word_count: int = 0) -> dict:
    """
    Classify an image for Phase 3 vision DLP.

    Args:
        image_path      : Path to the original image.
        ocr_word_count  : Number of OCR words retained by ocr_engine.

    Returns:
        {
            "document_type"     : str,
            "type_confidence"   : float,
            "sensitivity_level" : str,
            "ocr_failure_risk"  : bool,
            "vision_findings"   : list,
            "model_used"        : str,
            "fallback"          : bool,
        }
    """
    print(f"\n[Vision] Starting image classification: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"[Vision] Could not load image: {image_path}")
        return _build_result(
            document_type="unknown",
            type_confidence=0.0,
            ocr_failure_risk=False,
            model_used="opencv_heuristics",
            fallback=True,
        )

    model = _get_model()
    fallback = model is None or _model_used == "opencv_heuristics"

    try:
        stats = _compute_image_statistics(image)
        document_type, confidence = _classify_with_opencv(stats)
        ocr_failure_risk = _detect_ocr_failure_risk(
            ocr_word_count=ocr_word_count,
            stats=stats,
        )

        if model is not None and CLIP_AVAILABLE:
            clip_type, clip_conf = _classify_with_clip(image_path)
            if clip_type != "unknown" and clip_conf >= confidence:
                document_type, confidence = clip_type, clip_conf
                fallback = False

        print(
            f"[Vision] Document type={document_type} "
            f"confidence={confidence:.2f} "
            f"sensitivity={SENSITIVITY_BY_TYPE[document_type]} "
            f"ocr_failure_risk={ocr_failure_risk}"
        )

        return _build_result(
            document_type=document_type,
            type_confidence=confidence,
            ocr_failure_risk=ocr_failure_risk,
            model_used=_model_used or "opencv_heuristics",
            fallback=fallback,
        )

    except Exception as e:
        print(f"[Vision] Classification error: {e}")
        return _build_result(
            document_type="unknown",
            type_confidence=0.0,
            ocr_failure_risk=False,
            model_used="opencv_heuristics",
            fallback=True,
        )


# ---------------------------------------------------------------------------
# TASK A - DOCUMENT TYPE CLASSIFICATION
# ---------------------------------------------------------------------------

def _compute_image_statistics(image: np.ndarray) -> dict:
    """Extract OpenCV signals used by the fallback classifier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    area = max(1, h * w)
    aspect_ratio = w / max(1, h)

    contrast = float(gray.std())
    brightness = float(gray.mean())

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 160)
    edge_density = float(np.count_nonzero(edges)) / float(area)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 11
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangular_regions = 0
    text_like_regions = 0
    large_regions = 0

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        c_area = cw * ch
        if c_area < max(20, area * 0.0002):
            continue
        fill_ratio = cv2.contourArea(contour) / float(max(1, c_area))
        rect_aspect = cw / max(1, ch)

        if c_area > area * 0.015:
            large_regions += 1
        if 0.20 <= fill_ratio <= 0.95 and c_area > area * 0.002:
            rectangular_regions += 1
        if 1.5 <= rect_aspect <= 18 and 4 <= ch <= max(12, h * 0.12):
            text_like_regions += 1

    structured_score = (
        min(edge_density / 0.08, 1.0) * 0.40 +
        min(rectangular_regions / 10.0, 1.0) * 0.30 +
        min(text_like_regions / 18.0, 1.0) * 0.30
    )

    return {
        "width"              : w,
        "height"             : h,
        "aspect_ratio"       : aspect_ratio,
        "contrast"           : contrast,
        "brightness"         : brightness,
        "edge_density"       : edge_density,
        "rectangular_regions": rectangular_regions,
        "text_like_regions"  : text_like_regions,
        "large_regions"      : large_regions,
        "structured_score"   : round(float(structured_score), 3),
    }


def _classify_with_opencv(stats: dict) -> tuple:
    """
    Estimate document type using aspect ratio and structure signals.
    Returns (document_type, confidence).
    """
    aspect = stats["aspect_ratio"]
    edges = stats["edge_density"]
    rects = stats["rectangular_regions"]
    text_regions = stats["text_like_regions"]
    structured = stats["structured_score"]
    contrast = stats["contrast"]

    if structured < 0.14 and edges < 0.025:
        return "unknown", _clamp_confidence(0.20 + structured)

    if aspect >= 2.15 and structured >= 0.15 and edges >= 0.025:
        confidence = 0.62 + min(0.25, structured * 0.25) + min(0.10, text_regions / 80)
        return "cheque", _clamp_confidence(confidence)

    if 1.35 <= aspect <= 1.85 and structured >= 0.15 and edges >= 0.025:
        confidence = 0.55 + min(0.25, structured * 0.30) + min(0.10, rects / 60)
        return "id_card", _clamp_confidence(confidence)

    if 0.62 <= aspect <= 0.88 and structured >= 0.15 and edges >= 0.025:
        confidence = 0.52 + min(0.25, structured * 0.28) + min(0.10, text_regions / 70)
        return "passport", _clamp_confidence(confidence)

    if aspect >= 1.55 and structured >= 0.25 and stats["large_regions"] >= 4:
        confidence = 0.48 + min(0.28, structured * 0.30) + min(0.12, text_regions / 90)
        return "screenshot", _clamp_confidence(confidence)

    if structured >= 0.25 or (contrast >= 28 and text_regions >= 4):
        confidence = 0.42 + min(0.25, structured * 0.35) + min(0.10, text_regions / 100)
        return "document", _clamp_confidence(confidence)

    return "unknown", _clamp_confidence(0.25 + structured * 0.30)


def _classify_with_clip(image_path: str) -> tuple:
    """
    Optional CLIP zero-shot classifier.
    If anything goes wrong, return unknown so OpenCV remains authoritative.
    """
    if not CLIP_AVAILABLE or _model is None or _preprocess is None:
        return "unknown", 0.0

    try:
        from PIL import Image

        prompts = [
            "a photo of an identity card",
            "a photo of a passport",
            "a photo of a bank cheque",
            "a photo of a sensitive printed document",
            "a computer screenshot",
            "an unrelated image",
        ]
        labels = ["id_card", "passport", "cheque", "document", "screenshot", "unknown"]

        image = _preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
        text = clip.tokenize(prompts)

        with torch.no_grad():
            image_features = _model.encode_image(image)
            text_features = _model.encode_text(text)
            logits = (image_features @ text_features.T).softmax(dim=-1)
            scores = logits.cpu().numpy()[0]

        idx = int(np.argmax(scores))
        return labels[idx], _clamp_confidence(float(scores[idx]))
    except Exception as e:
        print(f"[Vision] CLIP inference error: {e}")
        return "unknown", 0.0


# ---------------------------------------------------------------------------
# TASK B - LOW-TEXT OCR FAILURE DETECTION
# ---------------------------------------------------------------------------

def _detect_ocr_failure_risk(ocr_word_count: int, stats: dict) -> bool:
    """
    OCR failure risk means text was sparse but the image looks structured.
    This is the key compensating signal for Phase 3.
    """
    if ocr_word_count >= 10:
        return False

    structured = stats["structured_score"] >= 0.15
    enough_edges = stats["edge_density"] >= 0.025
    enough_regions = (
        stats["rectangular_regions"] >= 3 or
        stats["text_like_regions"] >= 4 or
        stats["large_regions"] >= 1
    )

    return bool(structured and enough_edges and enough_regions)


# ---------------------------------------------------------------------------
# TASK C - HEATMAP HOOK
# ---------------------------------------------------------------------------

def _build_sensitive_region_heatmap(image_path: str) -> list:
    """
    Optional extension point for CLIP patch scoring.
    The required Phase 3 output does not include heatmap data, so this returns
    an empty list unless a future caller explicitly uses it.
    """
    if not CLIP_AVAILABLE:
        return []
    return []


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _build_result(
    document_type: str,
    type_confidence: float,
    ocr_failure_risk: bool,
    model_used: str,
    fallback: bool,
) -> dict:
    document_type = document_type if document_type in VALID_DOCUMENT_TYPES else "unknown"
    type_confidence = _clamp_confidence(type_confidence)
    sensitivity_level = SENSITIVITY_BY_TYPE[document_type]

    findings = []
    if document_type != "unknown" or ocr_failure_risk:
        findings.append({
            "type"               : "document_type",
            "value"              : document_type,
            "tokens"             : [],
            "confidence"         : _confidence_label(type_confidence),
            "source"             : "vision",
            "vision_confidence"  : type_confidence,
            "sensitivity_level"  : sensitivity_level,
            "ocr_failure_risk"   : ocr_failure_risk,
            "fp_risk"            : False,
            # Evidence-fusion metadata: lets _merge_vision_result gate solo vision findings
            "vision_is_fallback" : fallback,
            "vision_model_used"  : model_used,
            # A solo vision finding should never exceed MEDIUM without OCR corroboration
            "vision_solo_capped" : True,
        })

    return {
        "document_type"    : document_type,
        "type_confidence"  : type_confidence,
        "sensitivity_level": sensitivity_level,
        "ocr_failure_risk" : bool(ocr_failure_risk),
        "vision_findings"  : findings,
        "model_used"       : model_used or "opencv_heuristics",
        "fallback"         : bool(fallback),
    }


def _confidence_label(score: float) -> str:
    for threshold, label in CONFIDENCE_LABELS:
        if score >= threshold:
            return label
    return "low"


def _clamp_confidence(value: float) -> float:
    try:
        return round(float(min(1.0, max(0.0, value))), 3)
    except Exception:
        return 0.0


def is_vision_model_available() -> bool:
    """Return whether a non-OpenCV backend can be loaded in this environment."""
    return bool(CLIP_AVAILABLE or TORCHVISION_AVAILABLE or TIMM_AVAILABLE)


def get_vision_backend() -> str:
    if CLIP_AVAILABLE:
        return "clip"
    if TORCHVISION_AVAILABLE:
        return "torchvision"
    if TIMM_AVAILABLE:
        return "timm"
    return "opencv_heuristics"


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  Vision Classifier - Quick Tests")
    print(f"  Backend: {get_vision_backend()}")
    print("=" * 60)

    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            result = classify_image(path, ocr_word_count=0)
            print(result)
    else:
        print("Usage: python modules/vision_classifier.py <image1> [image2 ...]")