"""
Module 1: preprocess.py
-----------------------
Cleans and prepares an input image for OCR.
Handles: scanned documents, screenshots, ID cards, scene photos.

No ML. No cloud. Just OpenCV.

CHANGES (Anti-Obfuscation Update):
    1. _split_colour_channels() added — returns R, G, B channel images separately.
       Defends against text hidden in a single colour channel (a known DLP evasion).
    2. preprocess_image() now also returns channel_images dict alongside binary + scale_factor.
       ocr_engine.py uses these to run OCR on each channel and merge results.
    3. scale_factor bug fix retained from previous version.
"""

import cv2
import numpy as np
import os


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def preprocess_image(image_path: str, save_debug: bool = False) -> tuple:
    """
    Full preprocessing pipeline for any image type.

    Args:
        image_path  : Path to the input image file.
        save_debug  : If True, saves intermediate steps to output/debug/ folder.

    Returns:
        (clean_image, scale_factor, channel_images) tuple:
            clean_image    — preprocessed grayscale numpy array ready for Tesseract OCR.
            scale_factor   — float (1.0 or 2.0). Divide OCR bounding box coords by this
                             before drawing on original image.
            channel_images — dict {"R": arr, "G": arr, "B": arr} each preprocessed
                             independently. Used by ocr_engine to catch channel-hidden text.

    Raises:
        FileNotFoundError : If the image path does not exist.
        ValueError        : If the image cannot be read/decoded.
    """

    # ── Step 0: Load ──────────────────────────────────────────────────────────
    image = _load_image(image_path)

    # ── Step 1: Extract colour channels BEFORE grayscale conversion ───────────
    # Must happen on the original BGR image, not after grayscale.
    channel_images = _split_colour_channels(image, save_debug)

    # ── Step 2: Grayscale ─────────────────────────────────────────────────────
    gray = _to_grayscale(image)
    _debug_save(gray, "01_grayscale", save_debug)

    # ── Step 3: Auto-Invert ───────────────────────────────────────────────────
    gray = _auto_invert(gray)
    _debug_save(gray, "02_after_invert", save_debug)

    # ── Step 4: Resize if too small ───────────────────────────────────────────
    gray, scale_factor = _resize_if_small(gray)
    _debug_save(gray, "03_after_resize", save_debug)

    # ── Step 5: Denoise ───────────────────────────────────────────────────────
    gray = _denoise(gray)
    _debug_save(gray, "04_after_denoise", save_debug)

    # ── Step 6: Threshold ─────────────────────────────────────────────────────
    binary = _auto_threshold(gray)
    _debug_save(binary, "05_after_threshold", save_debug)

    print(f"[Preprocess] Scale factor applied: {scale_factor}x")
    print(f"[Preprocess] Colour channels prepared: {list(channel_images.keys())}")
    return binary, scale_factor, channel_images


# ─────────────────────────────────────────────
# ANTI-OBFUSCATION — COLOUR CHANNEL SPLITTING
# ─────────────────────────────────────────────

def _split_colour_channels(image: np.ndarray, save_debug: bool = False) -> dict:
    """
    Split BGR image into individual R, G, B channel images.
    Each channel is independently thresholded and returned as a binary image
    ready for Tesseract OCR.

    WHY THIS EXISTS:
        Attackers can hide sensitive text by writing it in a single colour channel
        (e.g. white text on white background in the blue channel only). This text is
        completely invisible in grayscale conversion because grayscale merges all three
        channels. By running OCR on each channel separately, we catch text that only
        appears in one channel.

    Example attack:
        Normal viewer sees a blank white image.
        Blue channel contains: "Aadhaar: 9183 0074 6619" in blue-on-white.
        Grayscale OCR: sees nothing. Channel OCR: catches it.

    Returns:
        {
            "R": binary_array,
            "G": binary_array,
            "B": binary_array
        }
        Returns empty dict if image is already grayscale (no channels to split).
    """
    if len(image.shape) == 2:
        print("[Preprocess] Grayscale image — colour channel splitting skipped.")
        return {}

    b_ch, g_ch, r_ch = cv2.split(image)
    channels = {"R": r_ch, "G": g_ch, "B": b_ch}
    result = {}

    for name, ch in channels.items():
        # Invert if dark background
        if np.mean(ch) < 127:
            ch = cv2.bitwise_not(ch)

        # Upscale small channel images for better OCR
        h, w = ch.shape[:2]
        if min(h, w) < 1000:
            ch = cv2.resize(ch, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Denoise lightly
        ch = cv2.fastNlMeansDenoising(ch, h=10, templateWindowSize=7, searchWindowSize=21)

        # Threshold
        _, binary = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result[name] = binary
        _debug_save(binary, f"channel_{name}", save_debug)
        print(f"[Preprocess] Channel {name} prepared. Shape: {binary.shape}")

    return result


# ─────────────────────────────────────────────
# INTERNAL STEPS
# ─────────────────────────────────────────────

def _load_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}. "
            "Check that it is a valid image file (PNG, JPG, BMP, TIFF)."
        )
    print(f"[Preprocess] Loaded image: {image_path} | Shape: {image.shape}")
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        print("[Preprocess] Image is already grayscale.")
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("[Preprocess] Converted to grayscale.")
    return gray


def _auto_invert(gray: np.ndarray) -> np.ndarray:
    mean_brightness = np.mean(gray)
    print(f"[Preprocess] Mean brightness: {mean_brightness:.1f}")
    if mean_brightness < 127:
        gray = cv2.bitwise_not(gray)
        print("[Preprocess] Dark background detected → image inverted.")
    else:
        print("[Preprocess] Light background detected → no inversion needed.")
    return gray


def _resize_if_small(gray: np.ndarray, min_dimension: int = 1000) -> tuple:
    """
    Upscale image if too small for Tesseract.
    Returns (resized_image, scale_factor).
    scale_factor is 2.0 if upscaled, else 1.0.
    """
    h, w = gray.shape[:2]
    print(f"[Preprocess] Image size: {w}x{h}")
    if min(h, w) < min_dimension:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        print(f"[Preprocess] Image too small → upscaled 2x to {gray.shape[1]}x{gray.shape[0]}")
        return gray, 2.0
    else:
        print("[Preprocess] Image size OK → no resize needed.")
        return gray, 1.0


def _denoise(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    print("[Preprocess] Denoising applied.")
    return denoised


def _auto_threshold(gray: np.ndarray) -> np.ndarray:
    local_var = _estimate_lighting_variance(gray)
    print(f"[Preprocess] Lighting variance score: {local_var:.1f}")
    VARIANCE_THRESHOLD = 500
    if local_var > VARIANCE_THRESHOLD:
        binary = cv2.adaptiveThreshold(
            gray, maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=31, C=10
        )
        print("[Preprocess] Adaptive threshold applied (uneven lighting detected).")
    else:
        _, binary = cv2.threshold(
            gray, thresh=0, maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print("[Preprocess] OTSU threshold applied (uniform lighting detected).")
    return binary


def _estimate_lighting_variance(gray: np.ndarray) -> float:
    h, w = gray.shape
    block_means = []
    rows, cols = 4, 4
    for r in range(rows):
        for c in range(cols):
            block = gray[
                r * h // rows : (r + 1) * h // rows,
                c * w // cols : (c + 1) * w // cols
            ]
            block_means.append(np.mean(block))
    return float(np.var(block_means))


# ─────────────────────────────────────────────
# DEBUG HELPER
# ─────────────────────────────────────────────

def _debug_save(image: np.ndarray, step_name: str, enabled: bool) -> None:
    if not enabled:
        return
    debug_dir = os.path.join("output", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"{step_name}.png")
    cv2.imwrite(path, image)
    print(f"[Debug] Saved: {path}")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_image>")
        sys.exit(1)
    img_path = sys.argv[1]
    print("\n── Running Preprocessing Pipeline ──")
    result, sf, channels = preprocess_image(img_path, save_debug=True)
    print(f"\n── Done. Output shape: {result.shape} | scale_factor: {sf} ──")
    print(f"── Colour channels extracted: {list(channels.keys())} ──")
    print("Debug images saved to: output/debug/")