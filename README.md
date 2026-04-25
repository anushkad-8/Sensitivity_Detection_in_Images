# OCR Sensitive Information Detector

A lightweight, local Python tool that:
1. Preprocesses images for OCR
2. Extracts text using Tesseract
3. Detects sensitive information using regex patterns
4. Highlights / flags findings on the image

No cloud. No ML models. No enterprise complexity.

---

## 📁 Project Structure

```
ocr_sensitive_detector/
│
├── modules/                    # All processing modules (one per pipeline stage)
│   ├── __init__.py             # Makes modules/ a Python package
│   ├── preprocess.py           # Module 1: Image cleaning (DONE ✅)
│   ├── ocr_engine.py           # Module 2: Text extraction via Tesseract (coming next)
│   └── sensitive_detector.py  # Module 3: Regex-based sensitive info detection (coming)
│
├── input/                      # Drop your images here
│   └── (your images go here)
│
├── output/                     # Results saved here automatically
│   ├── debug/                  # Intermediate preprocessing steps (when save_debug=True)
│   └── (annotated output images saved here)
│
├── tests/                      # Quick test scripts per module
│   └── test_preprocess.py      # Test for Module 1 (coming)
│
├── main.py                     # 🚀 Entry point — runs the full pipeline (coming)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Setup Instructions

### 1. Install Tesseract (System Binary)

**Ubuntu / Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
Then add to PATH.

Verify install:
```bash
tesseract --version
```

---

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Pipeline (once all modules are built)

```bash
python main.py input/your_image.png
```

---

## 🧪 Test Module 1 Alone

```bash
python modules/preprocess.py input/your_image.png
```

Debug images will be saved to `output/debug/` showing each preprocessing step.

---

## 🔍 Debug Mode

Pass `save_debug=True` in code to save intermediate images:

```python
from modules.preprocess import preprocess_image
result = preprocess_image("input/sample.png", save_debug=True)
```

This saves 5 intermediate images to `output/debug/`:
- `01_grayscale.png`
- `02_after_invert.png`
- `03_after_resize.png`
- `04_after_denoise.png`
- `05_after_threshold.png`

Useful for diagnosing why OCR might be underperforming on a specific image.

---

## 📦 Module Status

| Module | File | Status |
|--------|------|--------|
| 1. Preprocess | `modules/preprocess.py` | ✅ Complete |
| 2. OCR Engine | `modules/ocr_engine.py` | ✅ Complete |
| 3. Sensitive Detector | `modules/sensitive_detector.py` | ✅ Complete |
| Entry Point | `main.py` | 🔜 Final step |
