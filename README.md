# OCR Sensitive Information Detector — Image DLP
**Barclays Image DLP Project — Phase 1 Complete**

A fully local, lightweight DLP pipeline that detects sensitive information
inside images using OCR + regex classification.

---

## 🚀 Quick Start

```bash
# Single command — runs full pipeline
python main.py input/your_image.jpg

# Redact instead of highlight
python main.py input/your_image.jpg --mode redact

# ID cards / sparse text (use PSM 11)
python main.py input/idcard.jpg --psm 11

# Save preprocessing debug images
python main.py input/scan.jpg --debug
```

---

## 📁 Project Structure

```
ocr_sensitive_detector/
│
├── main.py                         ← 🚀 Entry point — run this
│
├── modules/
│   ├── preprocess.py               ✅ Image cleaning (grayscale, threshold, denoise)
│   ├── ocr_engine.py               ✅ Tesseract OCR + bounding box extraction
│   ├── sensitive_detector.py       ✅ Regex DLP — 14 pattern types
│   ├── annotator.py                ✅ Highlight / redact on original image
│   └── reporter.py                 ✅ Encrypted reports + audit log
│
├── input/                          ← Drop images here
│
├── output/
│   ├── <image>_annotated.jpg       ← Annotated output image
│   ├── <image>_redacted.jpg        ← Redacted output image
│   ├── debug/                      ← Preprocessing steps (--debug flag)
│   └── reports/
│       ├── <image>_report.enc      ← Encrypted findings (AES-128)
│       ├── .dlp_key                ← Encryption key (protect this!)
│       └── audit_log.jsonl         ← Safe audit trail (no sensitive values)
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_ocr_engine.py
│   ├── test_sensitive_detector.py
│   ├── test_annotator.py
│   └── test_reporter.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Install Tesseract (system binary)

**Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
Update `TESSERACT_PATH` in `modules/ocr_engine.py` if installed elsewhere.

**Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
**macOS:** `brew install tesseract`

Verify: `tesseract --version`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 🔍 Sensitive Patterns Detected (Phase 1)

| Type | Format | Example | Confidence |
|---|---|---|---|
| Email | local@domain.tld | sample@gmail.com | High |
| Phone (Indian) | +91 NNNNN NNNNN | +91 99999 99999 | High |
| PAN | AAAAA9999A | ABCDE1234F | High |
| Passport No. | A9999999 | J8369854 | High |
| GST Number | 99AAAAA9999A9Z9 | 27ABCDE1234F1Z5 | High |
| IFSC Code | AAAA0999999 | SBIN0001234 | High |
| SWIFT/BIC | AAAAAAAA | SBININBB | High |
| Aadhaar | 9999 9999 9999 | 9876 5432 1098 | Medium |
| Bank/Card | 9999 9999 9999 9999 | 4111 1111 1111 1111 | Medium |
| Voter ID | AAA9999999 | ABC1234567 | Medium |
| MICR Code | 999999999 | 400002009 | Medium |
| Date of Birth | DD/MM/YYYY | 12/04/1990 | Low |
| Driving Licence | AA99 9999 9999999 | MH12 2011 0012345 | Low |
| MRZ Line | A-Z 0-9 < (20-44 chars) | P<INDRAMADUGULA<< | High |

---

## 🖥️ CLI Options

```
python main.py <image_path> [options]

Options:
  --mode      highlight | redact    Annotation mode (default: highlight)
  --psm       3 | 6 | 11            Tesseract page segmentation (default: 6)
                                      6  = block text (scans, screenshots)
                                      11 = sparse text (ID cards, forms)
                                      3  = auto-detect
  --conf      0-100                 OCR word confidence filter (default: 60)
  --no-encrypt                      Plain JSON report (debug only)
  --debug                           Save preprocessing step images
```

---

## 📊 Output Files

| File | Contents | Encrypted |
|---|---|---|
| `output/<n>_annotated.jpg` | Original image with red boxes | No |
| `output/<n>_redacted.jpg` | Original image with black redactions | No |
| `output/reports/<n>_report.enc` | Full findings with masked+full values | ✅ AES-128 |
| `output/reports/audit_log.jsonl` | Type counts, timestamps, no values | No |

### Decrypt a report

```python
from modules.reporter import decrypt_report
report = decrypt_report("output/reports/myimage_report.enc")
for f in report["findings"]:
    print(f["type"], f["value_full"])
```

---

## 🧪 Run All Tests

```bash
python tests/test_preprocess.py      input/your_image.jpg
python tests/test_ocr_engine.py      input/your_image.jpg
python tests/test_sensitive_detector.py
python tests/test_annotator.py       input/your_image.jpg
python tests/test_reporter.py
```

---

## 🗺️ Phase Roadmap

| Phase | What | Status |
|---|---|---|
| **Phase 1** | OCR + Regex DLP + Annotate + Encrypted Report | ✅ **Complete** |
| **Phase 2** | + NLP Classifier (BERT) — contextual sensitivity | 🔜 Next |
| **Phase 3** | + Vision Models (CLIP / Detectron2) | 🔜 Upcoming |
| **Phase 4** | + Confidence Engine + OCR failure test suite | 🔜 Upcoming |
