# Image DLP Review Coverage

This file maps the industry review points to the current implementation.
Steganography is intentionally excluded from this scope.

## Covered Review Points

| Review point | Implementation |
|---|---|
| Detect sensitive information inside images | `main.py` runs preprocessing, OCR, regex DLP, NLP, confidence scoring, vision classification, annotation, reporting, and training storage. |
| OCR limitations: low contrast, blur, styled fonts, obfuscation | `tests/test_ocr_failure_scenarios.py` generates degraded document-like images and verifies that the vision layer compensates when OCR word count is low. |
| Confidence score handling | `modules/confidence_engine.py` combines regex, NLP, and OCR quality. `modules/ocr_engine.py` returns true raw, filtered, dropped word counts and dropped-word ratio. |
| Partial OCR and low-confidence outputs | Low-confidence OCR words are filtered before regex/NLP. Poor OCR quality lowers the OCR contribution in the unified score. |
| Secure structured metadata | `modules/reporter.py` writes encrypted reports. `modules/training_store.py` masks `finding_value`, `ocr_text_window`, and `full_ocr_text` before writing JSONL training records. |
| Screenshots/photos/scanned documents | OCR handles readable text. Phase 3 vision classifies document-like images even when text extraction is sparse. |
| ML-based classification | NLP handles contextual classification; vision handles image-level document type and OCR-failure risk. |

## Vision Layer Limitations

The vision layer is an image-level safety net. It can say that an image looks
like an ID card, cheque, passport, screenshot, or sensitive document when OCR
is weak. It does not produce exact word-level bounding boxes unless OCR
produced tokens.

For that reason, `annotator.py` may not draw a red box for a vision-only
finding. The report, audit metadata, training record, and final risk score
still flag the image as sensitive.

The optional heatmap/patch localization hook is intentionally graceful: if CLIP
is unavailable, the system returns no heatmap and continues with OpenCV
heuristics. This keeps the pipeline compatible with Windows Python 3.13
CPU-only environments.

## Remaining Scope Notes

- Use real organization-approved sample images before the final demo if
  available. The repository tests use synthetic degraded images so no private
  documents are stored.
- BERT/CLIP are optional because Python 3.13 CPU-only compatibility is limited.
  The production path stays functional with spaCy and OpenCV fallbacks.
