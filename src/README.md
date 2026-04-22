# Source Code — Agent Guide

## Module Overview

Each file is a self-contained stage in the preprocessing and training pipeline.

```
src/
├── preprocess.py      Phase 1  — clean, normalise, length-filter, deduplicate, split
├── labse_filter.py    Phase 1b — LaBSE cosine similarity quality filter (≥0.85)
├── tokenise.py        Phase 2  — SentencePiece BPE training and application
├── train_baseline.py  Phase 3  — direct Si→Ta Transformer (Pramodya et al. 2024 config)
├── cascade_pivot.py   Phase 4  — Si→En (Helsinki-NLP) + En→Ta (IndicTrans2) cascade
└── evaluate.py        Phase 5  — BLEU + chrF via sacreBLEU, comparison report
```

## Module Details

### preprocess.py

**Inputs:** `data/raw/corpus.si`, `data/raw/corpus.ta`  
**Outputs:** `data/processed/train.{si,ta}`, `data/test_set/test.{si,ta}`

Key functions:
- `nfc(text)` — Unicode NFC normalisation (critical: legacy Sri Lankan documents use inconsistent Unicode)
- `length_filter(pairs, min=5, max=100)` — removes pairs outside token count range
- `split_test(pairs, test_size=50, seed=42)` — fixed test split; seed must not change

### labse_filter.py

**Inputs:** `data/processed/train.{si,ta}`  
**Outputs:** `data/processed/train_filtered.{si,ta}`, `train_filtered_scores.txt`

Uses `sentence-transformers/LaBSE` to compute cosine similarity per pair.
Threshold 0.85 follows Feng et al. (2022). Lowering it to 0.75 recovers more pairs at higher noise.

### tokenise.py

**Inputs:** `data/processed/train_filtered.{si,ta}`  
**Outputs:** `data/processed/spm/si_spm.model`, `ta_spm.model`, tokenised files

Two modes: `train` (learns BPE vocab) and `apply` (encodes text using learned vocab).
`character_coverage=0.9995` handles the large Tamil character set (247 characters).

Exposes `encode_sentence()` and `decode_pieces()` for use in inference code.

### train_baseline.py

**Inputs:** Tokenised train files + SPM models  
**Outputs:** `models/baseline/model.pt`, `meta.json`

Architecture (must match Pramodya et al. 2024 for comparable results):
- `d_model=512`, `nhead=8`, `num_enc_layers=4`, `num_dec_layers=4`, `dim_feedforward=2048`
- Noam LR schedule with `warmup_steps=4000` (400 for PoC)
- Label smoothing = 0.1

`greedy_translate()` is used at inference time. For full research, switch to beam search (beam=4).

### cascade_pivot.py

**Inputs:** Raw Sinhala test sentences  
**Outputs:** `models/cascade/intermediate_en.txt`, `models/cascade/final_ta.txt`

Constants to change for fine-tuned full system:
- `SI_EN_MODEL` — replace with path to fine-tuned Si→En checkpoint
- `EN_TA_MODEL` — replace with path to fine-tuned IndicTrans2 checkpoint
- `INDIC_LANG_CODE = "tam_Taml"` — Tamil in ISO 639-3 + script code

`freeze_encoder(model)` — call after Stage 1 training to apply Zhang et al. (2022) parameter freezing.

### evaluate.py

**Inputs:** Reference file + hypothesis files from both systems  
**Outputs:** `results/baseline_scores.json`, `cascade_scores.json`, `comparison.md`

Uses `sacrebleu.corpus_bleu()` and `sacrebleu.corpus_chrf()` for reproducible tokenisation.
The comparison report (`comparison.md`) includes a sample translation table with intermediate English column for error propagation analysis.

## Extending the Pipeline

### Adding Tamil-to-Sinhala direction

1. In `cascade_pivot.py`: swap Si→En for Ta→En (IndicTrans2 in `ta-en` mode) and En→Si (Helsinki-NLP/opus-mt-en-si)
2. In `train_baseline.py`: swap `--train-si` and `--train-ta` arguments
3. In `evaluate.py`: swap `--reference` and `--source` accordingly

### Adding morpheme-based segmentation (Morfessor)

Morfessor was shown to improve Sinhala tokenisation (Pushpananda et al. 2014):

```bash
pip install morfessor
```

Then in `tokenise.py`, add a `train_morfessor()` function using the `morfessor` library
and compare results against BPE on the same test set.

### Scaling to full 25k corpus

No code changes needed. Replace `data/raw/corpus.{si,ta}` with the real corpus, then:
- Increase `--vocab-size` to 8000 in tokenise.py
- Increase `--epochs` to 50+ in train_baseline.py
- Increase `--test-size` to 500 in preprocess.py
- Use `--batch-size 32` and `--warmup-steps 4000` in train_baseline.py
