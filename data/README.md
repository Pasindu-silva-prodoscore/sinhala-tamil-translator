# Data Directory — Agent Guide

## What This Directory Contains

Parallel Sinhala-Tamil text data at each stage of the preprocessing pipeline.

```
data/
├── generate_sample_data.py   ← Script to recreate data/raw/ from scratch
├── raw/                      ← Stage 0: Original unprocessed files
│   ├── corpus.si             ← Sinhala sentences, one per line
│   └── corpus.ta             ← Tamil sentences, one per line (aligned with corpus.si)
├── processed/                ← Stage 1-2: Cleaned, filtered, tokenised
│   ├── train.si / train.ta              ← After preprocess.py
│   ├── train_filtered.si / .ta          ← After labse_filter.py
│   ├── train_filtered_scores.txt        ← LaBSE similarity scores
│   ├── spm/
│   │   ├── si_spm.model                 ← Sinhala SentencePiece model
│   │   ├── si_spm.vocab
│   │   ├── ta_spm.model                 ← Tamil SentencePiece model
│   │   └── ta_spm.vocab
│   └── tokenised/
│       ├── train_filtered.si            ← BPE-tokenised Sinhala
│       └── train_filtered.ta            ← BPE-tokenised Tamil
└── test_set/                 ← Held-out evaluation set (NEVER used in training)
    ├── test.si
    └── test.ta
```

## Critical Rules for Agents

1. **Never use test_set/ files as training input.** The test split is held out with `seed=42` in `src/preprocess.py`. Contaminating it makes BLEU scores meaningless.

2. **Always work on filtered data.** The pipeline is: `raw/ → processed/train.* → processed/train_filtered.* → processed/tokenised/`. Skip `labse_filter.py` only when explicitly debugging preprocessing.

3. **File format:** All `.si` and `.ta` files are plain UTF-8 text, one sentence per line. Line N of `corpus.si` is the translation of line N of `corpus.ta`.

4. **Replacing PoC data with real corpus:** Drop the real Pushpananda et al. (2013) corpus files into `raw/` as `corpus.si` and `corpus.ta`, then rerun the pipeline from `src/preprocess.py`.

## Replacing the Sample Data with Real Corpus

The real Sinhala-Tamil parallel corpus (~25,000 pairs) can be obtained from:
- Pushpananda et al. (2013) — contact authors
- Ranathunga et al. (2025) trilingual corpus (NAACL 2025)

For bridge language data:
- FLORES-200: `datasets.load_dataset('facebook/flores', 'sin_Sinh')` and `'tam_Taml'`
- Samanantar: `datasets.load_dataset('ai4bharat/samanantar', 'en-ta')` and `'en-si'`

Place Sinhala sentences in `raw/corpus.si`, Tamil in `raw/corpus.ta`, then run:

```bash
python src/preprocess.py --si data/raw/corpus.si --ta data/raw/corpus.ta
python src/labse_filter.py --si data/processed/train.si --ta data/processed/train.ta
python src/tokenise.py train --si data/processed/train_filtered.si --ta data/processed/train_filtered.ta --vocab-size 8000
python src/tokenise.py apply --si data/processed/train_filtered.si --ta data/processed/train_filtered.ta
```
