# AGENTS.md — data/

## Directory Purpose

Stores all parallel text data at each processing stage. The pipeline flows left to right:

```
raw/ → processed/ → processed/tokenised/
                         ↓
                    test_set/  (split off at processed/ stage, never modified again)
```

## Stage Descriptions

### data/raw/
Original unprocessed files as received. For the PoC this is the synthetic 118-pair corpus from `generate_sample_data.py`. For the full research system, this is replaced by the real Pushpananda et al. (2013) corpus.

Files: `corpus.si`, `corpus.ta`  
Format: UTF-8 plain text, one sentence per line, lines aligned (line N of .si = translation of line N of .ta)

### data/processed/
Output of `src/preprocess.py` and `src/labse_filter.py`.

| File | Created by | Description |
|------|-----------|-------------|
| train.si / train.ta | preprocess.py | After cleaning + length filter + dedup |
| train_filtered.si / .ta | labse_filter.py | After LaBSE quality filter |
| train_filtered_scores.txt | labse_filter.py | LaBSE similarity score per pair |
| spm/si_spm.model | tokenise.py | Sinhala BPE model |
| spm/ta_spm.model | tokenise.py | Tamil BPE model |
| tokenised/train_filtered.si / .ta | tokenise.py | Space-separated BPE pieces |

### data/test_set/
**Sacred. Never used as training input.** Fixed with `seed=42`.

| File | Created by | Description |
|------|-----------|-------------|
| test.si | preprocess.py | 20 Sinhala test sentences (PoC) / 500 (full) |
| test.ta | preprocess.py | Corresponding Tamil reference translations |

## Rules for Agents

1. **Do not write to test_set/ directly.** It is only written by `src/preprocess.py` during the initial pipeline setup. If you need to regenerate it, re-run preprocessing with the same seed.

2. **Line counts must match.** Every `.si` file must have the same number of lines as its corresponding `.ta` file. Mismatches will crash training and evaluation.

3. **Always use UTF-8.** Sinhala and Tamil use non-ASCII Unicode. Any file operation that doesn't specify `encoding='utf-8'` will corrupt data on non-UTF-8 systems.

4. **Do not commit model files or large corpora.** `data/processed/spm/*.model` and the real corpus files should be in `.gitignore`. Only the raw PoC corpus (118 pairs) and the scripts are committed.

## Replacing PoC Data with Real Corpus

```bash
# 1. Obtain real corpus and place files:
#    data/raw/corpus.si   (~25,000 Sinhala sentences)
#    data/raw/corpus.ta   (~25,000 Tamil sentences)

# 2. Run pipeline with full-scale parameters:
python src/preprocess.py \
    --si data/raw/corpus.si --ta data/raw/corpus.ta \
    --test-size 500 --min-tokens 5 --max-tokens 100

python src/labse_filter.py \
    --si data/processed/train.si --ta data/processed/train.ta \
    --threshold 0.85

python src/tokenise.py train \
    --si data/processed/train_filtered.si \
    --ta data/processed/train_filtered.ta \
    --vocab-size 8000

python src/tokenise.py apply \
    --si data/processed/train_filtered.si \
    --ta data/processed/train_filtered.ta
```

## Downloading Real Datasets

```python
from datasets import load_dataset

# FLORES-200 (high quality, small — for evaluation/bridge data)
flores_si = load_dataset('facebook/flores', 'sin_Sinh')
flores_ta = load_dataset('facebook/flores', 'tam_Taml')

# Samanantar (large English-Tamil bridge corpus)
samanantar_en_ta = load_dataset('ai4bharat/samanantar', 'en-ta')

# For English-Sinhala bridge data, use the IndicCorp / OPUS corpus
# or the Ranathunga et al. (2018) 10M pair corpus (contact authors)
```
