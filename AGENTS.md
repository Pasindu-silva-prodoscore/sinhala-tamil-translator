# AGENTS.md — Sinhala-Tamil Pivot MT PoC (Root)

## Project Purpose

This is the proof-of-concept implementation for a research dissertation on using English as a pivot language for Sinhala-Tamil neural machine translation. The goal is to demonstrate that a cascade Si→En→Ta pipeline substantially outperforms a direct Transformer trained on the small (~25k sentence pair) Sinhala-Tamil corpus.

## Architecture at a Glance

```
Raw parallel text
    ↓ src/preprocess.py       Unicode NFC, length filter, dedup, train/test split
    ↓ src/labse_filter.py     LaBSE cosine ≥ 0.85 quality filter
    ↓ src/tokenise.py         SentencePiece BPE (vocab=4000 PoC / 8000 full)
    ↓
    ├── src/train_baseline.py  4L/4L Transformer direct Si→Ta
    └── src/cascade_pivot.py   Helsinki-NLP Si→En  +  IndicTrans2 En→Ta
    ↓
    src/evaluate.py            BLEU + chrF (sacreBLEU) + comparison report
```

## Key Research Parameters (do not change without updating the dissertation)

| Parameter | Value | Source |
|-----------|-------|--------|
| LaBSE threshold | 0.85 | Feng et al. (2022) |
| Transformer depth | 4 enc / 4 dec layers | Pramodya et al. (2024) |
| d_model | 512 | Pramodya et al. (2024) |
| Attention heads | 8 | Pramodya et al. (2024) |
| Label smoothing | 0.1 | Vaswani et al. (2017) |
| Warmup steps | 4000 (full) / 400 (PoC) | Vaswani et al. (2017) |
| BPE vocab size | 8000 (full) / 4000 (PoC) | Tennage et al. (2017) |
| Test set seed | 42 | Fixed for reproducibility |
| Test set size | 500 (full) / 50 (PoC) | Research objective 5 |
| Si→En model | Helsinki-NLP/opus-mt-si-en | PoC only; fine-tune for full system |
| En→Ta model | ai4bharat/indictrans2-en-indic-1B | Gala et al. (2023) |

## Coding Standards

- Python 3.10+. Use built-in type hints (`list[str]`, not `List[str]`).
- All file I/O: UTF-8 encoding explicitly specified.
- No `as any`, no suppressed exceptions.
- Parallel files always have identical line counts. Any mismatch should raise `ValueError`, not silently proceed.
- Test set (`data/test_set/`) is sacred — never used as training input, never modified.

## File Naming Conventions

- Sinhala files: `.si` extension
- Tamil files: `.ta` extension  
- English intermediate: `.en` extension or `intermediate_en.txt`
- SentencePiece models: `{lang}_spm.model` and `{lang}_spm.vocab`

## What Agents Should NOT Do

- Change the test set seed (42) — this must remain fixed for the full dissertation to compare results
- Merge Sinhala and Tamil vocabularies into a single shared vocabulary (they use different scripts)
- Use `BLEU` implementations other than `sacrebleu.corpus_bleu()` — different tokenisers produce incomparable scores
- Remove `intermediate_en.txt` from cascade outputs — it is needed for error propagation analysis
- Add training data from `data/test_set/` — this contaminates the evaluation

## Common Tasks for Agents

### Switching to the full 25k corpus

1. Replace `data/raw/corpus.si` and `data/raw/corpus.ta`
2. Run `src/preprocess.py` with `--test-size 500 --min-tokens 5 --max-tokens 100`
3. Run `src/labse_filter.py` (same threshold)
4. Run `src/tokenise.py train` with `--vocab-size 8000`
5. Run `src/train_baseline.py` with `--epochs 50 --batch-size 32 --warmup-steps 4000`

### Fine-tuning the cascade models

In `src/cascade_pivot.py`:
- Replace `SI_EN_MODEL` constant with path to fine-tuned Si→En checkpoint
- Replace `EN_TA_MODEL` constant with path to fine-tuned IndicTrans2 checkpoint
- Call `freeze_encoder(si_en_model)` after Stage 1 training completes

### Adding Tamil-to-Sinhala direction

The PoC covers Si→Ta only. For Ta→Si:
- Step 1: IndicTrans2 in `ta-en` mode
- Step 2: `Helsinki-NLP/opus-mt-en-si` (or `opus-mt-mul-en` for multilingual)
- Swap `--train-si` / `--train-ta` arguments in `train_baseline.py`

## Dependencies and Versions

See `requirements.txt` for pinned versions. Key packages:
- `transformers>=4.38.0` — required for IndicTrans2 loading
- `sacrebleu>=2.3.1` — BLEU/chrF evaluation
- `sentencepiece>=0.1.99` — BPE tokenisation
- `sentence-transformers>=2.6.0` — LaBSE model

## Dissertation Chapter Mapping

| Chapter | PoC Component |
|---------|---------------|
| Ch 3.3 Dataset | `data/` directory, `generate_sample_data.py` |
| Ch 3.4 Preprocessing | `src/preprocess.py`, `src/labse_filter.py`, `src/tokenise.py` |
| Ch 3.5.1 Baseline | `src/train_baseline.py` |
| Ch 3.5.2 Cascade | `src/cascade_pivot.py` |
| Ch 3.6 Evaluation | `src/evaluate.py`, `results/` |
