# AGENTS.md — src/

## Module Responsibilities

Each module owns exactly one pipeline stage. Do not mix concerns across modules.

| Module | Stage | Reads from | Writes to |
|--------|-------|------------|-----------|
| preprocess.py | 1 | data/raw/ | data/processed/, data/test_set/ |
| labse_filter.py | 1b | data/processed/train.* | data/processed/train_filtered.* |
| tokenise.py | 2 | data/processed/train_filtered.* | data/processed/spm/, data/processed/tokenised/ |
| train_baseline.py | 3 | data/processed/tokenised/ | models/baseline/ |
| cascade_pivot.py | 4 | data/test_set/test.si | models/cascade/ |
| evaluate.py | 5 | data/test_set/, models/ | results/ |

## Invariants Every Agent Must Preserve

### preprocess.py
- `split_test()` must always use `seed=42`. Changing the seed changes the test set composition and invalidates published results.
- The test set size in the full research system is 500. In the PoC it is 50. This is controlled by `--test-size`, not hardcoded.
- Empty strings after cleaning must be filtered out before splitting.

### labse_filter.py
- The model `sentence-transformers/LaBSE` must not be swapped for another embedding model without documenting the change in the dissertation.
- Threshold 0.85 is a research parameter from Feng et al. (2022). Lowering it requires justification.
- Output files must be named `{stem}_{suffix}.si/ta` where `stem` matches the input stem. This naming is expected by `tokenise.py`.

### tokenise.py
- Separate models for Sinhala and Tamil. Never train a shared multilingual vocabulary — the scripts are disjoint and a shared vocab wastes capacity.
- `character_coverage=0.9995` is required for Tamil's 247-character set. Lowering it produces `<unk>` tokens for uncommon characters.
- Special token IDs: `<unk>=0`, `<s>=1`, `</s>=2`, `<pad>=3`. These must match what `train_baseline.py` expects.

### train_baseline.py
- Architecture constants (`d_model=512`, `nhead=8`, `num_enc_layers=4`, `num_dec_layers=4`) must match Pramodya et al. (2024) for the dissertation comparison to be valid.
- `ignore_index=3` in `CrossEntropyLoss` matches `<pad>` token ID. If pad ID changes in tokenise.py, update here too.
- `model.pt` is a `state_dict`, not the full model. Must load with matching architecture config.

### cascade_pivot.py
- `intermediate_en.txt` must always be saved. It is required by `evaluate.py` and for error propagation analysis.
- `INDIC_LANG_CODE = "tam_Taml"` is the IndicTrans2 language code for Tamil in Taml script. Do not guess other codes.
- `freeze_encoder()` is a utility for the full training phase. In the PoC it is demonstrated but not applied to training.
- Delete model from GPU memory between Step 1 and Step 2 when VRAM is limited (`del si_en_model; torch.cuda.empty_cache()`).

### evaluate.py
- Use `sacrebleu.corpus_bleu()` only, never sentence-level BLEU averaged over sentences. Corpus BLEU is the standard in MT research.
- The comparison report (`comparison.md`) must include the intermediate English column for error propagation analysis — this is directly tied to Research Sub-question 2.
- `n_samples=10` for PoC. Use `n_samples=20` for the full 500-sentence evaluation in the dissertation.

## Adding New Modules

If you add a new preprocessing step (e.g., Morfessor-based segmentation):
1. Create `src/morfessor_segment.py` as a standalone module with CLI
2. Insert it between `labse_filter.py` and `tokenise.py` in the pipeline
3. Update `data/AGENTS.md` with the new output files
4. Update the notebook `poc.ipynb` with a new phase cell
5. Document the change in the dissertation methodology chapter
