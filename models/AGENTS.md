# AGENTS.md — models/

## What Belongs Here

Model artifacts produced during training and inference. These are generated files — never edit them directly.

```
models/
├── baseline/
│   ├── model.pt          ← PyTorch state_dict (not the full model object)
│   ├── meta.json         ← Config + training history
│   └── hypothesis.ta     ← Model's translations of data/test_set/test.si
│
└── cascade/
    ├── intermediate_en.txt   ← Si→En output (Step 1 of cascade)
    └── final_ta.txt          ← En→Ta output (Step 2 = final cascade hypothesis)
```

## Loading Models

### Baseline

```python
import torch, json
from src.train_baseline import BaselineTransformer, build_vocab_from_spm_vocab

meta = json.loads(Path('models/baseline/meta.json').read_text())
si_vocab = build_vocab_from_spm_vocab('data/processed/spm/si_spm.model')
ta_vocab = build_vocab_from_spm_vocab('data/processed/spm/ta_spm.model')

model = BaselineTransformer(
    src_vocab_size=meta['si_vocab_size'],
    tgt_vocab_size=meta['ta_vocab_size'],
    d_model=meta['d_model'],
)
model.load_state_dict(torch.load('models/baseline/model.pt', map_location='cpu'))
model.eval()
```

### Cascade (HuggingFace models — loaded directly by cascade_pivot.py)

The cascade uses pretrained HuggingFace models. They are not stored in this directory;
they are downloaded to the HuggingFace cache (`~/.cache/huggingface/`) on first use.

For fine-tuned versions (full research system), save the fine-tuned model:

```python
si_en_model.save_pretrained('models/cascade/si_en_finetuned/')
en_ta_model.save_pretrained('models/cascade/en_ta_finetuned/')
```

Then update the constants in `src/cascade_pivot.py`:
```python
SI_EN_MODEL = 'models/cascade/si_en_finetuned'
EN_TA_MODEL = 'models/cascade/en_ta_finetuned'
```

## Checkpointing on Kaggle

Kaggle sessions are limited to 30 hours. For long training runs:

```python
checkpoint_path = f'models/baseline/ckpt_epoch{epoch}.pt'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, checkpoint_path)
```

Resume:
```python
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Do Not Commit

Large model files (`model.pt`, `*.bin`, HuggingFace cache) should be in `.gitignore`.
Commit only `meta.json` and hypothesis text files for reproducibility tracking.
