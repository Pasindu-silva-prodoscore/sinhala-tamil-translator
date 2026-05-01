#!/usr/bin/env python3
"""
train_baseline.py
=================
Phase 3 — Direct Sinhala→Tamil baseline Transformer.

Architecture (follows Pramodya et al. 2024, current SOTA for this pair):
  - MarianMT-style encoder-decoder Transformer
  - 4 encoder layers, 4 decoder layers, 8 attention heads, d_model=512
  - Adam optimiser, warmup_steps=4000, label_smoothing=0.1
  - BPE-tokenised input via the SentencePiece models trained in tokenise.py

For the PoC this trains on ~150 pairs for 20 epochs to show the pipeline
works end-to-end.  Full research training should use the complete 25k corpus
for 50+ epochs on a GPU.

Usage:
    python src/train_baseline.py \
        --train-si data/processed/tokenised/train_filtered.si \
        --train-ta data/processed/tokenised/train_filtered.ta \
        --si-spm   data/processed/spm/si_spm.model \
        --ta-spm   data/processed/spm/ta_spm.model \
        --out-dir  models/baseline \
        --epochs   20 \
        --batch-size 16
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParallelDataset(Dataset):
    def __init__(
        self,
        si_path: Path,
        ta_path: Path,
        si_vocab: dict[str, int],
        ta_vocab: dict[str, int],
        max_len: int = 128,
    ):
        si_lines = si_path.read_text(encoding="utf-8").splitlines()
        ta_lines = ta_path.read_text(encoding="utf-8").splitlines()
        self.pairs = list(zip(si_lines, ta_lines))
        self.si_vocab = si_vocab
        self.ta_vocab = ta_vocab
        self.max_len = max_len
        self.pad_id = si_vocab.get("<pad>", 3)
        self.bos_id = ta_vocab.get("<s>", 1)
        self.eos_id = ta_vocab.get("</s>", 2)

    def _encode(self, line: str, vocab: dict[str, int]) -> list[int]:
        unk = vocab.get("<unk>", 0)
        return [vocab.get(tok, unk) for tok in line.split()]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        si_line, ta_line = self.pairs[idx]
        src = self._encode(si_line, self.si_vocab)[: self.max_len]
        tgt_full = (
            [self.bos_id]
            + self._encode(ta_line, self.ta_vocab)[: self.max_len - 2]
            + [self.eos_id]
        )
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt_full, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_id: int = 3) -> dict[str, torch.Tensor]:
    src_lens = [b["src"].size(0) for b in batch]
    tgt_lens = [b["tgt"].size(0) for b in batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)

    for i, b in enumerate(batch):
        src_padded[i, : src_lens[i]] = b["src"]
        tgt_padded[i, : tgt_lens[i]] = b["tgt"]

    src_mask = src_padded != pad_id
    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,
    }


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def build_vocab_from_spm_vocab(model_path: Path) -> dict[str, int]:
    """
    Build a {piece: id} dict from a SentencePiece .vocab file
    (or directly from the model via the spm library).
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    return vocab


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class BaselineTransformer(nn.Module):
    """
    Minimal Transformer encoder-decoder for Sinhala→Tamil.

    Config follows Pramodya et al. (2024):
      4 enc layers, 4 dec layers, 8 heads, d_model=512, FFN=2048.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 3,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=~src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=~src_key_padding_mask,
        )
        return self.output_proj(out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(tgt[:, :-1], memory, src_mask)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: BaselineTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_mask = batch["src_mask"].to(device)

        logits = model(src, tgt, src_mask)
        tgt_out = tgt[:, 1:]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg = total_loss / max(len(loader), 1)
    print(f"  Epoch {epoch:3d} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
    return avg


# ---------------------------------------------------------------------------
# Greedy inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_translate(
    model: BaselineTransformer,
    src_tokens: list[int],
    bos_id: int,
    eos_id: int,
    device: torch.device,
    max_new_tokens: int = 128,
) -> list[int]:
    model.eval()
    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    src_mask = src != model.pad_id
    memory = model.encode(src, src_mask)

    generated = [bos_id]
    for _ in range(max_new_tokens):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model.decode(tgt, memory, src_mask)
        next_token = logits[0, -1].argmax().item()
        generated.append(next_token)
        if next_token == eos_id:
            break
    return generated[1:]


# ---------------------------------------------------------------------------
# Hypothesis generation
# ---------------------------------------------------------------------------

def generate_hypothesis(
    model: BaselineTransformer,
    test_si_path: Path,
    si_spm_path: Path,
    ta_spm_path: Path,
    out_path: Path,
    device: torch.device,
) -> None:
    import sentencepiece as spm

    si_sp = spm.SentencePieceProcessor()
    si_sp.load(str(si_spm_path))
    ta_sp = spm.SentencePieceProcessor()
    ta_sp.load(str(ta_spm_path))
    ta_vocab = build_vocab_from_spm_vocab(ta_spm_path)
    id_to_piece = {v: k for k, v in ta_vocab.items()}

    si_vocab = build_vocab_from_spm_vocab(si_spm_path)
    test_sentences = test_si_path.read_text(encoding="utf-8").splitlines()

    hypotheses: list[str] = []
    for sent in test_sentences:
        pieces = si_sp.encode(sent, out_type=str)
        token_ids = [si_vocab.get(p, si_vocab.get("<unk>", 0)) for p in pieces]
        generated_ids = greedy_translate(model, token_ids, bos_id=1, eos_id=2, device=device)
        decoded_pieces = [
            id_to_piece.get(i, "<unk>") for i in generated_ids if i not in (1, 2, 3)
        ]
        hypotheses.append(ta_sp.decode(decoded_pieces))

    out_path.write_text("\n".join(hypotheses), encoding="utf-8")
    print(f"  Wrote {len(hypotheses)} hypotheses → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train direct Sinhala→Tamil baseline")
    parser.add_argument("--train-si", required=True)
    parser.add_argument("--train-ta", required=True)
    parser.add_argument("--si-spm", required=True, help="Sinhala SPM model path")
    parser.add_argument("--ta-spm", required=True, help="Tamil SPM model path")
    parser.add_argument("--out-dir", default="models/baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--test-si", default=None, help="Sinhala test file for hypothesis generation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building vocabularies ...")
    si_vocab = build_vocab_from_spm_vocab(Path(args.si_spm))
    ta_vocab = build_vocab_from_spm_vocab(Path(args.ta_spm))
    print(f"  si vocab size: {len(si_vocab)}, ta vocab size: {len(ta_vocab)}")

    dataset = ParallelDataset(
        Path(args.train_si), Path(args.train_ta), si_vocab, ta_vocab
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=3),
    )
    print(f"  Dataset: {len(dataset)} pairs, {len(loader)} batches/epoch")

    model = BaselineTransformer(
        src_vocab_size=len(si_vocab),
        tgt_vocab_size=len(ta_vocab),
        d_model=args.d_model,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=3, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return args.d_model ** -0.5 * min(step ** -0.5, step * args.warmup_steps ** -1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train(model, loader, optimizer, scheduler, criterion, device, epoch)
        history.append({"epoch": epoch, "loss": loss})

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s")

    torch.save(model.state_dict(), out_dir / "model.pt")

    meta = {
        "si_vocab_size": len(si_vocab),
        "ta_vocab_size": len(ta_vocab),
        "d_model": args.d_model,
        "epochs": args.epochs,
        "history": history,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved model → {out_dir}/model.pt")
    print(f"  Saved meta  → {out_dir}/meta.json")

    if args.test_si:
        print("Generating hypotheses on test set ...")
        generate_hypothesis(
            model,
            test_si_path=Path(args.test_si),
            si_spm_path=Path(args.si_spm),
            ta_spm_path=Path(args.ta_spm),
            out_path=out_dir / "hypothesis.ta",
            device=device,
        )


if __name__ == "__main__":
    main()
