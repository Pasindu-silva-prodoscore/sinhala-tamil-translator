#!/usr/bin/env python3
"""
tokenise.py
===========
Phase 2 — BPE tokenisation via SentencePiece.

Steps:
  1. Train a SentencePiece BPE model on each language independently
     (vocabulary size 4,000 for PoC; 8,000 recommended for full training).
  2. Apply the trained model to all splits (train / test).
  3. Provide encode() / decode() helpers for use in training and evaluation.

Design notes:
  - Separate vocabulary per language (Sinhala vs Tamil scripts are disjoint).
  - BPE handles agglutinative morphology by splitting unseen inflected forms
    into known subword pieces (root + suffix tokens).
  - SentencePiece treats the input as a raw unicode stream — no whitespace
    pre-tokeniser needed for Indic scripts.

Usage:
    # Train BPE models
    python src/tokenise.py train \
        --si data/processed/train_filtered.si \
        --ta data/processed/train_filtered.ta \
        --model-dir data/processed/spm \
        --vocab-size 4000

    # Apply BPE to a file pair
    python src/tokenise.py apply \
        --si data/processed/train_filtered.si \
        --ta data/processed/train_filtered.ta \
        --model-dir data/processed/spm \
        --out-dir data/processed/tokenised
"""

import argparse
from pathlib import Path

import sentencepiece as spm


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_spm(
    input_file: Path,
    model_prefix: Path,
    vocab_size: int = 4000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
) -> None:
    """
    Train a SentencePiece BPE model.

    character_coverage=0.9995 is recommended for languages with large
    character sets like Sinhala (56 chars) and Tamil (247 chars).
    """
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )
    print(f"  Trained SPM model → {model_prefix}.model  (vocab={vocab_size})")


# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------

def load_spm(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def encode_file(sp: spm.SentencePieceProcessor, input_path: Path, output_path: Path) -> None:
    """Tokenise every line of input_path, write space-separated pieces to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = input_path.read_text(encoding="utf-8").splitlines()
    tokenised = [" ".join(sp.encode(line, out_type=str)) for line in lines]
    output_path.write_text("\n".join(tokenised), encoding="utf-8")
    print(f"  Tokenised {len(lines)} lines → {output_path}")


def decode_file(sp: spm.SentencePieceProcessor, input_path: Path, output_path: Path) -> None:
    """Detokenise space-separated piece sequences back to plain text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = input_path.read_text(encoding="utf-8").splitlines()
    decoded = [sp.decode(line.split()) for line in lines]
    output_path.write_text("\n".join(decoded), encoding="utf-8")
    print(f"  Detokenised {len(lines)} lines → {output_path}")


def encode_sentence(sp: spm.SentencePieceProcessor, sentence: str) -> list[str]:
    """Encode a single sentence to subword pieces (used at inference time)."""
    return sp.encode(sentence, out_type=str)


def decode_pieces(sp: spm.SentencePieceProcessor, pieces: list[str]) -> str:
    """Decode subword pieces back to plain text."""
    return sp.decode(pieces)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    si_path = Path(args.si)
    ta_path = Path(args.ta)
    model_dir = Path(args.model_dir)

    print("Training Sinhala BPE model ...")
    train_spm(si_path, model_dir / "si_spm", vocab_size=args.vocab_size)

    print("Training Tamil BPE model ...")
    train_spm(ta_path, model_dir / "ta_spm", vocab_size=args.vocab_size)

    print("Done.")


def cmd_apply(args: argparse.Namespace) -> None:
    si_path = Path(args.si)
    ta_path = Path(args.ta)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    si_sp = load_spm(model_dir / "si_spm.model")
    ta_sp = load_spm(model_dir / "ta_spm.model")

    stem = si_path.stem  # e.g. "train_filtered"
    encode_file(si_sp, si_path, out_dir / f"{stem}.si")
    encode_file(ta_sp, ta_path, out_dir / f"{stem}.ta")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SentencePiece BPE tokenisation")
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train BPE models for si and ta")
    p_train.add_argument("--si", required=True)
    p_train.add_argument("--ta", required=True)
    p_train.add_argument("--model-dir", default="data/processed/spm")
    p_train.add_argument("--vocab-size", type=int, default=4000)

    # apply
    p_apply = sub.add_parser("apply", help="Apply trained BPE models to a file pair")
    p_apply.add_argument("--si", required=True)
    p_apply.add_argument("--ta", required=True)
    p_apply.add_argument("--model-dir", default="data/processed/spm")
    p_apply.add_argument("--out-dir", default="data/processed/tokenised")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "apply":
        cmd_apply(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
