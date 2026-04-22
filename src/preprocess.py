#!/usr/bin/env python3
"""
preprocess.py
=============
Phase 1 of the Sinhala-Tamil Pivot MT PoC pipeline.

Responsibilities:
  - Load raw parallel text (Sinhala .si + Tamil .ta files, one sentence per line)
  - Unicode NFC normalise both sides
  - Strip leading/trailing whitespace, collapse internal whitespace
  - Length filter: remove pairs where either side has <5 or >100 whitespace-delimited tokens
  - Deduplicate exact source-side duplicates
  - Split into train / test sets (fixed 50-sentence PoC test set held out first)
  - Write processed files to data/processed/ and data/test_set/

Usage:
    python src/preprocess.py \
        --si  data/raw/corpus.si \
        --ta  data/raw/corpus.ta \
        --out data/processed \
        --test-out data/test_set \
        --test-size 50
"""

import argparse
import unicodedata
import re
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def nfc(text: str) -> str:
    """Apply Unicode NFC normalisation."""
    return unicodedata.normalize("NFC", text)


def clean_line(line: str) -> str:
    """NFC-normalise, strip edges, collapse internal whitespace."""
    line = nfc(line)
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line


def token_count(line: str) -> int:
    """Whitespace-delimited token count (works for Sinhala/Tamil scripts)."""
    return len(line.split())


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_parallel(si_path: Path, ta_path: Path) -> list[tuple[str, str]]:
    """Read and zip two parallel files into (si, ta) pairs."""
    si_lines = si_path.read_text(encoding="utf-8").splitlines()
    ta_lines = ta_path.read_text(encoding="utf-8").splitlines()

    if len(si_lines) != len(ta_lines):
        raise ValueError(
            f"Line count mismatch: {si_path} has {len(si_lines)} lines, "
            f"{ta_path} has {len(ta_lines)} lines."
        )

    return list(zip(si_lines, ta_lines))


def clean_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Apply clean_line to every sentence in every pair."""
    return [(clean_line(si), clean_line(ta)) for si, ta in pairs]


def length_filter(
    pairs: list[tuple[str, str]],
    min_tokens: int = 5,
    max_tokens: int = 100,
) -> list[tuple[str, str]]:
    """Remove pairs where either side is outside [min_tokens, max_tokens]."""
    kept = []
    dropped = 0
    for si, ta in pairs:
        si_n = token_count(si)
        ta_n = token_count(ta)
        if min_tokens <= si_n <= max_tokens and min_tokens <= ta_n <= max_tokens:
            kept.append((si, ta))
        else:
            dropped += 1
    print(f"  Length filter: kept {len(kept)}, dropped {dropped}")
    return kept


def deduplicate(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove pairs with duplicate Sinhala source sentences (keep first)."""
    seen: set[str] = set()
    result = []
    for si, ta in pairs:
        if si not in seen:
            seen.add(si)
            result.append((si, ta))
    removed = len(pairs) - len(result)
    print(f"  Deduplication: removed {removed} exact source duplicates")
    return result


def split_test(
    pairs: list[tuple[str, str]],
    test_size: int = 50,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Hold out a fixed test set.

    The test set is sampled first (before any model sees data) to prevent
    contamination.  The same seed is used throughout the project so the
    50-sentence set is always identical.
    """
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    test_indices = set(indices[:test_size])
    test = [pairs[i] for i in range(len(pairs)) if i in test_indices]
    train = [pairs[i] for i in range(len(pairs)) if i not in test_indices]
    print(f"  Split: {len(train)} train, {len(test)} test (seed={seed})")
    return train, test


def write_parallel(pairs: list[tuple[str, str]], out_dir: Path, stem: str) -> None:
    """Write (si, ta) pairs to <stem>.si and <stem>.ta in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    si_path = out_dir / f"{stem}.si"
    ta_path = out_dir / f"{stem}.ta"
    si_path.write_text("\n".join(si for si, _ in pairs), encoding="utf-8")
    ta_path.write_text("\n".join(ta for _, ta in pairs), encoding="utf-8")
    print(f"  Wrote {len(pairs)} pairs → {si_path} / {ta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Sinhala-Tamil parallel corpus")
    parser.add_argument("--si", required=True, help="Raw Sinhala file (one sentence per line)")
    parser.add_argument("--ta", required=True, help="Raw Tamil file (one sentence per line)")
    parser.add_argument("--out", default="data/processed", help="Output dir for train split")
    parser.add_argument("--test-out", default="data/test_set", help="Output dir for test split")
    parser.add_argument("--test-size", type=int, default=50, help="Number of test sentences")
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    si_path = Path(args.si)
    ta_path = Path(args.ta)

    print(f"Loading {si_path} + {ta_path} ...")
    pairs = load_parallel(si_path, ta_path)
    print(f"  Loaded {len(pairs)} pairs")

    print("Cleaning ...")
    pairs = clean_pairs(pairs)

    pairs = [(si, ta) for si, ta in pairs if si and ta]

    print("Length filtering ...")
    pairs = length_filter(pairs, args.min_tokens, args.max_tokens)

    print("Deduplicating ...")
    pairs = deduplicate(pairs)

    print("Splitting train / test ...")
    train, test = split_test(pairs, args.test_size, args.seed)

    print("Writing output ...")
    write_parallel(train, Path(args.out), "train")
    write_parallel(test, Path(args.test_out), "test")

    print("Done.")


if __name__ == "__main__":
    main()
