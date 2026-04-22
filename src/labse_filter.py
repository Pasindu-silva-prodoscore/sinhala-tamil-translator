#!/usr/bin/env python3
"""
labse_filter.py
===============
Phase 1b — LaBSE-based parallel sentence quality filtering.

Why:
  Even in "clean" government corpora, misaligned pairs, loose paraphrases,
  and OCR noise exist.  LaBSE (Language-agnostic BERT Sentence Embeddings,
  Feng et al. 2022) maps sentences from 100+ languages into a shared vector
  space.  Pairs where the cosine similarity of the embeddings is below a
  threshold are likely mistranslations or misalignments and are discarded.

Threshold:
  0.85 — recommended by Feng et al. (2022) for high-quality parallel data.
  Lowering to 0.75 recovers more pairs at the cost of some noise.

Usage:
    python src/labse_filter.py \
        --si  data/processed/train.si \
        --ta  data/processed/train.ta \
        --out data/processed \
        --threshold 0.85

Dependencies:
    sentence-transformers>=2.6.0  (pip install sentence-transformers)
"""

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/LaBSE"


def load_model() -> SentenceTransformer:
    print(f"Loading LaBSE model ({MODEL_NAME}) ...")
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch cosine similarity between rows of a and b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a_norm * b_norm).sum(axis=1)


def filter_by_labse(
    si_sentences: list[str],
    ta_sentences: list[str],
    model: SentenceTransformer,
    threshold: float = 0.85,
    batch_size: int = 64,
) -> tuple[list[str], list[str], list[float]]:
    """
    Encode all sentences and keep pairs above the similarity threshold.

    Returns:
        Filtered (si_sentences, ta_sentences, scores_kept)
    """
    print(f"Encoding {len(si_sentences)} Sinhala sentences ...")
    si_embs = model.encode(
        si_sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Encoding {len(ta_sentences)} Tamil sentences ...")
    ta_embs = model.encode(
        ta_sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Already normalised above, dot product == cosine similarity
    scores = (si_embs * ta_embs).sum(axis=1)

    kept_si, kept_ta, kept_scores = [], [], []
    dropped = 0
    for si, ta, score in zip(si_sentences, ta_sentences, scores):
        if score >= threshold:
            kept_si.append(si)
            kept_ta.append(ta)
            kept_scores.append(float(score))
        else:
            dropped += 1

    print(
        f"  LaBSE filter (threshold={threshold}): "
        f"kept {len(kept_si)}, dropped {dropped} "
        f"({dropped / max(len(si_sentences), 1) * 100:.1f}%)"
    )
    if kept_scores:
        print(
            f"  Score stats — mean: {np.mean(kept_scores):.4f}, "
            f"min: {np.min(kept_scores):.4f}, "
            f"max: {np.max(kept_scores):.4f}"
        )
    return kept_si, kept_ta, kept_scores


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LaBSE quality filtering for parallel corpus")
    parser.add_argument("--si", required=True, help="Sinhala input file")
    parser.add_argument("--ta", required=True, help="Tamil input file")
    parser.add_argument("--out", default="data/processed", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--suffix",
        default="filtered",
        help="Output file suffix, e.g. 'filtered' → train_filtered.si",
    )
    args = parser.parse_args()

    si_path = Path(args.si)
    ta_path = Path(args.ta)
    out_dir = Path(args.out)

    si_sentences = read_lines(si_path)
    ta_sentences = read_lines(ta_path)

    if len(si_sentences) != len(ta_sentences):
        raise ValueError(
            f"Line count mismatch: {si_path} ({len(si_sentences)}) "
            f"vs {ta_path} ({len(ta_sentences)})"
        )

    model = load_model()

    kept_si, kept_ta, scores = filter_by_labse(
        si_sentences, ta_sentences, model, args.threshold, args.batch_size
    )

    stem = si_path.stem  # e.g. "train"
    out_si = out_dir / f"{stem}_{args.suffix}.si"
    out_ta = out_dir / f"{stem}_{args.suffix}.ta"
    scores_path = out_dir / f"{stem}_{args.suffix}_scores.txt"

    write_lines(kept_si, out_si)
    write_lines(kept_ta, out_ta)
    write_lines([f"{s:.6f}" for s in scores], scores_path)

    print(f"  Wrote filtered pairs → {out_si} / {out_ta}")
    print(f"  Wrote similarity scores → {scores_path}")
    print("Done.")


if __name__ == "__main__":
    main()
