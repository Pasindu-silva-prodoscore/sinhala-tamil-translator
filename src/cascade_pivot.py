#!/usr/bin/env python3
"""
cascade_pivot.py
================
Phase 4 — Cascade pivot translation pipeline: Sinhala → English → Tamil.

Architecture (Wu & Wang 2009, sentence-cascade variant):
  Step 1  Si → En   Helsinki-NLP/opus-mt-inc-en  (HuggingFace, Indic→English multilingual)
  Step 2  En → Ta   ai4bharat/indictrans2-en-indic-1B  (HuggingFace, pretrained)

For the PoC both models are used inference-only (no fine-tuning).
The parameter freezing hook (Zhang et al. 2022) is demonstrated as a utility
function to be applied during full training.

Intermediate English outputs are saved so error propagation can be analysed
separately.

Usage:
    python src/cascade_pivot.py \
        --si  data/test_set/test.si \
        --out models/cascade \
        --device cpu

Outputs:
    models/cascade/intermediate_en.txt   (Step 1 output)
    models/cascade/final_ta.txt          (Step 2 output — final hypothesis)
"""

import argparse
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Step 1: Sinhala → English
# ---------------------------------------------------------------------------

SI_EN_MODEL = "Helsinki-NLP/opus-mt-inc-en"
SI_LANG_TAG = ">>sin<<"


def load_si_en(device: str = "cpu"):
    print(f"Loading Si→En model ({SI_EN_MODEL}) ...")
    tokenizer = MarianTokenizer.from_pretrained(SI_EN_MODEL)
    model = MarianMTModel.from_pretrained(SI_EN_MODEL).to(device)
    model.eval()
    return tokenizer, model


def translate_si_en(
    sentences: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: str = "cpu",
    batch_size: int = 16,
) -> list[str]:
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = [f"{SI_LANG_TAG} {s}" for s in sentences[i : i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            translated = model.generate(**inputs, num_beams=4, max_new_tokens=256)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(decoded)
        print(f"  Si→En: {min(i + batch_size, len(sentences))}/{len(sentences)}")
    return results


# ---------------------------------------------------------------------------
# Step 2: English → Tamil  (IndicTrans2)
# ---------------------------------------------------------------------------

EN_TA_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_SRC_LANG = "eng_Latn"
NLLB_TGT_LANG = "tam_Taml"


def load_en_ta(device: str = "cpu"):
    print(f"Loading En→Ta model ({EN_TA_MODEL}) ...")
    tokenizer = AutoTokenizer.from_pretrained(EN_TA_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(EN_TA_MODEL).to(device)
    model.eval()
    return tokenizer, model


def translate_en_ta(
    sentences: list[str],
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 8,
) -> list[str]:
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            src_lang=NLLB_SRC_LANG,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        tgt_lang_id = tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                num_beams=5,
                max_new_tokens=256,
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        results.extend(decoded)

        del inputs, generated_tokens
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"  En→Ta: {min(i + batch_size, len(sentences))}/{len(sentences)}")
    return results

        print(f"  En→Ta: {min(i + batch_size, len(sentences))}/{len(sentences)}")
    return results


# ---------------------------------------------------------------------------
# Full cascade
# ---------------------------------------------------------------------------

def run_cascade(
    si_sentences: list[str],
    device: str = "cpu",
    batch_size_si_en: int = 16,
    batch_size_en_ta: int = 8,
) -> tuple[list[str], list[str]]:
    """
    Run the full Si→En→Ta cascade.

    Returns:
        (intermediate_en, final_ta)
    """
    si_en_tok, si_en_model = load_si_en(device)
    en_sentences = translate_si_en(si_sentences, si_en_tok, si_en_model, device, batch_size_si_en)

    del si_en_model
    if device == "cuda":
        torch.cuda.empty_cache()

    en_ta_tok, en_ta_model = load_en_ta(device)
    ta_sentences = translate_en_ta(en_sentences, en_ta_tok, en_ta_model, device, batch_size_en_ta)

    return en_sentences, ta_sentences


# ---------------------------------------------------------------------------
# Parameter freezing utility (Zhang et al. 2022)
# ---------------------------------------------------------------------------

def freeze_encoder(model: torch.nn.Module) -> None:
    """
    Freeze all encoder parameters after Stage 1 of cascade training.

    Prevents the second-stage fine-tune from overwriting the source-language
    representations learned in Stage 1 (Zhang, Li and Liu 2022).

    Usage during full training:
        # After training Si→En model:
        freeze_encoder(si_en_model)
        # Then continue training the full cascade.
    """
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False
    frozen = sum(1 for n, p in model.named_parameters() if not p.requires_grad)
    print(f"  freeze_encoder: froze {frozen} parameter tensors")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade pivot Si→En→Ta translation")
    parser.add_argument("--si", required=True, help="Input Sinhala file (one sentence per line)")
    parser.add_argument("--out", default="models/cascade", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-si-en", type=int, default=16)
    parser.add_argument("--batch-en-ta", type=int, default=8)
    args = parser.parse_args()

    si_path = Path(args.si)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    si_sentences = si_path.read_text(encoding="utf-8").splitlines()
    si_sentences = [s for s in si_sentences if s.strip()]
    print(f"Loaded {len(si_sentences)} Sinhala sentences from {si_path}")

    en_sentences, ta_sentences = run_cascade(
        si_sentences,
        device=args.device,
        batch_size_si_en=args.batch_si_en,
        batch_size_en_ta=args.batch_en_ta,
    )

    en_out = out_dir / "intermediate_en.txt"
    ta_out = out_dir / "final_ta.txt"
    en_out.write_text("\n".join(en_sentences), encoding="utf-8")
    ta_out.write_text("\n".join(ta_sentences), encoding="utf-8")

    print(f"  Intermediate English → {en_out}")
    print(f"  Final Tamil          → {ta_out}")
    print("Done.")


if __name__ == "__main__":
    main()
