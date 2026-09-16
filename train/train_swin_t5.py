"""
Fine-tuning script for the Swin-T5 chest X-ray report generator.

Example:
    python train/train_swin_t5.py \
        --csv CheXpert-v1.0/train.csv \
        --images-root CheXpert-v1.0 \
        --epochs 5 --batch-size 8 --output checkpoints/swin_t5_cxr.pt

Produces a checkpoint compatible with ``SWIN_T5_CHECKPOINT`` in .env /
config.py, i.e. a plain ``state_dict`` for
``models.vlm_report_generator.SwinT5ForReportGeneration``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from models.vlm_report_generator import SwinT5ForReportGeneration  # noqa: E402
from train.dataset import CheXpertReportDataset, collate_fn  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CheXpert-style CSV with a Report column")
    p.add_argument("--images-root", required=True, help="Root directory the CSV's Path column is relative to")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=220)
    p.add_argument("--output", default="checkpoints/swin_t5_cxr.pt")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or Config.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoFeatureExtractor, T5Tokenizer

    feature_extractor = AutoFeatureExtractor.from_pretrained(Config.SWIN_ENCODER_ID)
    tokenizer = T5Tokenizer.from_pretrained(Config.T5_DECODER_ID)

    dataset = CheXpertReportDataset(
        csv_path=args.csv,
        images_root=args.images_root,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = SwinT5ForReportGeneration(Config.SWIN_ENCODER_ID, Config.T5_DECODER_ID).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / (progress.n + 1))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved checkpoint to {output_path}")
    print(f"Set SWIN_T5_CHECKPOINT={output_path} in your .env to use it in the app.")


if __name__ == "__main__":
    main()
