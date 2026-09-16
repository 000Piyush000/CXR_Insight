"""
CheXpert dataset loader for training the Swin-T5 report generator.

Expects the standard CheXpert-v1.0 layout:

    CheXpert-v1.0/
      train.csv
      train/patientXXXXX/studyN/view1_frontal.jpg

train.csv must have a ``Path`` column and a free-text ``Report`` /
``report`` column with the ground-truth radiology report. CheXpert's
public release ships only pathology labels, not free-text reports; if
you are training on CheXpert directly, pair it with a report-generation
split (e.g. CheXpert-Plus) or supply your own ``Report`` column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CheXpertReportDataset(Dataset):
    def __init__(self, csv_path: str, images_root: str, feature_extractor, tokenizer, max_length: int = 220):
        self.df = pd.read_csv(csv_path)
        report_col = next((c for c in self.df.columns if c.lower() == "report"), None)
        if report_col is None:
            raise ValueError(
                f"No 'Report' column found in {csv_path}. CheXpert's default "
                "release has no free-text reports - see the module docstring."
            )
        self.report_col = report_col
        self.images_root = Path(images_root)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_root / row["Path"]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]

        report_text = str(row[self.report_col])
        labels = self.tokenizer(
            report_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).input_ids[0]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}
