"""
Data loading and preprocessing for TriageNet.

Provides:
  - clean_text(): basic clinical-note cleaning
  - ChestXrayDataset: torch Dataset for the image branch
  - ClinicalNotesDataset: torch Dataset for the text branch
  - MultimodalDataset: pairs image + text + label for the fusion branch
  - build_transforms(): torchvision image transforms

The datasets expect a CSV with columns: note_id, note_text, image_path, label
and a base directory that image_path is relative to. This matches both the
synthetic sample data in data/sample/ and the format described for real data
in the README (data/raw/).
"""
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LABEL2IDX, MAX_TOKEN_LEN


def clean_text(text: str) -> str:
    """Light cleaning for clinical notes: keeps clinical shorthand (e.g. 'SpO2 91%'),
    just normalises whitespace and strips characters that break tokenisation."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9%.,\-\s]", " ", text)
    return text.strip()


def build_transforms(train: bool = True):
    """Standard ImageNet-style preprocessing for the DenseNet-121 branch."""
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.2))
    ops += [
        transforms.Grayscale(num_output_channels=3),  # X-rays are single-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def load_metadata(data_dir: str) -> pd.DataFrame:
    csv_path = Path(data_dir) / "clinical_notes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path}. See README section 'Datasets' for the required "
            "CSV format (note_id, note_text, image_path, label)."
        )
    df = pd.read_csv(csv_path)
    missing = {"note_id", "note_text", "image_path", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"clinical_notes.csv is missing columns: {missing}")
    df["label_idx"] = df["label"].map(LABEL2IDX)
    if df["label_idx"].isna().any():
        bad = df[df["label_idx"].isna()]["label"].unique()
        raise ValueError(f"Unrecognised label(s) in CSV: {bad}. Expected one of {list(LABEL2IDX)}")
    return df


class ChestXrayDataset(Dataset):
    def __init__(self, data_dir: str, df: pd.DataFrame = None, train: bool = True):
        self.data_dir = Path(data_dir)
        self.df = df if df is not None else load_metadata(data_dir)
        self.transform = build_transforms(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        label = int(row["label_idx"])
        return image, label


class ClinicalNotesDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, df: pd.DataFrame = None):
        self.data_dir = Path(data_dir)
        self.df = df if df is not None else load_metadata(data_dir)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = clean_text(row["note_text"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKEN_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = int(row["label_idx"])
        return item


class MultimodalDataset(Dataset):
    """Pairs an X-ray image and its clinical note for the fusion model."""

    def __init__(self, data_dir: str, tokenizer, df: pd.DataFrame = None, train: bool = True):
        self.data_dir = Path(data_dir)
        self.df = df if df is not None else load_metadata(data_dir)
        self.tokenizer = tokenizer
        self.img_transform = build_transforms(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.data_dir / row["image_path"]
        image = Image.open(img_path).convert("L")
        image = self.img_transform(image)

        text = clean_text(row["note_text"])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKEN_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        label = int(row["label_idx"])
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }


def train_val_test_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42):
    """Stratified-ish split by label to keep class balance reasonable across splits."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby("label"):
        n = len(group)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        test_parts.append(group.iloc[:n_test])
        val_parts.append(group.iloc[n_test:n_test + n_val])
        train_parts.append(group.iloc[n_test + n_val:])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, val_df, test_df
