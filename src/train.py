"""
CLI training script for TriageNet.

Trains any of the three model variants and saves a checkpoint + a small
training-curve plot to models/.

Usage:
    python train.py --modality image  --data-dir data/sample --epochs 5
    python train.py --modality text   --data-dir data/sample --epochs 5
    python train.py --modality fusion --data-dir data/sample --epochs 5 \
                     --image-ckpt models/image_model.pt --text-ckpt models/text_model.pt
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import (
    DEFAULT_BATCH_SIZE, DEFAULT_DATA_DIR, DEFAULT_EPOCHS, DEFAULT_LR,
    DEFAULT_MODELS_DIR, LABELS, NUM_CLASSES, RANDOM_SEED, TEXT_MODEL_NAME,
)
from data_preprocessing import (
    ChestXrayDataset, ClinicalNotesDataset, MultimodalDataset,
    load_metadata, train_val_test_split,
)
from image_model import XrayCNN
from text_model import ClinicalTextEncoder
from fusion_model import MultimodalFusionModel


def class_weights_from_df(df, num_classes=NUM_CLASSES):
    """Inverse-frequency class weights -- our labels are imbalanced (fewer
    Urgent cases than Normal in most real triage data), so this keeps the
    loss from just predicting the majority class."""
    counts = df["label_idx"].value_counts().reindex(range(num_classes), fill_value=0)
    weights = 1.0 / counts.clip(lower=1)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights.values, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, modality, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if modality == "image":
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
            elif modality == "text":
                y = batch["label"].to(device)
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            else:  # fusion
                y = batch["label"].to(device)
                logits = model(
                    batch["image"].to(device),
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )

            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser(description="Train a TriageNet model variant.")
    parser.add_argument("--modality", choices=["image", "text", "fusion"], required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--image-ckpt", default=None, help="Warm-start image branch (fusion only)")
    parser.add_argument("--text-ckpt", default=None, help="Warm-start text branch (fusion only)")
    parser.add_argument("--freeze-backbones", action="store_true",
                         help="Freeze image/text branches and only train the fusion head")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.data_dir)
    train_df, val_df, _ = train_val_test_split(df, seed=RANDOM_SEED)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  (Test split held out for evaluate.py)")

    tokenizer = None
    if args.modality in ("text", "fusion"):
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    if args.modality == "image":
        train_ds = ChestXrayDataset(args.data_dir, df=train_df, train=True)
        val_ds = ChestXrayDataset(args.data_dir, df=val_df, train=False)
        model = XrayCNN().to(device)
    elif args.modality == "text":
        train_ds = ClinicalNotesDataset(args.data_dir, tokenizer, df=train_df)
        val_ds = ClinicalNotesDataset(args.data_dir, tokenizer, df=val_df)
        model = ClinicalTextEncoder().to(device)
    else:
        train_ds = MultimodalDataset(args.data_dir, tokenizer, df=train_df, train=True)
        val_ds = MultimodalDataset(args.data_dir, tokenizer, df=val_df, train=False)
        model = MultimodalFusionModel(freeze_backbones=args.freeze_backbones).to(device)
        model.load_pretrained_branches(args.image_ckpt, args.text_ckpt)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    weights = class_weights_from_df(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    ckpt_path = models_dir / f"{args.modality}_model.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, args.modality, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, args.modality, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} "
              f"| val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best checkpoint to {ckpt_path}")

    # Save a training-curve plot for the report / demo slides.
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title(f"{args.modality} — loss"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title(f"{args.modality} — accuracy"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(models_dir / f"{args.modality}_training_curve.png", dpi=140)
    print(f"Saved training curve to {models_dir / (args.modality + '_training_curve.png')}")


if __name__ == "__main__":
    main()
