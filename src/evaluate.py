"""
Evaluation script for TriageNet.

Loads trained checkpoints for whichever of image / text / fusion models are
available in --models-dir, runs them on the held-out test split, and reports:

  - Recall (per class, with special attention to the Urgent class -- our
    primary success metric, target >= 90%)
  - Macro F1-score
  - AUC-ROC (one-vs-rest, macro-averaged)
  - Specificity (Urgent vs rest, binarised)
  - Confusion matrix

...for each available model, so the group can directly compare the fused
model against the single-modality baselines as described in the proposal's
evaluation strategy.

Usage:
    python evaluate.py --data-dir data/sample --models-dir models
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import (
    DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, LABELS, NUM_CLASSES,
    RANDOM_SEED, TEXT_MODEL_NAME, URGENT_IDX,
)
from data_preprocessing import (
    ChestXrayDataset, ClinicalNotesDataset, MultimodalDataset,
    load_metadata, train_val_test_split,
)
from image_model import XrayCNN
from text_model import ClinicalTextEncoder
from fusion_model import MultimodalFusionModel


def specificity_for_urgent(y_true, y_pred, urgent_idx=URGENT_IDX):
    """Specificity = TN / (TN + FP), binarising 'Urgent' vs 'everything else'.
    This tells us how good the model is at NOT crying wolf on non-urgent
    cases -- important since over-triage burns clinician time and trust."""
    y_true_bin = (np.array(y_true) == urgent_idx).astype(int)
    y_pred_bin = (np.array(y_pred) == urgent_idx).astype(int)
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


@torch.no_grad()
def get_predictions(model, loader, device, modality):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        if modality == "image":
            x, y = batch
            logits = model(x.to(device))
        elif modality == "text":
            y = batch["label"]
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        else:
            y = batch["label"]
            logits = model(
                batch["image"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(y.tolist() if torch.is_tensor(y) else y)
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def report_for_modality(modality, model, loader, device):
    probs, y_true = get_predictions(model, loader, device, modality)
    y_pred = probs.argmax(axis=1)

    print(f"\n=== {modality.upper()} MODEL ===")
    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    urgent_recall = recall_score(y_true, y_pred, labels=[URGENT_IDX], average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    specificity = specificity_for_urgent(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # can happen on tiny sample data if a class is missing from a split

    print(f"Urgent recall     : {urgent_recall:.3f}  (target >= 0.90)")
    print(f"Macro F1-score     : {macro_f1:.3f}")
    print(f"AUC-ROC (macro OvR): {auc:.3f}")
    print(f"Specificity (Urgent vs rest): {specificity:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    return {"urgent_recall": urgent_recall, "macro_f1": macro_f1, "auc_roc": auc, "specificity": specificity}


def main():
    parser = argparse.ArgumentParser(description="Evaluate & compare TriageNet model variants.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_metadata(args.data_dir)
    _, _, test_df = train_val_test_split(df, seed=RANDOM_SEED)
    print(f"Evaluating on held-out test split: {len(test_df)} examples")

    models_dir = Path(args.models_dir)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    summary = {}

    image_ckpt = models_dir / "image_model.pt"
    if image_ckpt.exists():
        model = XrayCNN().to(device)
        model.load_state_dict(torch.load(image_ckpt, map_location=device))
        loader = DataLoader(ChestXrayDataset(args.data_dir, df=test_df, train=False), batch_size=args.batch_size)
        summary["image"] = report_for_modality("image", model, loader, device)
    else:
        print(f"\n(skipping image model -- no checkpoint at {image_ckpt}; run train.py --modality image first)")

    text_ckpt = models_dir / "text_model.pt"
    if text_ckpt.exists():
        model = ClinicalTextEncoder().to(device)
        model.load_state_dict(torch.load(text_ckpt, map_location=device))
        loader = DataLoader(ClinicalNotesDataset(args.data_dir, tokenizer, df=test_df), batch_size=args.batch_size)
        summary["text"] = report_for_modality("text", model, loader, device)
    else:
        print(f"\n(skipping text model -- no checkpoint at {text_ckpt}; run train.py --modality text first)")

    fusion_ckpt = models_dir / "fusion_model.pt"
    if fusion_ckpt.exists():
        model = MultimodalFusionModel().to(device)
        model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        loader = DataLoader(MultimodalDataset(args.data_dir, tokenizer, df=test_df, train=False), batch_size=args.batch_size)
        summary["fusion"] = report_for_modality("fusion", model, loader, device)
    else:
        print(f"\n(skipping fusion model -- no checkpoint at {fusion_ckpt}; run train.py --modality fusion first)")

    if summary:
        print("\n=== COMPARISON TABLE (copy into report) ===")
        print(f"{'Model':<10}{'Urgent Recall':<16}{'Macro F1':<12}{'AUC-ROC':<12}{'Specificity':<12}")
        for name, m in summary.items():
            print(f"{name:<10}{m['urgent_recall']:<16.3f}{m['macro_f1']:<12.3f}{m['auc_roc']:<12.3f}{m['specificity']:<12.3f}")


if __name__ == "__main__":
    main()
