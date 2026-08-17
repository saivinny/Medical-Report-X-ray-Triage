"""
Report figures for TriageNet: confusion matrix heatmaps, ROC curves, and a
model-comparison bar chart -- the plots evaluate.py prints as numbers but
doesn't save as images.

Run this AFTER evaluate.py works (i.e. after training whichever models you
want plotted). It reuses the exact same prediction logic as evaluate.py, so
the numbers in these figures will match evaluate.py's printed output exactly.

Usage:
    python src/plot_report_figures.py --data-dir data/sample --models-dir models

Output (saved to <models-dir>/figures/):
    confusion_matrix_image.png   / _text.png / _fusion.png   (Figures 4-6)
    roc_curve_image.png / _text.png / _fusion.png             (Figures 7-8 etc.)
    model_comparison.png                                      (Figure 9)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix,
    f1_score, recall_score, roc_auc_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
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
from evaluate import get_predictions, specificity_for_urgent


def plot_confusion_matrix(y_true, y_pred, modality, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize": 11})
    ax.set_title(f"{modality.capitalize()} branch confusion matrix", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out_path = out_dir / f"confusion_matrix_{modality}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_roc_curves(y_true, probs, modality, out_dir):
    """One-vs-rest ROC curve per class, all on one figure."""
    palette = ["#1f3a5f", "#4a7ba6", "#c9843a"]  # navy / slate blue / amber accent
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, label in enumerate(LABELS):
        # Skip classes missing from this split (can happen on tiny sample data)
        if y_true_bin[:, i].sum() == 0 or y_true_bin[:, i].sum() == len(y_true_bin):
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})",
                color=palette[i % len(palette)], linewidth=2)
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"{modality.capitalize()} branch ROC curves by class", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out_path = out_dir / f"roc_curve_{modality}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_comparison_bar_chart(summary, out_dir):
    """Grouped bar chart: Urgent Recall / Macro F1 / AUC-ROC / Specificity
    side by side for every model variant that was evaluated."""
    metrics = ["urgent_recall", "macro_f1", "auc_roc", "specificity"]
    metric_labels = ["Urgent Recall", "Macro F1", "AUC-ROC", "Specificity"]
    models = list(summary.keys())

    # Professional, muted palette (navy / slate blue / teal) instead of
    # matplotlib's default red/orange/green cycle.
    palette = ["#1f3a5f", "#4a7ba6", "#8fb8c9", "#c9d6df"]
    colors = [palette[i % len(palette)] for i in range(len(models))]

    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model_name in enumerate(models):
        values = [summary[model_name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name.capitalize(),
                       color=colors[i], edgecolor="white", linewidth=0.6)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, color="#333333")

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model comparison on the held-out test split", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out_path = out_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures for TriageNet.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_metadata(args.data_dir)
    _, _, test_df = train_val_test_split(df, seed=RANDOM_SEED)

    models_dir = Path(args.models_dir)
    out_dir = models_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    summary = {}

    image_ckpt = models_dir / "image_model.pt"
    if image_ckpt.exists():
        model = XrayCNN().to(device)
        model.load_state_dict(torch.load(image_ckpt, map_location=device))
        loader = DataLoader(ChestXrayDataset(args.data_dir, df=test_df, train=False), batch_size=args.batch_size)
        probs, y_true = get_predictions(model, loader, device, "image")
        y_pred = probs.argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred, "image", out_dir)
        plot_roc_curves(y_true, probs, "image", out_dir)
        summary["image"] = {
            "urgent_recall": recall_score(y_true, y_pred, labels=[URGENT_IDX], average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "auc_roc": _safe_auc(y_true, probs),
            "specificity": specificity_for_urgent(y_true, y_pred),
        }
    else:
        print(f"(skipping image model -- no checkpoint at {image_ckpt})")

    text_ckpt = models_dir / "text_model.pt"
    if text_ckpt.exists():
        model = ClinicalTextEncoder().to(device)
        model.load_state_dict(torch.load(text_ckpt, map_location=device))
        loader = DataLoader(ClinicalNotesDataset(args.data_dir, tokenizer, df=test_df), batch_size=args.batch_size)
        probs, y_true = get_predictions(model, loader, device, "text")
        y_pred = probs.argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred, "text", out_dir)
        plot_roc_curves(y_true, probs, "text", out_dir)
        summary["text"] = {
            "urgent_recall": recall_score(y_true, y_pred, labels=[URGENT_IDX], average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "auc_roc": _safe_auc(y_true, probs),
            "specificity": specificity_for_urgent(y_true, y_pred),
        }
    else:
        print(f"(skipping text model -- no checkpoint at {text_ckpt})")

    fusion_ckpt = models_dir / "fusion_model.pt"
    if fusion_ckpt.exists():
        model = MultimodalFusionModel().to(device)
        model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        loader = DataLoader(MultimodalDataset(args.data_dir, tokenizer, df=test_df, train=False), batch_size=args.batch_size)
        probs, y_true = get_predictions(model, loader, device, "fusion")
        y_pred = probs.argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred, "fusion", out_dir)
        plot_roc_curves(y_true, probs, "fusion", out_dir)
        summary["fusion"] = {
            "urgent_recall": recall_score(y_true, y_pred, labels=[URGENT_IDX], average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "auc_roc": _safe_auc(y_true, probs),
            "specificity": specificity_for_urgent(y_true, y_pred),
        }
    else:
        print(f"(skipping fusion model -- no checkpoint at {fusion_ckpt})")

    if summary:
        plot_comparison_bar_chart(summary, out_dir)
        print(f"\nAll figures saved to: {out_dir.resolve()}")
    else:
        print("\nNo trained models found -- nothing to plot. Train at least one model first.")


def _safe_auc(y_true, probs):
    try:
        return roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    main()
