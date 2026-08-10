"""
Explainability tools for TriageNet, matching the "Critical Evaluation"
section of the proposal:

  - Grad-CAM for the image branch: highlights which regions of the X-ray
    most influenced the prediction.
  - Gradient x input token saliency for the text branch: ranks which words
    in the clinical note most influenced the prediction (a lightweight
    stand-in for full attention-attribution / Captum-based analysis).

Usage:
    python explainability.py --image path/to/xray.png --note "severe chest pain..."
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

from config import IDX2LABEL, TEXT_MODEL_NAME
from data_preprocessing import build_transforms, clean_text
from image_model import XrayCNN
from text_model import ClinicalTextEncoder


def grad_cam(model: XrayCNN, image_tensor: torch.Tensor, target_class: int = None):
    """Standard Grad-CAM: weight the last conv layer's feature maps by the
    gradient of the target class score w.r.t. those feature maps."""
    model.eval()
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output

    def bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    layer = model.last_conv_layer()
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)
    logits = model(image_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    model.zero_grad()
    logits[0, target_class].backward()

    acts = activations["value"][0]         # (C, H, W)
    grads = gradients["value"][0]          # (C, H, W)
    weights = grads.mean(dim=(1, 2))       # (C,) global-average-pooled gradients

    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)

    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy(), target_class, torch.softmax(logits, dim=1)[0].detach().cpu().numpy()


def overlay_cam_on_image(pil_image: Image.Image, cam: np.ndarray, out_path: Path):
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_image.size)
    cam_arr = np.array(cam_resized) / 255.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pil_image.convert("L"), cmap="gray")
    ax.imshow(cam_arr, cmap="jet", alpha=0.4)
    ax.set_title("Grad-CAM: model attention over the X-ray")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def token_saliency(model: ClinicalTextEncoder, tokenizer, text: str, target_class: int = None, top_k: int = 8):
    """Gradient x input-embedding saliency: ranks tokens by how much they
    push the target class's logit up or down. A lightweight, dependency-free
    alternative to Captum's Integrated Gradients for this coursework scope."""
    model.eval()
    enc = tokenizer(clean_text(text), return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    embedding_layer = model.encoder.get_input_embeddings()
    input_embeds = embedding_layer(input_ids).clone().detach().requires_grad_(True)

    out = model.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask)
    pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
    logits = model.classifier(pooled)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    model.zero_grad()
    logits[0, target_class].backward()

    grad_x_input = (input_embeds.grad * input_embeds).sum(dim=-1).squeeze(0)  # (seq_len,)
    scores = grad_x_input.detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    ranked = sorted(zip(tokens, scores), key=lambda t: -abs(t[1]))
    ranked = [(t, s) for t, s in ranked if t not in tokenizer.all_special_tokens]
    return ranked[:top_k], target_class, torch.softmax(logits, dim=1)[0].detach().numpy()


def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM (image) and token saliency (text) explainability.")
    parser.add_argument("--image", required=True, help="Path to a chest X-ray image")
    parser.add_argument("--note", required=True, help="Clinical note text")
    parser.add_argument("--image-ckpt", default="models/image_model.pt")
    parser.add_argument("--text-ckpt", default="models/text_model.pt")
    parser.add_argument("--out-dir", default="models/explain_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- image side ---
    img_model = XrayCNN()
    ckpt = Path(args.image_ckpt)
    if ckpt.exists():
        img_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        print(f"(no image checkpoint found at {ckpt} -- using ImageNet-initialised weights for demo purposes)")

    pil_image = Image.open(args.image).convert("L")
    tensor = build_transforms(train=False)(pil_image)
    cam, img_class, img_probs = grad_cam(img_model, tensor)
    overlay_cam_on_image(pil_image, cam, out_dir / "gradcam_overlay.png")
    print(f"Image branch prediction : {IDX2LABEL[img_class]}  (probs={np.round(img_probs, 3)})")
    print(f"Saved Grad-CAM overlay to {out_dir / 'gradcam_overlay.png'}")

    # --- text side ---
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    txt_model = ClinicalTextEncoder()
    ckpt = Path(args.text_ckpt)
    if ckpt.exists():
        txt_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        print(f"(no text checkpoint found at {ckpt} -- using pretrained-only weights for demo purposes)")

    ranked_tokens, txt_class, txt_probs = token_saliency(txt_model, tokenizer, args.note)
    print(f"\nText branch prediction  : {IDX2LABEL[txt_class]}  (probs={np.round(txt_probs, 3)})")
    print("Top contributing tokens (token, signed saliency score):")
    for tok, score in ranked_tokens:
        print(f"  {tok:<15} {score:+.4f}")


if __name__ == "__main__":
    main()
