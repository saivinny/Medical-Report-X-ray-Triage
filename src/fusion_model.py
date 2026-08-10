"""
Multimodal fusion model: combines the CNN image branch and the BERT text
branch via late fusion (feature concatenation + MLP head).

This is deliberately a *late* fusion design -- each branch is trained (or
pretrained) to produce a good feature vector on its own, and the fusion head
only has to learn how to weigh the two streams. This keeps the model small
and easier to train/evaluate/explain than a fully joint architecture, which
suits a coursework timeline.
"""
import torch
import torch.nn as nn

from config import NUM_CLASSES
from image_model import XrayCNN
from text_model import ClinicalTextEncoder


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        freeze_backbones: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_branch = XrayCNN(num_classes=num_classes)
        self.text_branch = ClinicalTextEncoder(num_classes=num_classes)

        if freeze_backbones:
            # Useful once each branch has been pretrained separately (via
            # train.py --modality image / --modality text) -- freeze them and
            # only train the fusion head, which is much faster and less
            # prone to overfitting on a small dataset.
            for p in self.image_branch.parameters():
                p.requires_grad = False
            for p in self.text_branch.parameters():
                p.requires_grad = False

        fused_dim = self.image_branch.feature_dim + self.text_branch.feature_dim  # 1024 + 768
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        img_feats = self.image_branch.extract_features(image)
        txt_feats = self.text_branch.extract_features(input_ids, attention_mask)
        fused = torch.cat([img_feats, txt_feats], dim=1)
        return self.fusion_head(fused)

    def load_pretrained_branches(self, image_ckpt: str = None, text_ckpt: str = None):
        """Optionally warm-start each branch from the single-modality
        checkpoints produced by `train.py --modality image` / `--modality text`."""
        if image_ckpt:
            self.image_branch.load_state_dict(torch.load(image_ckpt, map_location="cpu"))
        if text_ckpt:
            self.text_branch.load_state_dict(torch.load(text_ckpt, map_location="cpu"))
