"""
Image branch: DenseNet-121 CNN classifier for chest X-rays.

Uses transfer learning from ImageNet weights (torchvision), replacing the
final classifier layer with a small head for our 3-way triage labels.
Exposes penultimate features so the fusion model can reuse this branch
without duplicating the backbone.
"""
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

from config import NUM_CLASSES


class XrayCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        # torchvision's densenet121 exposes `.features` (conv layers) and
        # `.classifier` (final Linear(1024, 1000)). We keep `.features` as the
        # backbone and replace the classifier with our own head.
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1024  # DenseNet-121's final feature width

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the pooled 1024-d feature vector used by both the
        image-only classifier and the fusion model."""
        feats = self.features(x)
        feats = torch.relu(feats)
        feats = self.pool(feats).flatten(1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.classifier(feats)

    def last_conv_layer(self):
        """Returns the last convolutional layer -- used by explainability.py
        for Grad-CAM hook registration."""
        return self.features.norm5
