"""
Text branch: ClinicalBERT / BioBERT encoder for clinical notes.

Wraps a HuggingFace transformer (default: emilyalsentzer/Bio_ClinicalBERT,
trained on MIMIC-III notes) with a classification head for our 3-way
triage labels. Exposes the pooled [CLS] representation so the fusion
model can reuse this branch's features directly.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from config import NUM_CLASSES, TEXT_MODEL_NAME


class ClinicalTextEncoder(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, model_name: str = TEXT_MODEL_NAME, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.encoder.config.hidden_size  # 768 for BERT-base variants

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns the pooled [CLS] token representation, used by both the
        text-only classifier and the fusion model."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Some BERT variants expose pooler_output; fall back to CLS token if not.
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(input_ids, attention_mask)
        return self.classifier(feats)