"""
Basic sanity tests for TriageNet models -- checks forward-pass shapes so
CI (or a quick local `pytest`) catches broken wiring before a training run
wastes time. These do not test model *accuracy*; see evaluate.py for that.

Run with:
    cd triagenet-project/src && pytest ../tests/test_pipeline.py -v
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import NUM_CLASSES, IMAGE_SIZE, MAX_TOKEN_LEN
from image_model import XrayCNN
from text_model import ClinicalTextEncoder
from fusion_model import MultimodalFusionModel


def test_image_model_forward_shape():
    model = XrayCNN(pretrained=False)
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(dummy)
    assert out.shape == (2, NUM_CLASSES)


def test_image_model_feature_dim():
    model = XrayCNN(pretrained=False)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    feats = model.extract_features(dummy)
    assert feats.shape == (1, model.feature_dim)


def test_text_model_forward_shape():
    model = ClinicalTextEncoder()
    input_ids = torch.randint(0, 1000, (2, MAX_TOKEN_LEN))
    attention_mask = torch.ones(2, MAX_TOKEN_LEN, dtype=torch.long)
    out = model(input_ids, attention_mask)
    assert out.shape == (2, NUM_CLASSES)


def test_fusion_model_forward_shape():
    model = MultimodalFusionModel()
    image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, 1000, (2, MAX_TOKEN_LEN))
    attention_mask = torch.ones(2, MAX_TOKEN_LEN, dtype=torch.long)
    out = model(image, input_ids, attention_mask)
    assert out.shape == (2, NUM_CLASSES)


def test_fusion_output_is_valid_distribution_after_softmax():
    model = MultimodalFusionModel()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    input_ids = torch.randint(0, 1000, (1, MAX_TOKEN_LEN))
    attention_mask = torch.ones(1, MAX_TOKEN_LEN, dtype=torch.long)
    logits = model(image, input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
