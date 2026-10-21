"""Unit tests for models, transforms and reproducibility.
Run from the project root:  python -m pytest tests -v
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.config import Config
from src.data import get_transforms
from src.models import get_model, count_params, ResidualBlock
from src.utils import set_seed

CFG = Config()
EXPECTED_PARAMS = {
    "baseline": 1_148_874,
    "residual": 1_227_594,
    "deep_plain": 2_735_946,
    "deep_residual": 2_777_674,
}


@pytest.mark.parametrize("name", list(EXPECTED_PARAMS))
def test_output_shape(name):
    model = get_model(name, CFG).eval()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


@pytest.mark.parametrize("name, expected", list(EXPECTED_PARAMS.items()))
def test_param_count(name, expected):
    assert count_params(get_model(name, CFG)) == expected


def test_residual_block_stride1_keeps_shape():
    block = ResidualBlock(64, 64, stride=1)
    assert isinstance(block.shortcut, nn.Identity)          # plain identity shortcut
    assert block(torch.randn(2, 64, 32, 32)).shape == (2, 64, 32, 32)


def test_residual_block_stride2_downsamples():
    block = ResidualBlock(64, 128, stride=2)
    assert not isinstance(block.shortcut, nn.Identity)      # 1x1 conv shortcut
    assert block(torch.randn(2, 64, 32, 32)).shape == (2, 128, 16, 16)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_model("resnet50", CFG)


def test_same_seed_gives_identical_weights():
    set_seed(0)
    a = get_model("baseline", CFG)
    set_seed(0)
    b = get_model("baseline", CFG)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_training_steps_reduce_loss():
    set_seed(0)
    model = get_model("baseline", CFG).train()
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    first_loss = loss_fn(model(x), y).item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first_loss


def test_transforms_shape_and_normalization():
    train_tf, eval_tf = get_transforms()
    random_img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    assert train_tf(random_img).shape == (3, 32, 32)
    assert eval_tf(random_img).shape == (3, 32, 32)

    # An image filled with the dataset's average colour should normalize to about 0
    mean_img = Image.fromarray(np.full((32, 32, 3), [125, 123, 114], dtype=np.uint8))
    assert eval_tf(mean_img).abs().max() < 0.02
