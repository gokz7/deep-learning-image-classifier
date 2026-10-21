"""Models built from scratch (no pre-trained weights).
- BaselineCNN   : plain VGG-style CNN (Part 1)
- ResidualCNN   : same size, with residual blocks (Part 2, Option A)
- DeepCNN       : deeper version (2 blocks per stage), built with either
                  PlainBlock (no shortcut) or ResidualBlock (with shortcut) - Step 7b
"""
import torch.nn as nn


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 conv -> BatchNorm -> ReLU. Padding=1 keeps the image size the same."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------- Baseline ----------
class BaselineCNN(nn.Module):
    """Small VGG-style CNN: 3 stages of (conv-bn-relu x2 + maxpool), then a classifier head."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 32x32 -> 16x16
            conv_bn_relu(3, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            # Stage 2: 16x16 -> 8x8
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(2),
            # Stage 3: 8x8 -> 4x4
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (256, 4, 4) -> (256, 1, 1)
            nn.Flatten(),              # -> 256 numbers
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------- Blocks ----------
class ResidualBlock(nn.Module):
    """Simplified ResNet basic block: out = ReLU( BN(conv(ReLU(BN(conv(x))))) + shortcut(x) )."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # If size or channels change, resize x with a 1x1 conv so it can be added
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # the residual connection
        return self.relu(out)


class PlainBlock(nn.Module):
    """Same two conv layers as ResidualBlock, but WITHOUT the shortcut (control model)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


# ---------- Residual (Part 2, Option A) ----------
class ResidualCNN(nn.Module):
    """Stem + 3 residual blocks (64 -> 128 -> 256 channels), same head as the baseline."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.stem = conv_bn_relu(3, 64)                    # 32x32
        self.layers = nn.Sequential(
            ResidualBlock(64, 64, stride=1),               # 32x32
            ResidualBlock(64, 128, stride=2),              # 16x16
            ResidualBlock(128, 256, stride=2),             # 8x8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (256, 8, 8) -> (256, 1, 1)
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return self.classifier(x)


# ---------- Deeper models (Step 7b depth experiment) ----------
class DeepCNN(nn.Module):
    """Stem + 3 stages x `blocks_per_stage` blocks. `block` is PlainBlock or ResidualBlock.
    With 2 blocks per stage: 1 + 3*2*2 = 13 conv layers."""

    def __init__(self, block, blocks_per_stage: int = 2, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.stem = conv_bn_relu(3, 64)
        layers, in_ch = [], 64
        for out_ch, stride in [(64, 1), (128, 2), (256, 2)]:
            for i in range(blocks_per_stage):
                layers.append(block(in_ch, out_ch, stride if i == 0 else 1))
                in_ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return self.classifier(x)


# ---------- Helpers ----------
def get_model(name: str, cfg) -> nn.Module:
    """Build a model by name, using settings from the config."""
    if name == "baseline":
        return BaselineCNN(num_classes=cfg.num_classes, dropout=cfg.dropout)
    if name == "residual":
        return ResidualCNN(num_classes=cfg.num_classes, dropout=cfg.dropout)
    if name == "deep_plain":
        return DeepCNN(PlainBlock, blocks_per_stage=2, num_classes=cfg.num_classes, dropout=cfg.dropout)
    if name == "deep_residual":
        return DeepCNN(ResidualBlock, blocks_per_stage=2, num_classes=cfg.num_classes, dropout=cfg.dropout)
    raise ValueError(f"Unknown model name: {name}")


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
