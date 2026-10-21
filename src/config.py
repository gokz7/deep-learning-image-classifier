"""Central configuration: every setting and path lives here, nothing hard-coded elsewhere."""
from dataclasses import dataclass, asdict
import torch

# CIFAR-10 per-channel mean/std (computed on the 50k training images)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Data
    data_dir: str = "data"
    val_size: int = 5000          # held out from the 50k train set for model selection
    batch_size: int = 128
    num_workers: int = 2          # Colab gives 2 CPU cores

    # Model
    num_classes: int = 10
    dropout: float = 0.3

    # Training
    epochs: int = 30
    lr: float = 0.1               # peak LR for OneCycle schedule
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    use_amp: bool = True          # mixed precision, used only if a GPU is present

    # Paths
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    mlflow_dir: str = "mlruns"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        return asdict(self)
