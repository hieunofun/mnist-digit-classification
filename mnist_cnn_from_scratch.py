"""
Train CNN models from scratch for MNIST with PyTorch.

This script follows the same data/training flow as the MLP notebook:
load data, normalize, build TensorDataset/DataLoader, train, evaluate, and
visualize errors. It uses only the local mnist.npz file when available.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist_npz(data_path: str | os.PathLike = "mnist.npz") -> tuple[np.ndarray, ...]:
    """Load the local MNIST file first, then fall back to Keras if needed."""
    path = Path(data_path)
    if path.exists():
        data = np.load(path)
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise FileNotFoundError(
            f"Could not find {path}. Put mnist.npz in this folder or install TensorFlow "
            "so tf.keras.datasets.mnist.load_data() can download it."
        ) from exc

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


def build_loaders(
    data_path: str | os.PathLike = "mnist.npz",
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    x_train, y_train, x_test, y_test = load_mnist_npz(data_path)

    # CNN input shape: [N, C, H, W] = [N, 1, 28, 28].
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = (x_train - MNIST_MEAN) / MNIST_STD
    x_test = (x_test - MNIST_MEAN) / MNIST_STD

    x_train_tensor = torch.tensor(x_train).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


class MNISTCNN(nn.Module):
    """A compact CNN that reliably reaches >99% test accuracy on MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class StackedCNN(nn.Module):
    """Deeper CNN with configurable number of convolution layers."""

    def __init__(self, channels: tuple[int, ...] = (32, 32, 32, 64, 64, 64, 128, 128), use_bn: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1

        for idx, out_channels in enumerate(channels):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if idx in {2, 5}:
                layers.append(nn.MaxPool2d(2))
                layers.append(nn.Dropout2d(0.10 if idx == 2 else 0.20))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 1, 28, 28)).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TooManyPoolingCNN(nn.Module):
    """Intentionally bad architecture: repeated pooling collapses 28x28 maps."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in (16, 32, 64, 128, 128):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, 100.0 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
) -> tuple[nn.Module, dict[str, list[float]]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = run_one_epoch(model, test_loader, criterion, None, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%"
        )

    model.load_state_dict(best_state)
    print(f"Best test accuracy: {best_acc:.2f}%")
    return model, history


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["test_loss"], marker="o", label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["test_acc"], marker="o", label="Test accuracy")
    plt.axhline(99.0, color="red", linestyle="--", linewidth=1, label="Target 99%")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "cnn_training_history.png", dpi=160)
    plt.close()


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    images_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device, non_blocking=True))
            probs = torch.softmax(outputs, dim=1)
            conf, preds = probs.max(dim=1)

            images_all.append(images.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            probs_all.append(conf.cpu().numpy())

    return (
        np.concatenate(images_all),
        np.concatenate(labels_all),
        np.concatenate(preds_all),
        np.concatenate(probs_all),
    )


def plot_mistakes(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_images: int = 12,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images, labels, preds, confs = collect_predictions(model, test_loader, device)
    wrong = np.where(preds != labels)[0]

    if len(wrong) == 0:
        print("No misclassified images found.")
        return

    # Show the most confident wrong predictions first: these are useful error cases.
    wrong = wrong[np.argsort(confs[wrong])[::-1]][:max_images]
    cols = min(6, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))

    plt.figure(figsize=(2.4 * cols, 2.8 * rows))
    for i, idx in enumerate(wrong, start=1):
        image = images[idx, 0] * MNIST_STD + MNIST_MEAN
        plt.subplot(rows, cols, i)
        plt.imshow(image, cmap="gray")
        plt.title(f"True: {labels[idx]} | Pred: {preds[idx]}\nconf={confs[idx]:.2f}", color="red")
        plt.axis("off")

    plt.suptitle("Misclassified MNIST test images", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_misclassified_examples.png", dpi=160)
    plt.close()
    print(f"Saved {len(wrong)} misclassified examples to {output_dir / 'cnn_misclassified_examples.png'}")


def plot_confusion_matrix(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, labels, preds, _ = collect_predictions(model, test_loader, device)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("MNIST test confusion matrix")

    threshold = cm.max() / 2
    for row in range(10):
        for col in range(10):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "cnn_confusion_matrix.png", dpi=160)
    plt.close()


def show_pooling_failure(device: torch.device) -> None:
    try:
        TooManyPoolingCNN().to(device)(torch.zeros(1, 1, 28, 28, device=device))
    except RuntimeError as exc:
        print("Expected issue when stacking too many pool layers:")
        print(str(exc).splitlines()[0])


def run_deep_experiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> None:
    experiments = [
        ("base_4_conv_bn", MNISTCNN()),
        ("deep_8_conv_no_bn", StackedCNN(use_bn=False)),
        ("deep_8_conv_bn", StackedCNN(use_bn=True)),
    ]

    print("\nDeep CNN stacking experiment")
    print("name, parameters, best_test_accuracy")
    for name, model in experiments:
        print(f"\nTraining {name} ({count_parameters(model):,} parameters)")
        _, history = train_model(model, train_loader, test_loader, device, epochs=epochs)
        print(f"{name}, {count_parameters(model):,}, {max(history['test_acc']):.2f}%")

    show_pooling_failure(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN from scratch on MNIST.")
    parser.add_argument("--data", default="mnist.npz", help="Path to mnist.npz")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for the main CNN")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs", help="Folder for generated figures")
    parser.add_argument("--deep-experiment", action="store_true", help="Also compare deeper CNN variants")
    parser.add_argument("--deep-epochs", type=int, default=3, help="Epochs for each deep experiment model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)

    print(f"Device: {device}")
    train_loader, test_loader = build_loaders(args.data, batch_size=args.batch_size)
    model = MNISTCNN()
    print(f"Model parameters: {count_parameters(model):,}")

    model, history = train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
    )
    plot_history(history, output_dir)
    plot_mistakes(model, test_loader, device, output_dir)
    plot_confusion_matrix(model, test_loader, device, output_dir)

    if max(history["test_acc"]) >= 99.0:
        print("Target reached: test accuracy > 99%.")
    else:
        print("Target not reached yet. Try more epochs, CUDA/GPU, or lower dropout.")

    if args.deep_experiment:
        run_deep_experiment(train_loader, test_loader, device, epochs=args.deep_epochs)


if __name__ == "__main__":
    main()
