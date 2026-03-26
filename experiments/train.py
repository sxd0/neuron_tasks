from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEAD_TOKENS = ("classifier", "fc", "head", "last_linear")


@dataclass
class DataConfig:
    data_dir: str = "data/raw/my_dataset"
    image_size: int = 224
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    num_workers: int = 2


@dataclass
class ModelConfig:
    model_names: list[str] = field(default_factory=lambda: ["resnet18", "efficientnet_b0"])
    pretrained: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    epochs: int = 8
    freeze_epochs: int = 3
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    device: str = "auto"


@dataclass
class SweepConfig:
    run_sweep: bool = False
    batch_sizes: list[int] = field(default_factory=lambda: [8, 16])
    lr_heads: list[float] = field(default_factory=lambda: [1e-3, 3e-4])


@dataclass
class ExportConfig:
    export_onnx: bool = False
    onnx_opset: int = 17
    onnx_file_name: str = "best_model.onnx"


@dataclass
class PathConfig:
    output_dir: str = "models"
    report_dir: str = "reports/figures"


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def parse_list_of_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_list_of_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_list_of_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune two timm models and optionally export the best one to ONNX.")
    parser.add_argument("--data-dir", type=str, default="data/raw/my_dataset")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--report-dir", type=str, default="reports/figures")
    parser.add_argument("--model-names", type=str, default="resnet18,efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--sweep-batch-sizes", type=str, default="8,16")
    parser.add_argument("--sweep-lr-heads", type=str, default="0.001,0.0003")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument("--onnx-file-name", type=str, default="best_model.onnx")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.data.data_dir = args.data_dir
    cfg.data.image_size = args.image_size
    cfg.data.num_workers = args.num_workers
    cfg.model.model_names = parse_list_of_strings(args.model_names)
    cfg.model.pretrained = not args.no_pretrained
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.freeze_epochs = args.freeze_epochs
    cfg.train.lr_head = args.lr_head
    cfg.train.lr_backbone = args.lr_backbone
    cfg.train.weight_decay = args.weight_decay
    cfg.train.optimizer = args.optimizer
    cfg.train.scheduler = args.scheduler
    cfg.train.label_smoothing = args.label_smoothing
    cfg.train.seed = args.seed
    cfg.train.device = "cpu" if args.cpu_only else "auto"
    cfg.sweep.run_sweep = args.run_sweep
    cfg.sweep.batch_sizes = parse_list_of_ints(args.sweep_batch_sizes)
    cfg.sweep.lr_heads = parse_list_of_floats(args.sweep_lr_heads)
    cfg.export.export_onnx = args.export_onnx
    cfg.export.onnx_opset = args.onnx_opset
    cfg.export.onnx_file_name = args.onnx_file_name
    cfg.paths.output_dir = args.output_dir
    cfg.paths.report_dir = args.report_dir
    return cfg


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int, mean: list[float], std: list[float]):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def locate_val_dir(root: Path) -> Path:
    candidates = [root / "validation", root / "val"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Не найдена папка validation или val внутри data/raw/my_dataset.")


def build_dataloaders(data_cfg: DataConfig, train_cfg: TrainConfig):
    root = resolve_path(data_cfg.data_dir)
    train_transform, eval_transform = build_transforms(data_cfg.image_size, data_cfg.mean, data_cfg.std)

    train_ds = datasets.ImageFolder(root / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(locate_val_dir(root), transform=eval_transform)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_transform)

    generator = torch.Generator().manual_seed(train_cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Классы в train/val/test должны совпадать и идти в одинаковом порядке.")

    return train_loader, val_loader, test_loader, train_ds.classes


def set_classifier(model: nn.Module, num_classes: int) -> None:
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
        return

    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Sequential):
            layers = list(classifier.children())
            if not layers:
                raise ValueError("Пустой classifier.")
            last = layers[-1]
            if hasattr(last, "in_features"):
                layers[-1] = nn.Linear(last.in_features, num_classes)
                model.classifier = nn.Sequential(*layers)
                return
        if hasattr(classifier, "in_features"):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return

    raise ValueError("Не удалось заменить классификационную голову модели.")


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(token in name for token in HEAD_TOKENS):
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def create_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)
    set_classifier(model, num_classes)
    return model


def build_optimizer(model: nn.Module, cfg: TrainConfig, backbone_mode: bool):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        lr = cfg.lr_backbone if backbone_mode else cfg.lr_head
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    lr = 1e-3 if not backbone_mode else 3e-4
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg: TrainConfig, total_epochs: int):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(total_epochs // 2, 1), gamma=0.1)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return loss, acc, y_true, y_pred


def plot_history(history: dict, save_path: Path, model_name: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / f"{model_name}_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / f"{model_name}_accuracy.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
    plt.tight_layout()
    plt.savefig(save_path / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_single_model(
    model_name: str,
    cfg: ProjectConfig,
    train_loader,
    val_loader,
    test_loader,
    class_names: list[str],
    device: torch.device,
) -> dict:
    num_classes = len(class_names)
    model = create_model(model_name, num_classes=num_classes, pretrained=cfg.model.pretrained).to(device)
    freeze_backbone(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    output_dir = resolve_path(cfg.paths.output_dir)
    report_dir = resolve_path(cfg.paths.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = output_dir / f"{model_name}_best.pth"

    start = time.time()
    total_epochs = cfg.train.epochs

    for epoch in range(1, total_epochs + 1):
        backbone_mode = epoch > cfg.train.freeze_epochs
        if epoch == cfg.train.freeze_epochs + 1:
            unfreeze_all(model)

        optimizer = build_optimizer(model, cfg.train, backbone_mode=backbone_mode)
        scheduler = build_scheduler(optimizer, cfg.train, total_epochs=max(total_epochs - epoch + 1, 1))

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_checkpoint)

        print(
            f"[{model_name}] epoch {epoch}/{total_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    plot_history(history, report_dir, model_name)
    plot_confusion_matrix(y_true, y_pred, class_names, report_dir, model_name)

    result = {
        "model_name": model_name,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_val_acc),
        "training_seconds": round(time.time() - start, 2),
        "checkpoint_path": str(best_checkpoint.relative_to(PROJECT_ROOT)),
        "class_names": class_names,
        "history": history,
    }
    return result


def export_to_onnx(model_name: str, cfg: ProjectConfig, class_names: list[str], device: torch.device) -> Path:
    output_dir = resolve_path(cfg.paths.output_dir)
    checkpoint_path = output_dir / f"{model_name}_best.pth"
    onnx_path = output_dir / cfg.export.onnx_file_name

    model = create_model(model_name, len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=cfg.export.onnx_opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    labels_path = output_dir / "labels.json"
    preprocess_path = output_dir / "preprocess.json"
    save_json(labels_path, {"classes": class_names})
    save_json(
        preprocess_path,
        {
            "image_size": cfg.data.image_size,
            "mean": cfg.data.mean,
            "std": cfg.data.std,
        },
    )
    return onnx_path


def run_training(cfg: ProjectConfig) -> None:
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(cfg.data, cfg.train)

    output_dir = resolve_path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.sweep.run_sweep:
        rows = []
        sweep_counter = 0
        for batch_size, lr_head, model_name in product(
            cfg.sweep.batch_sizes,
            cfg.sweep.lr_heads,
            cfg.model.model_names,
        ):
            sweep_counter += 1
            print(f"Запуск sweep #{sweep_counter}: model={model_name}, batch_size={batch_size}, lr_head={lr_head}")
            local_cfg = ProjectConfig(
                data=cfg.data,
                model=ModelConfig(model_names=[model_name], pretrained=cfg.model.pretrained),
                train=TrainConfig(
                    seed=cfg.train.seed,
                    batch_size=batch_size,
                    epochs=cfg.train.epochs,
                    freeze_epochs=cfg.train.freeze_epochs,
                    lr_head=lr_head,
                    lr_backbone=cfg.train.lr_backbone,
                    weight_decay=cfg.train.weight_decay,
                    optimizer=cfg.train.optimizer,
                    scheduler=cfg.train.scheduler,
                    label_smoothing=cfg.train.label_smoothing,
                    device=cfg.train.device,
                ),
                sweep=cfg.sweep,
                export=cfg.export,
                paths=cfg.paths,
            )
            train_loader, val_loader, test_loader, class_names = build_dataloaders(local_cfg.data, local_cfg.train)
            result = train_single_model(model_name, local_cfg, train_loader, val_loader, test_loader, class_names, device)
            result["batch_size"] = batch_size
            result["lr_head"] = lr_head
            rows.append(result)
        df = pd.DataFrame(rows).sort_values(by=["test_accuracy", "best_val_accuracy"], ascending=False)
        df.to_csv(output_dir / "sweep_results.csv", index=False)
        print(df[["model_name", "batch_size", "lr_head", "test_accuracy", "best_val_accuracy"]])
        return

    results = []
    for model_name in cfg.model.model_names:
        result = train_single_model(model_name, cfg, train_loader, val_loader, test_loader, class_names, device)
        results.append(result)

    df = pd.DataFrame(results).sort_values(by=["test_accuracy", "best_val_accuracy"], ascending=False)
    df.to_csv(output_dir / "results.csv", index=False)
    print(df[["model_name", "test_accuracy", "best_val_accuracy", "training_seconds"]])

    best_model_name = df.iloc[0]["model_name"]
    best_payload = {
        "best_model_name": best_model_name,
        "results": df.to_dict(orient="records"),
        "config": asdict(cfg),
    }
    save_json(output_dir / "best_run_summary.json", best_payload)

    if cfg.export.export_onnx:
        onnx_path = export_to_onnx(best_model_name, cfg, class_names, device)
        print(f"ONNX exported to: {onnx_path}")


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args)
    run_training(cfg)
