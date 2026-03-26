"""Training CLI entrypoint for WeatherMamba Pro."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch

from weathermamba.data import PointCloudAugmentation, WeatherPointCloudDataset, build_dataloader
from weathermamba.engine import Trainer
from weathermamba.models import create_weather_mamba_model
from weathermamba.utils import choose_device, load_yaml, save_yaml, set_seed, setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


class ConfigError(ValueError):
    """Raised when user-provided config values are invalid."""


def parse_stage_depths(raw: str):
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) < 2:
        raise ConfigError("stage_depths must contain at least 2 integers, e.g. 2,2,2")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train WeatherMamba Pro")

    parser.add_argument("--model-config", type=str, default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--data-config", type=str, default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--train-config", type=str, default=str(DEFAULT_TRAIN_CONFIG))

    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-points", type=int, default=None)

    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--stage-depths", type=str, default=None)

    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    return build_parser().parse_args(argv)


def apply_overrides(args, model_cfg, data_cfg, train_cfg):
    if args.dataset_path is not None:
        data_cfg["dataset_path"] = args.dataset_path
    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        data_cfg.setdefault("loading", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg.setdefault("loading", {})["num_workers"] = args.num_workers
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.num_points is not None:
        data_cfg["num_points"] = args.num_points

    if args.hidden_dim is not None:
        model_cfg["hidden_dim"] = args.hidden_dim
    if args.num_classes is not None:
        model_cfg["num_classes"] = args.num_classes
    if args.stage_depths is not None:
        model_cfg["stage_depths"] = parse_stage_depths(args.stage_depths)

    if args.device is not None:
        train_cfg["device"] = args.device
    if args.seed is not None:
        train_cfg["seed"] = args.seed


def validate_required_paths(data_cfg):
    dataset_path = str(data_cfg.get("dataset_path", "")).strip()
    if not dataset_path:
        raise ConfigError(
            "dataset_path is not set. Please edit configs/data.yaml or pass --dataset-path."
        )


def build_datasets_and_loaders(data_cfg):
    loading = data_cfg.get("loading", {})
    batch_size = int(loading.get("batch_size", 2))
    num_workers = int(loading.get("num_workers", 4))
    pin_memory = bool(loading.get("pin_memory", True))

    train_aug = None
    if data_cfg.get("augmentation", {}).get("enabled", True):
        train_aug = PointCloudAugmentation({"augmentation": data_cfg.get("augmentation", {})})

    train_dataset = WeatherPointCloudDataset(
        root_dir=data_cfg["dataset_path"],
        split=data_cfg.get("train_split", "train"),
        num_points=int(data_cfg.get("num_points", 32768)),
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        file_suffixes=data_cfg.get("file_suffixes", [".bin", ".txt"]),
        augmentation=train_aug,
        unknown_weather_index=int(data_cfg.get("unknown_weather_index", 3)),
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )

    val_loader = None
    val_split = data_cfg.get("val_split", "val")
    try:
        val_dataset = WeatherPointCloudDataset(
            root_dir=data_cfg["dataset_path"],
            split=val_split,
            num_points=int(data_cfg.get("num_points", 32768)),
            ignore_index=int(data_cfg.get("ignore_index", 255)),
            file_suffixes=data_cfg.get("file_suffixes", [".bin", ".txt"]),
            augmentation=None,
            unknown_weather_index=int(data_cfg.get("unknown_weather_index", 3)),
        )
        val_loader = build_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
        )
    except Exception:
        val_loader = None

    return train_loader, val_loader


def run(args) -> int:
    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.train_config)
    apply_overrides(args, model_cfg, data_cfg, train_cfg)
    validate_required_paths(data_cfg)

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = choose_device(train_cfg.get("device", "cuda"))
    logger = setup_logger(level=train_cfg.get("log_level", "INFO"))

    run_name = args.experiment_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(train_cfg.get("output_dir", "outputs/weathermamba_pro"))
    run_dir = out_root / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(model_cfg, run_dir / "model_resolved.yaml")
    save_yaml(data_cfg, run_dir / "data_resolved.yaml")
    save_yaml(train_cfg, run_dir / "train_resolved.yaml")

    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {run_dir}")

    train_loader, val_loader = build_datasets_and_loaders(data_cfg)

    model = create_weather_mamba_model(**model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        amp=bool(train_cfg.get("amp", False)),
        grad_clip=train_cfg.get("grad_clip", None),
        log_interval=int(train_cfg.get("log_interval", 20)),
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        points = batch["points"].to(device)
        weather_type = batch["weather_type"].to(device)
        with torch.no_grad():
            logits = model(points, weather_type)
        logger.info(
            "Dry-run successful | "
            f"points={tuple(points.shape)} logits={tuple(logits.shape)} weather={tuple(weather_type.shape)}"
        )
        return 0

    epochs = int(train_cfg.get("epochs", 50))
    save_interval = int(train_cfg.get("save_interval", 1))
    val_interval = int(train_cfg.get("val_interval", 1))

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader, epoch)
        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f}"
        )

        val_metrics = None
        if val_loader is not None and epoch % val_interval == 0:
            val_metrics = trainer.evaluate(val_loader)
            logger.info(
                f"Epoch {epoch}/{epochs} | val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['acc']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                Trainer.save_checkpoint(
                    ckpt_dir / "best_model.pth",
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                    },
                )

        if epoch % save_interval == 0:
            Trainer.save_checkpoint(
                ckpt_dir / f"epoch_{epoch}.pth",
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
            )

    Trainer.save_checkpoint(
        ckpt_dir / "last_model.pth",
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )
    logger.info("Training finished.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except ConfigError as exc:
        print(f"[ConfigError] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
