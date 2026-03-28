"""Evaluation CLI entrypoint for WeatherMamba Pro."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from weathermamba.data import WeatherPointCloudDataset, build_dataloader
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


def parse_bool(raw):
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"Invalid boolean value: {raw!r}")


def resolve_dataset_root(base_path: str, dataset_name: Optional[str]) -> str:
    base = Path(base_path).expanduser()
    if dataset_name:
        candidate = base / dataset_name
        if candidate.exists():
            return str(candidate)
    return str(base)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WeatherMamba Pro")

    parser.add_argument("--model-config", type=str, default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--data-config", type=str, default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--train-config", type=str, default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Alias of --train-config for README compatibility.",
    )

    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--source-dataset", type=str, default=None)
    parser.add_argument("--target-dataset", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None, help="Weather subset name, e.g. dense_fog.")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--stage-depths", type=str, default=None)
    parser.add_argument("--use-manf", nargs="?", const=True, type=parse_bool, default=None)
    parser.add_argument("--use-radm", nargs="?", const=True, type=parse_bool, default=None)
    parser.add_argument("--use-wgrg", nargs="?", const=True, type=parse_bool, default=None)

    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    return build_parser().parse_args(argv)


def apply_overrides(args, model_cfg, data_cfg, train_cfg):
    if args.dataset_path is not None:
        data_cfg["dataset_path"] = args.dataset_path
    if args.source_dataset is not None:
        data_cfg["source_dataset"] = args.source_dataset
    if args.target_dataset is not None:
        data_cfg["target_dataset"] = args.target_dataset
    if args.subset is not None:
        data_cfg["eval_subset"] = args.subset

    if args.batch_size is not None:
        data_cfg.setdefault("loading", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg.setdefault("loading", {})["num_workers"] = args.num_workers
    if args.num_points is not None:
        data_cfg["num_points"] = args.num_points

    if args.hidden_dim is not None:
        model_cfg["hidden_dim"] = args.hidden_dim
    if args.num_classes is not None:
        model_cfg["num_classes"] = args.num_classes
    if args.stage_depths is not None:
        model_cfg["stage_depths"] = parse_stage_depths(args.stage_depths)
    if args.use_manf is not None:
        model_cfg["use_manf"] = bool(args.use_manf)
    if args.use_radm is not None:
        model_cfg["use_radm"] = bool(args.use_radm)
    if args.use_wgrg is not None:
        model_cfg["use_wgrg"] = bool(args.use_wgrg)

    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.seed is not None:
        train_cfg["seed"] = args.seed


def validate_required_paths(args, data_cfg):
    dataset_path = str(data_cfg.get("dataset_path", "")).strip()
    if not dataset_path:
        raise ConfigError("dataset_path is not set. Please edit configs/data.yaml or pass --dataset-path.")
    if not Path(dataset_path).expanduser().exists():
        raise ConfigError(f"dataset_path does not exist: {dataset_path}")

    if not args.dry_run:
        ckpt = str(args.checkpoint or "").strip()
        if not ckpt:
            raise ConfigError("--checkpoint is required for evaluation (or pass --dry-run).")
        if not Path(ckpt).expanduser().exists():
            raise ConfigError(f"checkpoint does not exist: {ckpt}")


def build_eval_loader(data_cfg):
    loading = data_cfg.get("loading", {})
    batch_size = int(loading.get("batch_size", 2))
    num_workers = int(loading.get("num_workers", 4))
    pin_memory = bool(loading.get("pin_memory", True))

    source_dataset = data_cfg.get("source_dataset")
    target_dataset = data_cfg.get("target_dataset")
    dataset_root = str(data_cfg["dataset_path"])
    eval_root = resolve_dataset_root(dataset_root, target_dataset or source_dataset)

    eval_split = data_cfg.get("eval_subset") or data_cfg.get("test_split") or data_cfg.get("val_split", "val")

    dataset = WeatherPointCloudDataset(
        root_dir=eval_root,
        split=eval_split,
        num_points=int(data_cfg.get("num_points", 32768)),
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        file_suffixes=data_cfg.get("file_suffixes", [".bin", ".txt"]),
        augmentation=None,
        unknown_weather_index=int(data_cfg.get("unknown_weather_index", 3)),
    )

    loader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )
    return loader, eval_root, eval_split


def _extract_state_dict(checkpoint: Dict):
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ConfigError("Unsupported checkpoint format. Expected model_state_dict/state_dict mapping.")


def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> float:
    preds = logits.argmax(dim=-1)
    valid = labels != ignore_index
    valid_count = valid.sum().item()
    if valid_count == 0:
        return 0.0
    correct = (preds[valid] == labels[valid]).sum().item()
    return correct / valid_count


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    ignore_index: int,
    predictions_dir: Optional[Path] = None,
):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    num_steps = 0
    num_saved = 0

    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        weather_type = batch["weather_type"].to(device, non_blocking=True)

        logits = model(points, weather_type)
        bsz, npts, ncls = logits.shape
        loss = criterion(logits.reshape(bsz * npts, ncls), labels.reshape(bsz * npts))
        acc = _compute_accuracy(logits, labels, ignore_index)

        total_loss += loss.item()
        total_acc += acc
        num_steps += 1

        if predictions_dir is not None:
            preds = logits.argmax(dim=-1).cpu()
            file_paths = batch["file_path"]
            for idx, file_path in enumerate(file_paths):
                pred_path = predictions_dir / f"{num_saved:06d}_{Path(file_path).stem}.pt"
                torch.save({"file_path": file_path, "prediction": preds[idx]}, pred_path)
                num_saved += 1

    metrics = {
        "loss": total_loss / max(1, num_steps),
        "acc": total_acc / max(1, num_steps),
        "num_batches": int(num_steps),
        "num_predictions_saved": int(num_saved),
    }
    return metrics


def run(args) -> int:
    train_cfg_path = args.config or args.train_config

    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(train_cfg_path)
    apply_overrides(args, model_cfg, data_cfg, train_cfg)
    validate_required_paths(args, data_cfg)

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(train_cfg.get("device", "cuda"))
    logger = setup_logger(level=train_cfg.get("log_level", "INFO"))

    run_name = args.experiment_name or datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    out_root = Path(train_cfg.get("output_dir", "outputs/weathermamba_pro"))
    run_dir = out_root / run_name
    preds_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(model_cfg, run_dir / "model_resolved.yaml")
    save_yaml(data_cfg, run_dir / "data_resolved.yaml")
    save_yaml(train_cfg, run_dir / "train_resolved.yaml")

    eval_loader, eval_root, eval_split = build_eval_loader(data_cfg)

    model = create_weather_mamba_model(**model_cfg).to(device)

    if args.dry_run:
        batch = next(iter(eval_loader))
        points = batch["points"].to(device)
        weather_type = batch["weather_type"].to(device)
        logits = model(points, weather_type)
        logger.info(
            "Dry-run successful | "
            f"points={tuple(points.shape)} logits={tuple(logits.shape)} weather={tuple(weather_type.shape)}"
        )
        return 0

    checkpoint = torch.load(str(Path(args.checkpoint).expanduser()), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ConfigError("Unsupported checkpoint format. Expected a dict-like .pth file.")
    state_dict = _extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    predictions_dir = preds_dir if args.save_predictions else None
    metrics = evaluate(
        model=model,
        loader=eval_loader,
        device=device,
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        predictions_dir=predictions_dir,
    )
    metrics["eval_root"] = str(eval_root)
    metrics["eval_split"] = str(eval_split)

    save_yaml(metrics, run_dir / "metrics.yaml")

    logger.info(f"Device: {device}")
    logger.info(f"Eval root: {eval_root} (split={eval_split})")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    if incompatible.missing_keys:
        logger.info(f"Missing keys while loading checkpoint: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        logger.info(f"Unexpected keys while loading checkpoint: {len(incompatible.unexpected_keys)}")
    logger.info(
        f"Evaluation finished | loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
        f"batches={metrics['num_batches']}"
    )
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
