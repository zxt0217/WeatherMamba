"""Summarize WeatherMamba mIoU from evaluation outputs.

The test dataloader samples or pads each point cloud to a fixed size, so sampled
predictions cannot be safely matched back to full-resolution labels unless the
sampled labels or sample indices were saved. This script therefore uses labels
stored inside prediction ``.pt`` files when available, or falls back to the
``metrics.yaml`` written by ``scripts/test.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import yaml


class EvaluationError(RuntimeError):
    """Raised when mIoU cannot be computed from the available artifacts."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate or summarize WeatherMamba mIoU")
    parser.add_argument("--prediction-dir", type=str, required=True)
    parser.add_argument("--ground-truth-root", type=str, default=None)
    parser.add_argument("--label-map", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None, help="Optional path to metrics.yaml.")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    return build_parser().parse_args(argv)


def _update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    preds_flat = preds.reshape(-1).cpu().long()
    labels_flat = labels.reshape(-1).cpu().long()

    valid = (
        (labels_flat != ignore_index)
        & (labels_flat >= 0)
        & (labels_flat < num_classes)
        & (preds_flat >= 0)
        & (preds_flat < num_classes)
    )
    if valid.sum().item() == 0:
        return

    encoded = labels_flat[valid] * num_classes + preds_flat[valid]
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion_matrix += counts.reshape(num_classes, num_classes)


def _summarize_confusion_matrix(confusion_matrix: torch.Tensor) -> Dict:
    cm = confusion_matrix.to(torch.float64)
    true_positive = torch.diag(cm)
    gt_count = cm.sum(dim=1)
    pred_count = cm.sum(dim=0)
    union = gt_count + pred_count - true_positive

    per_class_iou = []
    valid_ious = []
    for cls_idx in range(cm.shape[0]):
        if union[cls_idx].item() > 0:
            iou = true_positive[cls_idx] / union[cls_idx]
            per_class_iou.append(float(iou.item()))
            valid_ious.append(iou)
        else:
            per_class_iou.append(None)

    total_valid = cm.sum().item()
    total_correct = true_positive.sum().item()
    miou = float(torch.stack(valid_ious).mean().item()) if valid_ious else 0.0
    overall_accuracy = float(total_correct / total_valid) if total_valid > 0 else 0.0

    return {
        "per_class_iou": per_class_iou,
        "miou": miou,
        "overall_accuracy": overall_accuracy,
        "valid_points": int(total_valid),
        "confusion_matrix": confusion_matrix.tolist(),
    }


def _find_metrics_path(prediction_dir: Path, explicit_metrics: Optional[str]) -> Optional[Path]:
    candidates = []
    if explicit_metrics:
        candidates.append(Path(explicit_metrics).expanduser())
    candidates.append(prediction_dir.parent / "metrics.yaml")

    for path in candidates:
        if path.exists():
            return path
    return None


def _load_metrics(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        metrics = yaml.safe_load(f) or {}
    required = {"per_class_iou", "miou", "overall_accuracy"}
    missing = sorted(required - set(metrics))
    if missing:
        raise EvaluationError(f"Metrics file is missing required fields: {', '.join(missing)}")
    return metrics


def _compute_from_prediction_files(prediction_dir: Path, num_classes: int, ignore_index: int) -> Optional[Dict]:
    if not prediction_dir.exists():
        return None

    prediction_files = sorted(prediction_dir.glob("*.pt"))
    if not prediction_files:
        return None

    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    num_files_used = 0

    for path in prediction_files:
        data = torch.load(str(path), map_location="cpu")
        if "prediction" not in data:
            raise EvaluationError(f"Prediction file is missing 'prediction': {path}")
        if "labels" not in data:
            return None

        preds = torch.as_tensor(data["prediction"])
        labels = torch.as_tensor(data["labels"])
        if preds.numel() != labels.numel():
            raise EvaluationError(f"Prediction/label length mismatch in {path}")

        _update_confusion_matrix(confusion_matrix, preds, labels, num_classes, ignore_index)
        num_files_used += 1

    metrics = _summarize_confusion_matrix(confusion_matrix)
    metrics["num_prediction_files"] = int(num_files_used)
    return metrics


def _format_percent(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _print_metrics(metrics: Dict, source: str) -> None:
    print(f"Source: {source}")
    print(f"mIoU: {_format_percent(metrics.get('miou'))}")
    print(f"Overall accuracy: {_format_percent(metrics.get('overall_accuracy'))}")
    if "valid_points" in metrics:
        print(f"Valid points: {metrics['valid_points']}")

    per_class_iou = metrics.get("per_class_iou", [])
    if isinstance(per_class_iou, dict):
        items = per_class_iou.items()
    else:
        items = ((f"class_{idx:02d}", value) for idx, value in enumerate(per_class_iou))

    print("Per-class IoU:")
    for name, value in items:
        print(f"  {name}: {_format_percent(value)}")


def run(args) -> int:
    prediction_dir = Path(args.prediction_dir).expanduser()

    metrics = _compute_from_prediction_files(
        prediction_dir=prediction_dir,
        num_classes=int(args.num_classes),
        ignore_index=int(args.ignore_index),
    )
    source = "saved prediction labels"

    if metrics is None:
        metrics_path = _find_metrics_path(prediction_dir, args.metrics)
        if metrics_path is None:
            raise EvaluationError(
                "Could not compute mIoU from prediction files because sampled labels are not present, "
                "and no metrics.yaml was found. Re-run scripts/test.py so it writes metrics.yaml, "
                "or pass --save-predictions with the updated code to save sampled labels."
            )
        metrics = _load_metrics(metrics_path)
        source = str(metrics_path)

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    _print_metrics(metrics, source)
    if args.ground_truth_root or args.label_map:
        print("Note: ground-truth-root and label-map are accepted for README compatibility.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except EvaluationError as exc:
        print(f"[EvaluationError] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
