"""Create quick top-down PNG visualizations from saved WeatherMamba predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


class VisualizationError(RuntimeError):
    """Raised when a prediction file cannot be visualized safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize saved WeatherMamba predictions")
    parser.add_argument("--prediction-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--point-size", type=float, default=0.35)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--compare-labels", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    return build_parser().parse_args(argv)


def _load_raw_points(path: Path) -> np.ndarray:
    if path.suffix == ".bin":
        return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

    if path.suffix == ".txt":
        data = np.loadtxt(path, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] >= 4:
            return data[:, :4]
        return np.pad(data, ((0, 0), (0, 4 - data.shape[1])))

    raise VisualizationError(f"Unsupported point cloud suffix: {path.suffix}")


def _extract_points_and_predictions(data: dict, prediction_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    if "prediction" not in data:
        raise VisualizationError(f"Missing 'prediction' in {prediction_path}")

    preds = torch.as_tensor(data["prediction"]).cpu().numpy().reshape(-1)
    labels = None
    if "labels" in data:
        labels = torch.as_tensor(data["labels"]).cpu().numpy().reshape(-1)

    if "points" in data:
        points = torch.as_tensor(data["points"]).cpu().numpy()
    else:
        file_path = data.get("file_path")
        if not file_path:
            raise VisualizationError(f"Missing sampled points and file_path in {prediction_path}")
        points = _load_raw_points(Path(file_path))
        if points.shape[0] != preds.shape[0]:
            raise VisualizationError(
                f"{prediction_path} does not contain sampled points, and raw point count "
                f"({points.shape[0]}) differs from prediction count ({preds.shape[0]})."
            )

    if points.shape[0] != preds.shape[0]:
        raise VisualizationError(f"Point/prediction length mismatch in {prediction_path}")
    if labels is not None and labels.shape[0] != preds.shape[0]:
        raise VisualizationError(f"Label/prediction length mismatch in {prediction_path}")

    title = Path(str(data.get("file_path", prediction_path.name))).name
    return points, preds, labels, title


def _plot_values(
    ax,
    points: np.ndarray,
    values: np.ndarray,
    title: str,
    num_classes: int,
    ignore_index: int,
    point_size: float,
) -> None:
    valid = (values != ignore_index) & (values >= 0)
    invalid = ~valid

    if invalid.any():
        ax.scatter(
            points[invalid, 0],
            points[invalid, 1],
            c="#d0d0d0",
            s=point_size,
            linewidths=0,
        )

    if valid.any():
        ax.scatter(
            points[valid, 0],
            points[valid, 1],
            c=values[valid],
            cmap="tab20",
            vmin=0,
            vmax=max(1, num_classes - 1),
            s=point_size,
            linewidths=0,
        )

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _save_visualization(
    prediction_path: Path,
    output_path: Path,
    num_classes: int,
    ignore_index: int,
    point_size: float,
    dpi: int,
    compare_labels: bool,
) -> None:
    data = torch.load(str(prediction_path), map_location="cpu")
    points, preds, labels, title = _extract_points_and_predictions(data, prediction_path)

    if compare_labels and labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        _plot_values(axes[0], points, preds, f"{title} prediction", num_classes, ignore_index, point_size)
        _plot_values(axes[1], points, labels, f"{title} label", num_classes, ignore_index, point_size)
    else:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        _plot_values(ax, points, preds, f"{title} prediction", num_classes, ignore_index, point_size)

    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run(args) -> int:
    prediction_dir = Path(args.prediction_dir).expanduser()
    if not prediction_dir.exists():
        raise VisualizationError(f"prediction-dir does not exist: {prediction_dir}")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else prediction_dir.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = sorted(prediction_dir.glob("*.pt"))[: max(0, int(args.max_files))]
    if not prediction_files:
        raise VisualizationError(f"No .pt prediction files found in {prediction_dir}")

    for prediction_path in prediction_files:
        output_path = output_dir / f"{prediction_path.stem}.png"
        _save_visualization(
            prediction_path=prediction_path,
            output_path=output_path,
            num_classes=int(args.num_classes),
            ignore_index=int(args.ignore_index),
            point_size=float(args.point_size),
            dpi=int(args.dpi),
            compare_labels=bool(args.compare_labels),
        )
        print(f"Wrote {output_path}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except VisualizationError as exc:
        print(f"[VisualizationError] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
