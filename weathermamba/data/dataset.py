"""Dataset and dataloader utilities for WeatherMamba Pro."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WeatherPointCloudDataset(Dataset):
    """Point cloud dataset with fixed-size point sampling for training."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_points: int = 32768,
        ignore_index: int = 255,
        file_suffixes: Optional[Sequence[str]] = None,
        augmentation=None,
        unknown_weather_index: int = 3,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = int(num_points)
        self.ignore_index = int(ignore_index)
        self.file_suffixes = tuple(file_suffixes or [".bin", ".txt"])
        self.augmentation = augmentation
        self.unknown_weather_index = int(unknown_weather_index)

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples = self._scan_samples(split_dir)
        if not self.samples:
            raise RuntimeError(f"No point cloud files found in {split_dir} for suffixes {self.file_suffixes}")

    def _scan_samples(self, split_dir: Path) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        for suffix in self.file_suffixes:
            pattern = str(split_dir / "**" / f"*{suffix}")
            for file_path in sorted(glob.glob(pattern, recursive=True)):
                samples.append(
                    {
                        "point_path": file_path,
                        "suffix": suffix,
                    }
                )
        return samples

    @staticmethod
    def _infer_weather_type(file_path: str, unknown_weather_index: int) -> int:
        lower = file_path.lower()
        if "rain" in lower:
            return 0
        if "snow" in lower:
            return 1
        if "fog" in lower:
            return 2
        return unknown_weather_index

    @staticmethod
    def _resolve_label_path(point_path: Path) -> Optional[Path]:
        if point_path.suffix != ".bin":
            txt_label = point_path.with_suffix(".label")
            return txt_label if txt_label.exists() else None

        candidates = [
            Path(str(point_path).replace("velodyne", "labels")).with_suffix(".label"),
            point_path.with_suffix(".label"),
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    @staticmethod
    def _safe_load_txt(path: Path) -> np.ndarray:
        data = np.loadtxt(path, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        return data

    @staticmethod
    def _load_kitti_label(path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None or not path.exists():
            return None
        labels = np.fromfile(str(path), dtype=np.uint32)
        return (labels & 0xFFFF).astype(np.int64)

    def _load_points_labels(self, point_path: Path, suffix: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if suffix == ".bin":
            points = np.fromfile(str(point_path), dtype=np.float32).reshape(-1, 4)
            labels = self._load_kitti_label(self._resolve_label_path(point_path))
            return points, labels

        if suffix == ".txt":
            data = self._safe_load_txt(point_path)
            if data.shape[1] >= 5:
                points = data[:, :4]
                labels = data[:, 4].astype(np.int64)
            else:
                points = data[:, :4] if data.shape[1] >= 4 else np.pad(data, ((0, 0), (0, 4 - data.shape[1])))
                labels = None
            return points.astype(np.float32, copy=False), labels

        raise ValueError(f"Unsupported file suffix: {suffix}")

    def _fixed_size_sample(self, points: np.ndarray, labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        num = points.shape[0]

        if num == 0:
            points = np.zeros((self.num_points, 4), dtype=np.float32)
            labels_out = np.full((self.num_points,), self.ignore_index, dtype=np.int64)
            return points, labels_out

        replace = num < self.num_points
        indices = np.random.choice(num, size=self.num_points, replace=replace)

        points = points[indices]
        if labels is None:
            labels_out = np.full((self.num_points,), self.ignore_index, dtype=np.int64)
        else:
            if labels.shape[0] != num:
                valid_len = min(labels.shape[0], num)
                points = points[:valid_len]
                labels = labels[:valid_len]
                replace = valid_len < self.num_points
                indices = np.random.choice(valid_len, size=self.num_points, replace=replace)
                points = points[indices]
            labels_out = labels[indices].astype(np.int64, copy=False)

        return points.astype(np.float32, copy=False), labels_out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        info = self.samples[index]
        point_path = Path(info["point_path"])

        points, labels = self._load_points_labels(point_path, info["suffix"])

        if self.augmentation is not None and self.split == "train":
            points, labels = self.augmentation(points, labels)

        points, labels = self._fixed_size_sample(points, labels)

        weather_type = self._infer_weather_type(str(point_path), self.unknown_weather_index)

        return {
            "points": torch.from_numpy(points).float(),
            "labels": torch.from_numpy(labels).long(),
            "weather_type": torch.tensor(weather_type, dtype=torch.long),
            "file_path": str(point_path),
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
) -> DataLoader:
    """Build a standard PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(num_workers)),
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
