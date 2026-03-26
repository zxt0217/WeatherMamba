"""Trainer for WeatherMamba Pro."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


class Trainer:
    """Simple trainer with checkpointing and validation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        ignore_index: int = 255,
        amp: bool = False,
        grad_clip: Optional[float] = None,
        log_interval: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ignore_index = int(ignore_index)
        self.amp = bool(amp and device.type == "cuda")
        self.grad_clip = grad_clip
        self.log_interval = max(1, int(log_interval))

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.scaler = GradScaler(enabled=self.amp)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bsz, npts, ncls = logits.shape
        return self.criterion(logits.reshape(bsz * npts, ncls), labels.reshape(bsz * npts))

    @staticmethod
    def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> float:
        preds = logits.argmax(dim=-1)
        valid = labels != ignore_index
        valid_count = valid.sum().item()
        if valid_count == 0:
            return 0.0
        correct = (preds[valid] == labels[valid]).sum().item()
        return correct / valid_count

    def train_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        num_steps = 0

        for step, batch in enumerate(loader, start=1):
            points = batch["points"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            weather_type = batch["weather_type"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp):
                logits = self.model(points, weather_type)
                loss = self._compute_loss(logits, labels)

            if self.amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            acc = self._compute_accuracy(logits.detach(), labels, self.ignore_index)
            total_loss += loss.item()
            total_acc += acc
            num_steps += 1

            if step % self.log_interval == 0:
                print(
                    f"[Train] epoch={epoch} step={step}/{len(loader)} "
                    f"loss={total_loss / num_steps:.4f} acc={total_acc / num_steps:.4f}"
                )

        return {
            "loss": total_loss / max(1, num_steps),
            "acc": total_acc / max(1, num_steps),
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        num_steps = 0

        for batch in loader:
            points = batch["points"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            weather_type = batch["weather_type"].to(self.device, non_blocking=True)

            logits = self.model(points, weather_type)
            loss = self._compute_loss(logits, labels)

            acc = self._compute_accuracy(logits, labels, self.ignore_index)
            total_loss += loss.item()
            total_acc += acc
            num_steps += 1

        return {
            "loss": total_loss / max(1, num_steps),
            "acc": total_acc / max(1, num_steps),
        }

    @staticmethod
    def save_checkpoint(path: Path, state: Dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path))
