import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, base_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self):
        step = self._get_current_step()
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step / max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _get_current_step(self):
        state = self.optimizer.state.get(self.optimizer.param_groups[0]["params"][0], {})
        return state.get("step", 0)

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict["warmup_steps"]
        self.max_steps = state_dict["max_steps"]
        self.base_lr = state_dict["base_lr"]
        self.min_lr = state_dict["min_lr"]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_epochs: int = 10,
        grad_accumulation_steps: int = 1,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.grad_accumulation_steps = grad_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval

        self.model.to(self.device)

        decay_params = []
        no_decay_params = []
        for _, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
        )

        steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.grad_accumulation_steps
        )
        self.total_steps = steps_per_epoch * self.max_epochs

        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            max_steps=self.total_steps,
            base_lr=learning_rate,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _prepare_batch(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            input_ids, labels = batch
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
            labels = batch.get("labels", batch.get("targets"))
        else:
            raise ValueError(
                f"Batch must be a tuple/list of (input_ids, labels) or a dict, "
                f"got {type(batch)}"
            )

        input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        return input_ids, labels

    def train(self):
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Total training steps: {self.total_steps}")
        print(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
        print(f"Effective batch size: {self._get_effective_batch_size()}")
        print("-" * 60)

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            train_metrics = self.train_epoch()

            epoch_duration = time.time() - epoch_start_time

            if self.val_dataloader is not None:
                val_metrics = self.validate()
                train_metrics["val_loss"] = val_metrics["val_loss"]
                train_metrics["val_perplexity"] = val_metrics["val_perplexity"]

                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]

            train_metrics["epoch_duration_sec"] = epoch_duration

            self.save_checkpoint(epoch, train_metrics)
            self._log_epoch_summary(epoch, train_metrics, epoch_duration)

        print("-" * 60)
        print("Training complete!")
        if self.val_dataloader is not None:
            print(f"Best validation loss: {self.best_val_loss:.4f}")

    def train_epoch(self) -> dict:
        self.model.train()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        step_start_time = time.time()

        for step, batch in enumerate(self.train_dataloader):
            input_ids, labels = self._prepare_batch(batch)

            _, loss = self.model(input_ids, labels)

            if self.grad_accumulation_steps > 1:
                loss = loss / self.grad_accumulation_steps

            loss.backward()

            batch_loss = loss.item()
            if self.grad_accumulation_steps > 1:
                batch_loss *= self.grad_accumulation_steps
            total_loss += batch_loss

            batch_tokens = input_ids.numel()
            total_tokens += batch_tokens
            num_batches += 1

            if (step + 1) % self.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            if (step + 1) % self.log_interval == 0:
                elapsed = time.time() - step_start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
                avg_loss = total_loss / num_batches
                current_lr = self._get_lr()
                perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

                print(
                    f"Epoch [{self.epoch + 1}/{self.max_epochs}] "
                    f"Step [{step + 1}/{len(self.train_dataloader)}] "
                    f"| Loss: {avg_loss:.4f} "
                    f"| PPL: {perplexity:.2f} "
                    f"| LR: {current_lr:.2e} "
                    f"| Tok/s: {tokens_per_sec:.0f}"
                )

                total_tokens = 0
                step_start_time = time.time()

        remaining_steps = len(self.train_dataloader) % self.grad_accumulation_steps
        if remaining_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        return {
            "train_loss": avg_loss,
            "train_perplexity": perplexity,
            "learning_rate": self._get_lr(),
            "global_step": self.global_step,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            input_ids, labels = self._prepare_batch(batch)

            _, loss = self.model(input_ids, labels)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }

    def save_checkpoint(self, epoch: int, metrics: dict):
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"epoch_{epoch}.pt"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"] + 1

        print(f"Checkpoint loaded from: {path}")
        print(f"Resuming from epoch {self.epoch}")

    def _log_epoch_summary(self, epoch: int, metrics: dict, duration: float):
        print("-" * 60)
        print(
            f"Epoch {epoch + 1}/{self.max_epochs} complete "
            f"({duration:.1f}s)"
        )
        print(f"  Train Loss: {metrics['train_loss']:.4f}")
        print(f"  Train PPL:  {metrics['train_perplexity']:.2f}")
        if "val_loss" in metrics:
            print(f"  Val Loss:   {metrics['val_loss']:.4f}")
            print(f"  Val PPL:    {metrics['val_perplexity']:.2f}")
        print(f"  LR:         {metrics['learning_rate']:.2e}")
        print(f"  Global Step: {metrics['global_step']}")
        print("-" * 60)

    def _get_effective_batch_size(self) -> int:
        batch_size = getattr(self.train_dataloader, "batch_size", 1)
        return batch_size * self.grad_accumulation_steps

