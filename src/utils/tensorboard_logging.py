import os
from numbers import Number

import numpy as np
import torch


class TensorBoardLogger:
    """Small wrapper around SummaryWriter with safe no-op behavior when disabled."""

    def __init__(self, enabled: bool, log_dir: str, run_index: int, flush_secs: int = 30):
        self.enabled = bool(enabled)
        self.writer = None
        self.run_log_dir = None

        if not self.enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging is enabled, but 'tensorboard' is not installed. "
                "Install tensorboard==2.15.2 to continue."
            ) from exc

        self.run_log_dir = os.path.join(log_dir, f"run_{run_index}")
        os.makedirs(self.run_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_log_dir, flush_secs=flush_secs)

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            detached = value.detach()
            if detached.numel() == 1:
                return float(detached.item())
            return float(detached.to(torch.float32).mean().item())
        if isinstance(value, np.generic):
            return float(value.item())
        if isinstance(value, Number):
            return float(value)
        raise TypeError(f"Unsupported scalar value type for TensorBoard logging: {type(value)}")

    def log_scalar(self, name: str, value, step: int):
        if self.writer is None:
            return
        self.writer.add_scalar(name, self._to_float(value), int(step))

    def log_text(self, name: str, text: str, step: int = 0):
        if self.writer is None:
            return
        self.writer.add_text(name, str(text), global_step=int(step))

    def close(self):
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()

