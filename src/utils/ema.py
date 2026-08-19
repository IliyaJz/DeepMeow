"""
ema.py — Exponential Moving Average of Model Weights
=====================================================
During training, model weights bounce around noisily due to mini-batch
gradients.  EMA maintains a smoothed copy of the weights:

    EMA_weights = decay * EMA_weights + (1 - decay) * current_weights

Why does this help?
  - EMA weights are smoother and more stable than the live training weights
  - At validation time, using EMA weights typically gives +0.5-1% mAP
  - It acts like an ensemble of recent checkpoints for free

How to use:
    ema = ModelEMA(model, decay=0.9999)
    for batch in loader:
        optimizer.step()      # update model weights normally
        ema.update(model)     # update EMA shadow copy
    metrics = validate(ema.ema_model, ...)   # evaluate on EMA copy
"""

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average wrapper around a PyTorch model.

    Keeps a shadow (EMA) copy of the model's weights.  After each
    optimizer step, call ema.update(model) to blend the new weights
    into the shadow copy.

    Args:
        model  (nn.Module): The live training model to track.
        decay  (float):     EMA smoothing factor. Typical: 0.999-0.9999.
                            Higher = slower to react to new weights.
        warmup_steps (int): Number of steps over which decay is ramped up
                            from 0 to decay (prevents EMA from being
                            dominated by random initial weights).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 warmup_steps: int = 2000):
        # Deep copy: EMA model has same architecture but separate weights
        self.ema_model    = copy.deepcopy(model).eval()
        self.decay        = decay
        self.warmup_steps = warmup_steps
        self.step         = 0

        # EMA model never needs gradients
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def _current_decay(self) -> float:
        """
        Ramp-up decay: during early training we use a lower decay so
        the EMA adapts quickly.  After warmup_steps we use full decay.

        Formula (same as YOLOv5):
            d = min(decay, (1 + step) / (warmup_steps + step))
        """
        return min(self.decay,
                   (1 + self.step) / (self.warmup_steps + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Blend current model weights into the EMA shadow copy.
        Called once per optimizer step (once per batch).
        """
        self.step += 1
        d = self._current_decay()

        # Update parameters
        ema_params  = dict(self.ema_model.named_parameters())
        live_params = dict(model.named_parameters())
        for name, ema_p in ema_params.items():
            if name in live_params:
                live_p = live_params[name].detach()
                ema_p.copy_(d * ema_p + (1.0 - d) * live_p)

        # Update buffers (BatchNorm running_mean, running_var, etc.)
        ema_bufs  = dict(self.ema_model.named_buffers())
        live_bufs = dict(model.named_buffers())
        for name, ema_b in ema_bufs.items():
            if name in live_bufs:
                ema_b.copy_(live_bufs[name].detach())

    def state_dict(self) -> dict:
        """Return EMA model state for checkpointing."""
        return {
            "ema_model_state": self.ema_model.state_dict(),
            "step":            self.step,
            "decay":           self.decay,
        }

    def load_state_dict(self, state: dict):
        """Restore EMA model from a checkpoint."""
        self.ema_model.load_state_dict(state["ema_model_state"])
        self.step  = state.get("step",  0)
        self.decay = state.get("decay", self.decay)

    def to(self, device):
        """Move EMA model to the specified device."""
        self.ema_model = self.ema_model.to(device)
        return self
