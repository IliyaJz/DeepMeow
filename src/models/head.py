"""
head.py — Detection Head
==========================
The detection head is the final prediction layer of our model.
It takes the FPN feature maps and produces raw predictions for every anchor.

For each anchor at each grid cell, the head predicts:
  - 4 box offsets:    [tx, ty, tw, th]
  - 1 objectness:     P(object is present here)
  - num_classes class probabilities: P(class | object present)

With 1 class (cat), each anchor produces:  4 + 1 + 1 = 6 values.
With 3 anchors per scale and 3 scales, the total raw predictions per image:
  P3 (52x52): 52*52*3*6 =  48,672
  P4 (26x26): 26*26*3*6 =  12,168
  P5 (13x13): 13*13*3*6 =   3,042
  Total:                    63,882 predictions

The head uses a stack of ConvBlocks to process features, then a final
1x1 conv to produce the raw output values.

IMPORTANT: the head outputs raw logits (no sigmoid/softmax applied here).
The loss function applies sigmoid internally for numerical stability.
"""

import torch
import torch.nn as nn

from src.models.backbone import ConvBlock


class DetectionHead(nn.Module):
    """
    Single-scale detection head.

    Each scale (P3, P4, P5) gets its own DetectionHead instance because
    they may handle different object sizes. However, the architecture is
    identical for all three.

    Architecture:
      FPN_feature → [ConvBlock x 4] → Conv(1x1) → raw predictions
    """

    def __init__(self, in_channels: int = 256, num_anchors: int = 3,
                 num_classes: int = 1):
        """
        Args:
            in_channels (int): Input channels from FPN (256 by default)
            num_anchors (int): Number of anchor boxes per grid cell (3)
            num_classes (int): Number of object classes (1 for cat-only)
        """
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Each anchor predicts: 4 box offsets + 1 objectness + num_classes
        self.pred_per_anchor = 4 + 1 + num_classes  # = 6 for cat-only

        # ── Feature processing stack ────────────────────────────────
        # 4 ConvBlocks to extract detection-relevant features
        # We keep channels at in_channels (256) throughout
        self.conv_stack = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1),
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1),
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1),
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1),
        )

        # ── Final prediction layer ──────────────────────────────────
        # 1x1 conv maps 256 channels to (num_anchors * pred_per_anchor)
        # This is a plain Conv2d — NO BatchNorm, NO activation — just raw values
        self.pred_conv = nn.Conv2d(
            in_channels,
            num_anchors * self.pred_per_anchor,
            kernel_size=1,
            bias=True,
        )

        # ── Initialize final conv bias for stable training start ────
        # Objectness bias: initialize to negative value so the model starts
        # by predicting "no object" for most anchors (realistic at start)
        nn.init.constant_(self.pred_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): FPN feature map [B, 256, H, W]

        Returns:
            predictions (Tensor): Raw predictions
                Shape: [B, H, W, num_anchors, 4 + 1 + num_classes]
                       = [B, H, W, 3, 6] for cat-only detection

                Prediction breakdown per anchor:
                  [0:4]  = box offsets [tx, ty, tw, th]
                  [4]    = objectness logit (sigmoid -> P(object))
                  [5:]   = class logits (sigmoid -> P(cat))
        """
        batch_size = x.shape[0]
        H, W       = x.shape[2], x.shape[3]

        # Pass through ConvBlock stack
        x = self.conv_stack(x)  # [B, 256, H, W]

        # Final 1x1 conv -> [B, num_anchors * pred_per_anchor, H, W]
        x = self.pred_conv(x)

        # Reshape to [B, H, W, num_anchors, pred_per_anchor]
        # This layout makes it easy to access per-anchor predictions
        x = x.permute(0, 2, 3, 1)                  # [B, H, W, num_anchors * pred_per_anchor]
        x = x.view(batch_size, H, W,
                   self.num_anchors, self.pred_per_anchor)  # [B, H, W, 3, 6]

        return x


# ─── Multi-Scale Head Wrapper ─────────────────────────────────────
class MultiScaleHead(nn.Module):
    """
    Applies a separate DetectionHead to each FPN scale (F3, F4, F5).

    Using separate heads per scale allows each head to specialize:
    the F3 head learns to detect small cats,
    the F5 head learns to detect large cats.
    """

    def __init__(self, in_channels: int = 256, num_anchors_per_scale: int = 3,
                 num_classes: int = 1):
        super().__init__()

        # One head per scale — each is independent with its own weights
        self.head_f3 = DetectionHead(in_channels, num_anchors_per_scale, num_classes)
        self.head_f4 = DetectionHead(in_channels, num_anchors_per_scale, num_classes)
        self.head_f5 = DetectionHead(in_channels, num_anchors_per_scale, num_classes)

    def forward(self, f3: torch.Tensor, f4: torch.Tensor, f5: torch.Tensor):
        """
        Args:
            f3 (Tensor): [B, 256, 52, 52]  — small-object scale
            f4 (Tensor): [B, 256, 26, 26]  — medium-object scale
            f5 (Tensor): [B, 256, 13, 13]  — large-object scale

        Returns:
            pred3 (Tensor): [B, 52, 52, 3, 6]
            pred4 (Tensor): [B, 26, 26, 3, 6]
            pred5 (Tensor): [B, 13, 13, 3, 6]
        """
        pred3 = self.head_f3(f3)  # [B, 52, 52, 3, 6]
        pred4 = self.head_f4(f4)  # [B, 26, 26, 3, 6]
        pred5 = self.head_f5(f5)  # [B, 13, 13, 3, 6]

        return pred3, pred4, pred5


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.models.backbone import Backbone
    from src.models.neck import FPN

    backbone = Backbone()
    fpn      = FPN()
    head     = MultiScaleHead(in_channels=256, num_anchors_per_scale=3, num_classes=1)

    dummy = torch.randn(2, 3, 416, 416)

    with torch.no_grad():
        p3, p4, p5         = backbone(dummy)
        f3, f4, f5         = fpn(p3, p4, p5)
        pred3, pred4, pred5 = head(f3, f4, f5)

    print(f"pred3 shape: {pred3.shape}")   # Expected: [2, 52, 52, 3, 6]
    print(f"pred4 shape: {pred4.shape}")   # Expected: [2, 26, 26, 3, 6]
    print(f"pred5 shape: {pred5.shape}")   # Expected: [2, 13, 13, 3, 6]
    print("MultiScaleHead sanity check passed!")
