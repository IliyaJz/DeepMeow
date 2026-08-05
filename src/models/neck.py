"""
neck.py — Feature Pyramid Network (FPN)
=========================================
The FPN is the second component of our detection pipeline,
sitting between the backbone and the detection head.

Problem it solves:
  Our backbone produces three feature maps at different scales:
    P3: [B, 256,  52, 52]  — fine-grained spatial detail, weak semantics
    P4: [B, 512,  26, 26]  — medium-level features
    P5: [B, 1024, 13, 13]  — strong semantics (knows WHAT the object is),
                              but low spatial resolution (poor at WHERE exactly)

  Small cats are best detected in shallow maps (P3, lots of spatial detail).
  Large cats are best detected in deep maps (P5, strong semantic meaning).

  But P3 is "dumb" (it hasn't seen the whole image, lacks context) and
  P5 is "blind" (it lost too much spatial detail in downsampling).

FPN solution:
  Build a top-down pathway that "flows" semantic information from P5 back up
  to P3 via upsampling + lateral connections.

  Architecture:
    P5                      → C5 (1x1 conv to reduce channels) = 256ch
    C5 (upsample 2x) + P4  → C4 (3x3 conv to refine)          = 256ch
    C4 (upsample 2x) + P3  → C3 (3x3 conv to refine)          = 256ch

  Now every scale has 256 channels AND contains both spatial detail AND
  semantic context, making detection more accurate at all sizes.

Output:
    F3: [B, 256, 52, 52]  — enriched with P5 semantics
    F4: [B, 256, 26, 26]
    F5: [B, 256, 13, 13]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import ConvBlock


# ─── FPN ──────────────────────────────────────────────────────────
class FPN(nn.Module):
    """
    Feature Pyramid Network — merges multi-scale backbone features
    into a unified set of feature maps, each with 256 channels.

    Top-down pathway:
      P5 → lateral_c5 (1x1) → C5
      C5 upsample + P4 → lateral_c4 (1x1) → C4 → output_c4 (3x3)
      C4 upsample + P3 → lateral_c3 (1x1) → C3 → output_c3 (3x3)
    """

    def __init__(self, out_channels: int = 256):
        """
        Args:
            out_channels (int): Number of output channels for each FPN level.
                                We use 256 throughout — standard in literature.
        """
        super().__init__()
        self.out_channels = out_channels

        # ── Lateral connections (1x1 convolutions) ─────────────────
        # These reduce the backbone channel counts to a uniform 256.
        # 1x1 conv does NOT change spatial size — only adjusts channels.
        self.lateral_c5 = nn.Conv2d(1024, out_channels, kernel_size=1)  # P5: 1024 -> 256
        self.lateral_c4 = nn.Conv2d(512,  out_channels, kernel_size=1)  # P4: 512  -> 256
        self.lateral_c3 = nn.Conv2d(256,  out_channels, kernel_size=1)  # P3: 256  -> 256

        # ── Output convolutions (3x3 convolutions) ─────────────────
        # After adding top-down + lateral features, we apply a 3x3 conv
        # to smooth out any aliasing artifacts from upsampling.
        self.output_c5 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_c4 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_c3 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)

        # ── Weight initialization ───────────────────────────────────
        self._init_weights()

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        """
        Args:
            p3 (Tensor): Backbone P3 feature map [B, 256,  52, 52]
            p4 (Tensor): Backbone P4 feature map [B, 512,  26, 26]
            p5 (Tensor): Backbone P5 feature map [B, 1024, 13, 13]

        Returns:
            Tuple of 3 tensors (F3, F4, F5), each [B, 256, H, W]
        """
        # ── Step 1: Reduce channels via lateral 1x1 convolutions ───
        c5 = self.lateral_c5(p5)  # [B, 256, 13, 13]
        c4 = self.lateral_c4(p4)  # [B, 256, 26, 26]
        c3 = self.lateral_c3(p3)  # [B, 256, 52, 52]

        # ── Step 2: Top-down pathway ────────────────────────────────
        # Upsample C5 (13x13) by factor 2 and add to C4 (26x26)
        # nearest neighbor mode is fast and avoids checkerboard artifacts
        top_down_c4 = c4 + F.interpolate(c5, scale_factor=2, mode="nearest")
        # Upsample the merged C4 (26x26) by factor 2 and add to C3 (52x52)
        top_down_c3 = c3 + F.interpolate(top_down_c4, scale_factor=2, mode="nearest")

        # ── Step 3: Smooth outputs with 3x3 convolutions ───────────
        f5 = self.output_c5(c5)            # [B, 256, 13, 13]
        f4 = self.output_c4(top_down_c4)   # [B, 256, 26, 26]
        f3 = self.output_c3(top_down_c3)   # [B, 256, 52, 52]

        return f3, f4, f5

    def _init_weights(self):
        """Initialize lateral and output conv weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.models.backbone import Backbone

    backbone = Backbone()
    fpn      = FPN(out_channels=256)

    dummy = torch.randn(2, 3, 416, 416)

    with torch.no_grad():
        p3, p4, p5 = backbone(dummy)
        f3, f4, f5 = fpn(p3, p4, p5)

    print(f"F3 shape: {f3.shape}")   # Expected: [2, 256, 52, 52]
    print(f"F4 shape: {f4.shape}")   # Expected: [2, 256, 26, 26]
    print(f"F5 shape: {f5.shape}")   # Expected: [2, 256, 13, 13]
    print("FPN sanity check passed!")
