"""
backbone.py — Custom CNN Feature Extractor
===========================================
The "backbone" is the first part of our detector.
It looks at the raw image and computes feature maps
(think of them as rich, compressed summaries of the image).

Architecture inspired by Darknet-53 (used in YOLOv3):
  - Stacked ConvBlocks (Conv + BatchNorm + LeakyReLU)
  - Residual blocks (skip connections) to allow training deeper networks
  - Multiple output scales for detecting cats of different sizes

Output:
  - P3: [B, 128, 52, 52]  — fine-grained features for small cats
  - P4: [B, 256, 26, 26]  — medium-scale features
  - P5: [B, 512, 13, 13]  — high-level semantic features for large cats

Why 3 scales?
  A 416×416 image is processed by increasingly deep layers.
  Shallow layers see fine details (edges, textures) but no "meaning".
  Deep layers see high-level patterns (cat face, body shape) but lose detail.
  We use all three levels to detect cats at any size.

Key math:
  Output size after convolution: floor((H + 2*padding - kernel) / stride) + 1
  After each downsampling step, spatial size halves: 416 → 208 → 104 → 52 → 26 → 13
"""

import torch
import torch.nn as nn


# ─── Building Block 1: ConvBlock ──────────────────────────────────
class ConvBlock(nn.Module):
    """
    The fundamental unit: Conv2d → BatchNorm → LeakyReLU.
    
    Why BatchNorm?
      Normalizes the output of each layer so values don't explode/vanish.
      Also acts as light regularization.
    
    Why LeakyReLU instead of ReLU?
      Regular ReLU outputs 0 for all negative inputs (neurons can "die").
      LeakyReLU outputs a tiny negative value instead (0.1 * x), keeping
      gradients alive during backpropagation.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        """
        Args:
            in_channels  (int): Number of input feature channels.
            out_channels (int): Number of output feature channels (filters).
            kernel_size  (int): Convolution filter size (3×3 by default).
            stride       (int): Step size; stride=2 halves spatial dimensions.
            padding      (int): Zero-padding around the input.
        """
        super().__init__()
        
        self.block = nn.Sequential(
            # ── Convolution ─────────────────────────────────────────
            # Learns spatial patterns (edges, shapes, textures)
            # bias=False because BatchNorm has its own bias term
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            
            # ── Batch Normalization ─────────────────────────────────
            # Normalizes each channel across the batch.
            # Keeps activations in a healthy range during training.
            nn.BatchNorm2d(out_channels),
            
            # ── Activation ─────────────────────────────────────────
            # f(x) = x if x > 0  else  0.1 * x
            # negative_slope=0.1 → the leaky part
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Building Block 2: ResidualBlock ──────────────────────────────
class ResidualBlock(nn.Module):
    """
    A residual block with a skip connection (from ResNet paper, He et al. 2015).
    
    Structure:
        input  ──────────────────────────► (+) ──► output
               └─► Conv(1×1) ─► Conv(3×3) ─┘
    
    Why residual connections?
      Without them, very deep networks suffer from vanishing gradients —
      the error signal gets weaker and weaker as it propagates back.
      
      With a skip connection, the gradient has a "highway" to flow
      directly to earlier layers without passing through all those
      multiplications.
      
      Instead of learning: output = F(input)
      The block learns:    output = input + F(input)
      This is called "learning the residual" F(input).
    
    The channel bottleneck (channels → channels//2 → channels) also
    reduces computation.
    """

    def __init__(self, channels: int):
        """
        Args:
            channels (int): Number of input AND output channels (they must match
                            so we can add the skip connection).
        """
        super().__init__()
        hidden = channels // 2  # Bottleneck: compress, then restore
        
        self.conv1 = ConvBlock(channels, hidden,   kernel_size=1, padding=0)  # 1×1 conv
        self.conv2 = ConvBlock(hidden,   channels, kernel_size=3, padding=1)  # 3×3 conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection: add input directly to the output
        # This is the "residual" — the block only needs to learn
        # what's DIFFERENT from the input.
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual  # Element-wise addition


# ─── Building Block 3: Stage ──────────────────────────────────────
class Stage(nn.Module):
    """
    One processing stage in the backbone:
      1. Downsample spatial size by ×2 (stride=2 convolution)
      2. Apply N residual blocks to extract deeper features

    After each stage: spatial size halves, channels double.
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        """
        Args:
            in_channels  (int): Input channels.
            out_channels (int): Output channels (doubled from input).
            num_blocks   (int): Number of residual blocks to stack.
        """
        super().__init__()
        
        # Stride=2 → spatial size halves (e.g. 208×208 → 104×104)
        self.downsample = ConvBlock(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1
        )
        
        # Stack multiple residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)   # Spatial size halves
        x = self.res_blocks(x)   # Feature depth increases
        return x


# ─── Main Backbone ────────────────────────────────────────────────
class Backbone(nn.Module):
    """
    Custom CNN backbone that processes a 416×416 image through 5 stages
    and returns feature maps at 3 different spatial resolutions.
    
    Architecture overview:
    
    Input:  [B, 3,   416, 416]
                ↓
    stem:   [B, 32,  416, 416]  (first ConvBlock, no downsampling)
                ↓
    stage1: [B, 64,  208, 208]  (1 residual block)
                ↓
    stage2: [B, 128, 104, 104]  (2 residual blocks)
                ↓
    stage3: [B, 256,  52,  52]  (8 residual blocks)  ← P3 output
                ↓
    stage4: [B, 512,  26,  26]  (8 residual blocks)  ← P4 output
                ↓
    stage5: [B, 1024, 13,  13]  (4 residual blocks)  ← P5 output
    
    P3 (52×52): Best for small cats — still has spatial detail
    P4 (26×26): Good for medium-sized cats
    P5 (13×13): Best for large/dominant cats — deep semantic meaning
    """

    def __init__(self):
        super().__init__()
        
        # ── Stem: initial feature extraction ───────────────────────
        # 3 channels (RGB) → 32 feature channels, no downsampling yet
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        # Output: [B, 32, 416, 416]
        
        # ── Stage 1 ────────────────────────────────────────────────
        # 32 → 64 channels, 416 → 208 spatial, 1 residual block
        self.stage1 = Stage(32, 64, num_blocks=1)
        # Output: [B, 64, 208, 208]
        
        # ── Stage 2 ────────────────────────────────────────────────
        # 64 → 128 channels, 208 → 104 spatial, 2 residual blocks
        self.stage2 = Stage(64, 128, num_blocks=2)
        # Output: [B, 128, 104, 104]
        
        # ── Stage 3 ────────────────────────────────────────────────
        # 128 → 256 channels, 104 → 52 spatial, 8 residual blocks
        self.stage3 = Stage(128, 256, num_blocks=8)
        # Output: [B, 256, 52, 52]  ← this is P3
        
        # ── Stage 4 ────────────────────────────────────────────────
        # 256 → 512 channels, 52 → 26 spatial, 8 residual blocks
        self.stage4 = Stage(256, 512, num_blocks=8)
        # Output: [B, 512, 26, 26]  ← this is P4
        
        # ── Stage 5 ────────────────────────────────────────────────
        # 512 → 1024 channels, 26 → 13 spatial, 4 residual blocks
        self.stage5 = Stage(512, 1024, num_blocks=4)
        # Output: [B, 1024, 13, 13]  ← this is P5
        
        # ── Weight initialization ───────────────────────────────────
        # Initialize weights properly for faster convergence
        self._init_weights()

    def forward(self, x: torch.Tensor):
        """
        Forward pass through all 5 stages.
        
        Args:
            x (Tensor): Input image batch, shape [B, 3, 416, 416]
        
        Returns:
            Tuple of 3 feature maps (P3, P4, P5):
              P3: [B, 256, 52, 52]
              P4: [B, 512, 26, 26]
              P5: [B, 1024, 13, 13]
        """
        x = self.stem(x)    # [B, 32,  416, 416]
        x = self.stage1(x)  # [B, 64,  208, 208]
        x = self.stage2(x)  # [B, 128, 104, 104]
        
        p3 = self.stage3(x)  # [B, 256, 52, 52]   ← save for FPN
        p4 = self.stage4(p3) # [B, 512, 26, 26]   ← save for FPN
        p5 = self.stage5(p4) # [B, 1024, 13, 13]  ← save for FPN
        
        # Return all three scales — the FPN (next week) will use them
        return p3, p4, p5

    def _init_weights(self):
        """
        Initialize convolution weights using Kaiming (He) initialization.
        
        Why? Random weights close to 0 lead to vanishing gradients.
        Kaiming init scales weights based on layer size to keep
        activation variance stable at the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm: start with scale=1, bias=0 (identity transform)
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias,   0.0)


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    # Run this file directly to verify the backbone works:
    #   python src/models/backbone.py
    
    model = Backbone()
    
    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone parameters: {total_params:,}")
    
    # Test with a fake batch of 2 images (416×416 RGB)
    dummy_input = torch.randn(2, 3, 416, 416)
    
    with torch.no_grad():  # No gradients needed for a forward-pass test
        p3, p4, p5 = model(dummy_input)
    
    print(f"P3 shape: {p3.shape}")   # Expected: [2, 256, 52, 52]
    print(f"P4 shape: {p4.shape}")   # Expected: [2, 512, 26, 26]
    print(f"P5 shape: {p5.shape}")   # Expected: [2, 1024, 13, 13]
    print("✓ Backbone forward pass OK!")
