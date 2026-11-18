"""U-Net written from scratch (no pretrained weights), used as the course baseline.

Every design choice can be switched off or replaced for the architecture ablation:
skip connections, learnable vs fixed upsampling, normalisation type, residual blocks, dropout.

    input 3×352×352
    stem   ConvBlock(3 → C)                    352
    down1  MaxPool + ConvBlock(C → 2C)         176
    down2  MaxPool + ConvBlock(2C → 4C)         88
    down3  MaxPool + ConvBlock(4C → 8C)         44
    down4  MaxPool + ConvBlock(8C → 16C)        22   (bottleneck, optional Dropout2d)
    up1..4 Upsample ×2, concat skip, ConvBlock  44 → 352
    head   1×1 conv → 1 logit per pixel
"""
import torch
import torch.nn as nn


def make_norm(kind, channels):
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {kind}")


class ConvBlock(nn.Module):
    """Two 3×3 conv → norm → ReLU (VGG-style). With residual=True, adds a shortcut like ResNet."""

    def __init__(self, cin, cout, norm="batch", residual=False):
        super().__init__()
        bias = norm == "none"  # a bias before a normalisation layer is redundant
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=bias),
            make_norm(norm, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=bias),
            make_norm(norm, cout),
        )
        self.shortcut = None
        if residual:
            self.shortcut = nn.Identity() if cin == cout else nn.Sequential(
                nn.Conv2d(cin, cout, 1, bias=bias), make_norm(norm, cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.body(x)
        if self.shortcut is not None:
            y = y + self.shortcut(x)
        return self.act(y)


class Up(nn.Module):
    """Double the resolution, halve the channels, optionally concatenate the encoder skip."""

    def __init__(self, cin, mode, skip, norm, residual):
        super().__init__()
        cout = cin // 2
        if mode == "transpose":
            self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        elif mode == "bilinear":
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                    nn.Conv2d(cin, cout, kernel_size=1))
        else:
            raise ValueError(f"Unknown upsample mode: {mode}")
        self.skip = skip
        self.conv = ConvBlock(cout * 2 if skip else cout, cout, norm, residual)

    def forward(self, x, skip_feature):
        x = self.up(x)
        if self.skip:
            x = torch.cat([x, skip_feature], dim=1)
        return self.conv(x)


class CustomUNet(nn.Module):
    def __init__(self, base_channels=32, skip=True, upsample="transpose", norm="batch",
                 block="plain", dropout=0.0):
        super().__init__()
        residual = block == "residual"
        c = [base_channels * 2 ** i for i in range(5)]
        self.stem = ConvBlock(3, c[0], norm, residual)
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), ConvBlock(c[i], c[i + 1], norm, residual)) for i in range(4))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.ups = nn.ModuleList(Up(c[i + 1], upsample, skip, norm, residual) for i in reversed(range(4)))
        self.head = nn.Conv2d(c[0], 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        # He initialisation, matched to ReLU (L4: var(w) = 2 / fan)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = [self.stem(x)]
        for down in self.downs:
            features.append(down(features[-1]))
        x = self.dropout(features.pop())
        for up in self.ups:
            x = up(x, features.pop())
        return self.head(x)
