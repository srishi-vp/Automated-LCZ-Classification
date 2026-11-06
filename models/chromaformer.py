
import torch
import torch.nn as nn
import math



class SpectralDependencyModule(nn.Module):
    """
    Spectral Dependency Module.
    Learns relationships between feature channels.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.shape

        features = x.reshape(batch, channels, height * width)

        attention = torch.bmm(
            features,
            features.transpose(1, 2)
        )

        attention = attention / math.sqrt(height * width)
        attention = torch.softmax(attention, dim=-1)

        output = torch.bmm(attention, features)
        output = output.reshape(batch, channels, height, width)

        return x + self.gamma * output


from torchvision.models.swin_transformer import SwinTransformerBlock


class SwinFeatureBlock(nn.Module):
    """
    Swin Transformer block for spatial feature extraction.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, channels, num_heads, window_size=4,
                 shift_size=0):
        super().__init__()

        self.swin = SwinTransformerBlock(
            dim=channels,
            num_heads=num_heads,
            window_size=[window_size, window_size],
            shift_size=[shift_size, shift_size],
        )

    def forward(self, x):
        # Convert NCHW to NHWC for torchvision Swin
        x = x.permute(0, 2, 3, 1)

        x = self.swin(x)

        # Convert NHWC back to NCHW
        x = x.permute(0, 3, 1, 2)

        return x

class PatchEmbedding(nn.Module):
    """
    Converts satellite input channels into feature embeddings.
    Input:  (B, 18, 32, 32)
    Output: (B, 96, 8, 8)
    """

    def __init__(self, in_channels=18, embed_dim=96):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=4,
            stride=4
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)

        # NCHW -> NHWC for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)

        return x

class PatchMerging(nn.Module):
    """
    Reduces spatial resolution and increases feature channels.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(
            4 * in_channels,
            2 * in_channels,
            bias=False
        )

    def forward(self, x):
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)

        # NHWC -> NCHW
        return x.permute(0, 3, 1, 2)


class ChromaFormerStage(nn.Module):
    """
    One stage containing spectral attention and Swin blocks.
    """

    def __init__(self, channels, num_heads, depth):
        super().__init__()

        self.sdm = SpectralDependencyModule(channels)

        self.blocks = nn.ModuleList([
            SwinFeatureBlock(
                channels=channels,
                num_heads=num_heads,
                window_size=4,
                shift_size=0 if i % 2 == 0 else 2
            )
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.sdm(x)

        for block in self.blocks:
            x = block(x)

        return x



class LCZClassificationHead(nn.Module):
    def __init__(self, in_channels=768, num_classes=17):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class ChromaFormer(nn.Module):
    """
    ChromaFormer-inspired model adapted for So2Sat-LCZ42.

    Input:  (B, 18, 32, 32)
    Output: (B, 17)
    """

    def __init__(self, in_channels=18, num_classes=17):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, 96)

        self.stage1 = ChromaFormerStage(96, 3, 2)
        self.merge1 = PatchMerging(96)

        self.stage2 = ChromaFormerStage(192, 6, 2)
        self.merge2 = PatchMerging(192)

        self.stage3 = ChromaFormerStage(384, 12, 6)
        self.merge3 = PatchMerging(384)

        self.stage4 = ChromaFormerStage(768, 24, 2)

        self.classifier = LCZClassificationHead(
            in_channels=768,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.patch_embed(x)

        x = self.stage1(x)
        x = self.merge1(x)

        x = self.stage2(x)
        x = self.merge2(x)

        x = self.stage3(x)
        x = self.merge3(x)

        x = self.stage4(x)

        return self.classifier(x)

if __name__ == "__main__":

    print("Starting ChromaFormer test...")

    # Create the model
    model = ChromaFormer(
        in_channels=18,
        num_classes=17
    )

    # Create a sample satellite image batch
    sample_input = torch.randn(2, 18, 32, 32)

    # Run the model
    with torch.no_grad():
        output = model(sample_input)

    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Expected output: (2, 17)")

    if output.shape == (2, 17):
        print("FORWARD PASS SUCCESSFUL!")
    else:
        print("OUTPUT DIMENSION MISMATCH!")
        


