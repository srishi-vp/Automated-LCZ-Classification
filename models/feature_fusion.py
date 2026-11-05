
import torch
import torch.nn as nn

from models.res2net_block import Res2NetBlock
from models.transformer_block import TransformerBlock


class FeatureFusion(nn.Module):

    def __init__(self):
        super().__init__()

        # Sentinel-1 feature extraction
        self.sentinel1_stem = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.sentinel1_res2net = Res2NetBlock(32, 64)

        # Sentinel-2 feature extraction
        self.sentinel2_stem = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.sentinel2_res2net = Res2NetBlock(32, 64)

        # Reduce spatial dimensions to 8 x 8
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # Transformer processes fused spatial tokens
        self.transformer = TransformerBlock(
            embed_dim=128,
            num_heads=4
        )

    def forward(self, sen1, sen2):

        # Extract Sentinel-1 features
        f1 = self.sentinel1_stem(sen1)
        f1 = self.sentinel1_res2net(f1)

        # Extract Sentinel-2 features
        f2 = self.sentinel2_stem(sen2)
        f2 = self.sentinel2_res2net(f2)

        # Reduce spatial dimensions
        f1 = self.pool(f1)
        f2 = self.pool(f2)

        # Concatenate both modalities
        fused = torch.cat([f1, f2], dim=1)

        # Convert feature maps into spatial tokens
        batch_size = fused.shape[0]

        fused = fused.flatten(2).transpose(1, 2)

        # Transformer feature learning
        output = self.transformer(fused)

        return output


if __name__ == "__main__":

    model = FeatureFusion()

    # Dummy Sentinel-1 and Sentinel-2 inputs
    sen1 = torch.randn(2, 8, 32, 32)
    sen2 = torch.randn(2, 10, 32, 32)

    output = model(sen1, sen2)

    print("Sentinel-1 input:", sen1.shape)
    print("Sentinel-2 input:", sen2.shape)
    print("Fused output:", output.shape)
