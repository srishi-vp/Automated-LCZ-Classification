
import torch
import torch.nn as nn


class Res2NetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale=4):
        super().__init__()

        if out_channels % scale != 0:
            raise ValueError("out_channels must be divisible by scale")

        self.scale = scale
        self.width = out_channels // scale

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.width, self.width,
                    kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(self.width),
                nn.ReLU(inplace=True)
            )
            for _ in range(scale - 1)
        ])

        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.conv1(x)

        chunks = torch.split(x, self.width, dim=1)

        outputs = [chunks[0]]
        previous = chunks[0]

        for i in range(1, self.scale):
            previous = self.convs[i - 1](
                chunks[i] + previous
            )
            outputs.append(previous)

        x = torch.cat(outputs, dim=1)

        x = self.conv3(x)

        return self.relu(x + residual)


if __name__ == "__main__":

    model = Res2NetBlock(64, 128)

    sample = torch.randn(2, 64, 32, 32)

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
