
import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class TSSERes2Former(nn.Module):

    def __init__(self, num_classes=17):
        super().__init__()

        # Extract and fuse Sentinel-1 and Sentinel-2 features
        self.feature_fusion = FeatureFusion()

        # Classification head for 17 LCZ classes
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, sen1, sen2):

        # Extract fused spatial features
        features = self.feature_fusion(sen1, sen2)

        # Average the spatial tokens
        features = features.mean(dim=1)

        # Generate class scores
        output = self.classifier(features)

        return output


if __name__ == "__main__":

    model = TSSERes2Former(num_classes=17)

    sen1 = torch.randn(2, 8, 32, 32)
    sen2 = torch.randn(2, 10, 32, 32)

    output = model(sen1, sen2)

    print("Sentinel-1:", sen1.shape)
    print("Sentinel-2:", sen2.shape)
    print("Model output:", output.shape)
    print("Predicted classes:", torch.argmax(output, dim=1))
