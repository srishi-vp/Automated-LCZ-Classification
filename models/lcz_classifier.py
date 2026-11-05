import torch
import torch.nn as nn

from models.feature_fusion import FeatureFusion


class LCZClassifier(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()

        self.feature_fusion = FeatureFusion()

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, sen1, sen2):
        features = self.feature_fusion(sen1, sen2)

        # Average the 64 spatial tokens
        features = features.mean(dim=1)

        # Predict 17 LCZ classes
        output = self.classifier(features)

        return output


if __name__ == "__main__":
    model = LCZClassifier(num_classes=17)

    sen1 = torch.randn(2, 8, 32, 32)
    sen2 = torch.randn(2, 10, 32, 32)

    output = model(sen1, sen2)

    print("Classification output:", output.shape)
    print("Predicted classes:", torch.argmax(output, dim=1))