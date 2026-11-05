import torch
from models.lcz_classifier import LCZClassifier

model = LCZClassifier(num_classes=17)
model.eval()

with torch.no_grad():
    sen1 = torch.randn(4, 8, 32, 32)
    sen2 = torch.randn(4, 10, 32, 32)

    output = model(sen1, sen2)

assert output.shape == (4, 17)
assert torch.isfinite(output).all()

print("Output shape:", output.shape)
print("All predictions are finite.")
print("Full model integration test passed!")