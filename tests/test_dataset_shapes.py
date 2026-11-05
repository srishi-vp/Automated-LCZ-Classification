
import torch

# Simulate one So2Sat-LCZ42 sample
sen1 = torch.randn(8, 32, 32)
sen2 = torch.randn(10, 32, 32)
label = torch.tensor(5)

print("Sentinel-1:", sen1.shape)
print("Sentinel-2:", sen2.shape)
print("Label:", label.item())

assert sen1.shape == (8, 32, 32)
assert sen2.shape == (10, 32, 32)
assert 0 <= label.item() <= 16

print("\nAll shape checks passed!")
