
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Save graphs in the project's graphs folder
PROJECT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

# Class counts from your validation dataset
counts = np.array([
    256, 1254, 2353, 849, 757, 1906, 474, 3395, 1914,
    860, 2287, 382, 1202, 2747, 202, 672, 2609
])

classes = [
    "LCZ 1", "LCZ 2", "LCZ 3", "LCZ 4", "LCZ 5",
    "LCZ 6", "LCZ 7", "LCZ 8", "LCZ 9", "LCZ 10",
    "LCZ A", "LCZ B", "LCZ C", "LCZ D", "LCZ E",
    "LCZ F", "LCZ G"
]

percentages = counts / counts.sum() * 100

# Graph 1: class distribution
plt.figure(figsize=(12, 6))
plt.bar(classes, counts)
plt.title("So2Sat-LCZ42: Validation Class Distribution")
plt.xlabel("Local Climate Zone (LCZ) Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "01_class_distribution.png", dpi=150)
plt.show()

# Graph 2: class percentages
plt.figure(figsize=(12, 6))
plt.bar(classes, percentages)
plt.title("So2Sat-LCZ42: Percentage per LCZ Class")
plt.xlabel("Local Climate Zone (LCZ) Class")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "02_class_percentages.png", dpi=150)
plt.show()

# Graph 3: compare largest and smallest classes
plt.figure(figsize=(8, 5))
plt.bar(
    ["LCZ 8: Large low-rise", "LCZ E: Bare rock or paved"],
    [counts[7], counts[14]]
)
plt.title("Class Imbalance in the Validation Dataset")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "03_class_imbalance.png", dpi=150)
plt.show()

print("Done! Three graphs saved in:", GRAPH_DIR)

