
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT / "report"
REPORT_DIR.mkdir(exist_ok=True)

counts = [
    256, 1254, 2353, 849, 757, 1906, 474, 3395, 1914,
    860, 2287, 382, 1202, 2747, 202, 672, 2609
]

classes = [
    "LCZ 1 - Compact high-rise",
    "LCZ 2 - Compact mid-rise",
    "LCZ 3 - Compact low-rise",
    "LCZ 4 - Open high-rise",
    "LCZ 5 - Open mid-rise",
    "LCZ 6 - Open low-rise",
    "LCZ 7 - Lightweight low-rise",
    "LCZ 8 - Large low-rise",
    "LCZ 9 - Sparsely built",
    "LCZ 10 - Heavy industry",
    "LCZ A - Dense trees",
    "LCZ B - Scattered trees",
    "LCZ C - Bush, scrub",
    "LCZ D - Low plants",
    "LCZ E - Bare rock or paved",
    "LCZ F - Bare soil or sand",
    "LCZ G - Water"
]

total = sum(counts)
percentages = [count / total * 100 for count in counts]

df = pd.DataFrame({
    "Class index": range(17),
    "LCZ class": classes,
    "Sample count": counts,
    "Percentage": [round(p, 2) for p in percentages]
})

df.to_csv(REPORT_DIR / "class_distribution.csv", index=False)

report = f"""
So2Sat-LCZ42 Dataset Exploration Report

1. Dataset Overview
The validation split contains {total:,} samples.
Each sample contains:
- Sentinel-1 data: 32 x 32 pixels with 8 channels.
- Sentinel-2 data: 32 x 32 pixels with 10 channels.
- Ground-truth label: a 17-element one-hot vector.

2. Ground Truth
Each label vector has 17 positions representing LCZ classes.
The position containing 1 indicates the class, with indices 0 to 16.

3. Class Distribution
The largest class is LCZ 8 (Large low-rise), with
{counts[7]:,} samples ({percentages[7]:.2f}%).

The smallest class is LCZ E (Bare rock or paved), with
{counts[14]:,} samples ({percentages[14]:.2f}%).

The largest class has about {counts[7] / counts[14]:.1f}
times as many samples as the smallest class, showing class imbalance.

4. Data Quality
The validation data check found:
- Sentinel-1: 0 NaN and 0 Inf.
- Sentinel-2: 0 NaN and 0 Inf.
- Labels: 0 NaN and 0 Inf.

5. Preprocessing Recommendations
- Keep the class-index mapping consistent.
- Fit normalization parameters on training data only.
- Consider class weighting or balanced sampling during training.
- Preserve the order of Sentinel-1 and Sentinel-2 channels.

6. Visualization
The associated graphs show class counts, class percentages,
class imbalance, and sample Sentinel-1/Sentinel-2 channel patches.

Note: Data-quality results above apply to the validation split checked.
"""

output = REPORT_DIR / "dataset_exploration_report.txt"
output.write_text(report, encoding="utf-8")

print("Report saved to:", output)
print("Class distribution CSV saved to:", REPORT_DIR / "class_distribution.csv")
