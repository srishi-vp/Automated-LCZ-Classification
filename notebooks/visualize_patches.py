

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT / "data" / "validation.h5"
GRAPH_DIR = PROJECT / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    print("validation.h5 not found in the data folder.")
    print("Please check where your dataset file is saved.")
else:
    with h5py.File(DATA_FILE, "r") as f:
        sen1 = f["sen1"][0]
        sen2 = f["sen2"][0]
        label = int(np.argmax(f["label"][0]))

    print("Sample LCZ class index:", label)

    # Sentinel-1: all 8 radar channels
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        image = sen1[:, :, i]
        low, high = np.percentile(image, [2, 98])
        ax.imshow(image, cmap="gray", vmin=low, vmax=high)
        ax.set_title(f"Sentinel-1 Channel {i+1}")
        ax.axis("off")
    fig.suptitle("Sentinel-1 Sample Patch")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "04_sentinel1_patch.png", dpi=150)
    plt.show()

    # Sentinel-2: all 10 optical bands
    band_names = [
        "B2 Blue", "B3 Green", "B4 Red", "B5 Red Edge",
        "B6 Red Edge", "B7 Red Edge", "B8 NIR",
        "B8A Narrow NIR", "B11 SWIR", "B12 SWIR"
    ]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        image = sen2[:, :, i]
        low, high = np.percentile(image, [2, 98])
        ax.imshow(image, cmap="gray", vmin=low, vmax=high)
        ax.set_title(band_names[i])
        ax.axis("off")
    fig.suptitle("Sentinel-2 Sample Patch")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "05_sentinel2_patch.png", dpi=150)
    plt.show()

    print("Saved both patch visualizations in:", GRAPH_DIR)
