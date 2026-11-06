
"""
So2Sat-LCZ42 dataset exploration
Run from VS Code using your project's .venv Python.

First run downloads the validation split from Hugging Face to data/.
It may take time and needs several GB of free disk space.
"""

from pathlib import Path
import gzip
import shutil
import requests
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project folders (this script is saved in the notebooks folder)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

GRAPHS_DIR = PROJECT_DIR / "graphs"
REPORT_DIR = PROJECT_DIR / "report"
for folder in (DATA_DIR, GRAPHS_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

URL = "https://huggingface.co/datasets/zhu-xlab/So2Sat-LCZ42/resolve/main/v4/validation.h5.gz"
GZ_PATH = DATA_DIR / "validation.h5.gz"
H5_PATH = DATA_DIR / "validation.h5"

CLASS_NAMES = [
    "LCZ 1 Compact high-rise",
    "LCZ 2 Compact mid-rise",
    "LCZ 3 Compact low-rise",
    "LCZ 4 Open high-rise",
    "LCZ 5 Open mid-rise",
    "LCZ 6 Open low-rise",
    "LCZ 7 Lightweight low-rise",
    "LCZ 8 Large low-rise",
    "LCZ 9 Sparsely built",
    "LCZ 10 Heavy industry",
    "LCZ A Dense trees",
    "LCZ B Scattered trees",
    "LCZ C Bush, scrub",
    "LCZ D Low plants",
    "LCZ E Bare rock or paved",
    "LCZ F Bare soil or sand",
    "LCZ G Water",
]
S2_BANDS = [
    "B2 - Blue", "B3 - Green", "B4 - Red", "B5 - Red edge",
    "B6 - Red edge", "B7 - Red edge", "B8 - Near infrared",
    "B8A - Narrow near infrared", "B11 - SWIR", "B12 - SWIR",
]

def download_and_extract():
    if H5_PATH.exists():
        print(f"Using existing dataset: {H5_PATH}")
        return
    if not GZ_PATH.exists():
        print("Downloading validation split. This can take a while...")
        with requests.get(URL, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            done = 0
            with GZ_PATH.open("wb") as f:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\rDownloaded {done / total:.0%}", end="")
        print("\nDownload complete.")
    print("Extracting validation file. Please wait...")
    with gzip.open(GZ_PATH, "rb") as source, H5_PATH.open("wb") as dest:
        shutil.copyfileobj(source, dest, length=4 * 1024 * 1024)
    print(f"Extracted to: {H5_PATH}")

def main():
    print("So2Sat-LCZ42 Dataset Exploration")
    print("Project folder:", PROJECT_DIR)
    download_and_extract()

    with h5py.File(H5_PATH, "r") as f:
        print("\nDataset fields and shapes:")
        for key in f.keys():
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")
        n = f["label"].shape[0]
        labels = f["label"][:]
        sen1_shape, sen2_shape = f["sen1"].shape, f["sen2"].shape

        binary = bool(np.all((labels == 0) | (labels == 1)))
        one_hot = bool(np.all(labels.sum(axis=1) == 1))
        class_indices = np.argmax(labels, axis=1)
        counts = np.bincount(class_indices, minlength=17)
        percentages = counts / n * 100
        print("\nClass counts:")
        for i, count in enumerate(counts):
            print(f"{i}: {count} ({percentages[i]:.2f}%)")

        print("\nChecking for missing or infinite values...")
        for name in ("sen1", "sen2", "label"):
            dataset = f[name]
            nan_count = 0
            inf_count = 0

            for start in range(0, dataset.shape[0], 256):
                block = dataset[start:start + 256]
                nan_count += int(np.isnan(block).sum())
                inf_count += int(np.isinf(block).sum())

            print(f"{name}: NaN={nan_count}, Inf={inf_count}")

if __name__ == "__main__":
    main()
    
