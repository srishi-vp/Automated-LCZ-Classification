
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class So2SatLCZDataset(Dataset):

    def __init__(self, h5_path):
        self.h5_path = h5_path

        # Read dataset dimensions without loading everything into RAM
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["sen1"].shape[0]

            print("Dataset loaded successfully")
            print("Sentinel-1 shape:", f["sen1"].shape)
            print("Sentinel-2 shape:", f["sen2"].shape)
            print("Label shape:", f["label"].shape)

        self.file = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # Open the HDF5 file when needed
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        sen1 = self.file["sen1"][index]
        sen2 = self.file["sen2"][index]
        label = self.file["label"][index]

        # Convert one-hot labels into class indices
        if np.ndim(label) > 0:
            label = np.argmax(label)

        sen1 = torch.tensor(
            np.asarray(sen1, dtype=np.float32)
        )

        sen2 = torch.tensor(
            np.asarray(sen2, dtype=np.float32)
        )

        label = torch.tensor(
            int(label), dtype=torch.long
        )

        return sen1, sen2, label
