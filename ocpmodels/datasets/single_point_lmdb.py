import errno
import os
import pickle
from pathlib import Path

import lmdb
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ocpmodels.common.registry import registry

@registry.register_dataset("single_point_lmdb")
class SinglePointLmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing single point computations.
    Useful for Initial Structure to Relaxed Energy (IS2RE) task.

    Args:
        config (dict): Dataset configuration
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(SinglePointLmdbDataset, self).__init__()

        self.config = config

        self.db_path = Path(self.config["src"])
        if not self.db_path.is_file():
            raise FileNotFoundError(
                errno.ENOENT, "LMDB file not found", str(self.db_path)
            )

        self.metadata_path = self.db_path.parent / "metadata.npz"

        self.env = self.connect_db(self.db_path)

        self._keys = [
            f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
        ]
        self.transform = transform

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        # Return features.
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        sample = pickle.loads(datapoint_pickled)

        # REBUILD a new Data object using raw tensor fields (avoids version issues)
        atomic_numbers = sample.atomic_numbers.long()
        pos = sample.pos
        cell = sample.cell
        edge_index = sample.edge_index
        cell_offsets = sample.cell_offsets

        # Use distances as edge_attr if edge_attr is None
        edge_attr = getattr(sample, 'edge_attr', None)
        if edge_attr is None and hasattr(sample, "distances") and sample.distances is not None:
            edge_attr = sample.distances.unsqueeze(-1)
        elif edge_attr is None:
            raise ValueError("No edge_attr or distances found in sample!")

        # Choose energy label (y/y_relaxed) if present
        if hasattr(sample, "y_relaxed"):
            y_relaxed = torch.tensor([sample.y_relaxed], dtype=torch.float32)
        elif hasattr(sample, "y"):
            y_relaxed = torch.tensor([sample.y], dtype=torch.float32)
        else:
            y_relaxed = None

        data = Data(
            atomic_numbers=atomic_numbers,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cell_offsets=cell_offsets,
            y_relaxed=y_relaxed,
            natoms=torch.tensor([pos.size(0)], dtype=torch.long),
        )

        if self.transform is not None:
            data = self.transform(data)

        return data

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        self.env.close()
