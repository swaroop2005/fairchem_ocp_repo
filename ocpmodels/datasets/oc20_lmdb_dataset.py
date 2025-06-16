"""
Robust OC20 LMDB → PyG Dataset
------------------------------
• Reads OC20 IS2RE (or S2EF) LMDB files.
• Pass-through: uses the pre-computed edge_index, edge_attr, cell_offsets
  already stored in each LMDB entry.
• Returns a torch_geometric.data.Data object with keys used by GemNet/OCP, but
  is robust to PyG version mismatches (does NOT unpickle old Data objects).
• Handles both tensor and numpy array types in LMDB.
"""

import lmdb
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def to_tensor(x):
    """Convert numpy array or list to torch tensor if not already."""
    if isinstance(x, torch.Tensor):
        return x
    import numpy as np
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (float, int)):
        return torch.tensor([x], dtype=torch.float32)
    return torch.tensor(x)

class OC20LmdbDataset(Dataset):
    """
    Args
    ----
    lmdb_path : str
        Path to `data.lmdb` file (train or val).
    target_key : str
        Which energy label to use.  'y_relaxed' for IS2RE
        or 'y' for generic S2EF frame.
    """

    def __init__(self, lmdb_path: str, target_key: str = "y_relaxed"):
        self.lmdb_path = lmdb_path
        self.target_key = target_key

        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin() as txn:
            self.keys = [k for k, _ in txn.cursor() if not k.startswith(b'__')]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        with self.env.begin(write=False) as txn:
            sample = pickle.loads(txn.get(self.keys[idx]))

        # Handle both dict and Data object style
        if isinstance(sample, dict):
            z = to_tensor(sample["atomic_numbers"]).long()
            pos = to_tensor(sample["pos"]).float()
            cell = to_tensor(sample["cell"]).squeeze().float() if "cell" in sample else None
            edge_index = to_tensor(sample["edge_index"]).long() if "edge_index" in sample else None
            cell_offsets = to_tensor(sample["cell_offsets"]).float() if "cell_offsets" in sample else None
            edge_attr = sample.get("edge_attr", None)
            if edge_attr is None and "distances" in sample and sample["distances"] is not None:
                edge_attr = to_tensor(sample["distances"]).unsqueeze(-1).float()
            elif edge_attr is not None:
                edge_attr = to_tensor(edge_attr).float()
            else:
                raise ValueError("No edge_attr or distances found in sample!")
            y = torch.tensor([sample[self.target_key]], dtype=torch.float32)
        else:
            # OCP Data object, still robustly extract tensors
            z = to_tensor(sample.atomic_numbers).long()
            pos = to_tensor(sample.pos).float()
            cell = to_tensor(sample.cell).squeeze().float() if hasattr(sample, "cell") else None
            edge_index = to_tensor(sample.edge_index).long() if hasattr(sample, "edge_index") else None
            cell_offsets = to_tensor(sample.cell_offsets).float() if hasattr(sample, "cell_offsets") else None
            edge_attr = getattr(sample, "edge_attr", None)
            if edge_attr is None and hasattr(sample, "distances") and sample.distances is not None:
                edge_attr = to_tensor(sample.distances).unsqueeze(-1).float()
            elif edge_attr is not None:
                edge_attr = to_tensor(edge_attr).float()
            else:
                raise ValueError("No edge_attr or distances found in sample!")
            y = torch.tensor([getattr(sample, self.target_key)], dtype=torch.float32)

        data = Data(
            z=z,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cell_offsets=cell_offsets,
            y=y,
            natoms=torch.tensor([pos.size(0)], dtype=torch.long),
        )
        return data
