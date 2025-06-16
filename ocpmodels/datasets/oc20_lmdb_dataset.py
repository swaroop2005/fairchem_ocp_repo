import lmdb
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

def to_tensor(x, dtype=None):
    # Converts numpy/array/list/scalar to torch tensor
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype)
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t if dtype is None else t.to(dtype)
    if isinstance(x, (float, int)):
        return torch.tensor([x], dtype=dtype or torch.float32)
    if isinstance(x, list):
        t = torch.tensor(x)
        return t if dtype is None else t.to(dtype)
    # fallback
    return torch.tensor(x, dtype=dtype)

class OC20LmdbDataset(Dataset):
    """
    Minimal OC20 LMDB → PyG Dataset (robust to PyG version mismatches).
    Always returns a new Data object, never the pickled one!
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
            # Only get data keys, ignore __len__ and __keys__ if present
            self.keys = [k for k, _ in txn.cursor() if not k.startswith(b"__")]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        with self.env.begin(write=False) as txn:
            sample = pickle.loads(txn.get(self.keys[idx]))

        # Handle both dict and old Data object-style samples
        if isinstance(sample, dict):
            z = to_tensor(sample["atomic_numbers"], dtype=torch.long)
            pos = to_tensor(sample["pos"], dtype=torch.float)
            cell = to_tensor(sample["cell"], dtype=torch.float) if "cell" in sample else None
            edge_index = to_tensor(sample["edge_index"], dtype=torch.long) if "edge_index" in sample else None
            cell_offsets = to_tensor(sample["cell_offsets"], dtype=torch.float) if "cell_offsets" in sample else None
            edge_attr = sample.get("edge_attr", None)
            if edge_attr is None and "distances" in sample and sample["distances"] is not None:
                edge_attr = to_tensor(sample["distances"], dtype=torch.float).unsqueeze(-1)
            elif edge_attr is not None:
                edge_attr = to_tensor(edge_attr, dtype=torch.float)
            else:
                edge_attr = None
            y = sample.get(self.target_key, None)
            if y is not None:
                y = to_tensor(y, dtype=torch.float)
                if y.ndim == 0:
                    y = y.unsqueeze(0)
        else:
            # Old Data object (from previous PyG)
            z = to_tensor(getattr(sample, "atomic_numbers"), dtype=torch.long)
            pos = to_tensor(getattr(sample, "pos"), dtype=torch.float)
            cell = to_tensor(getattr(sample, "cell"), dtype=torch.float) if hasattr(sample, "cell") else None
            edge_index = to_tensor(getattr(sample, "edge_index"), dtype=torch.long) if hasattr(sample, "edge_index") else None
            cell_offsets = to_tensor(getattr(sample, "cell_offsets"), dtype=torch.float) if hasattr(sample, "cell_offsets") else None
            edge_attr = getattr(sample, "edge_attr", None)
            if edge_attr is None and hasattr(sample, "distances") and getattr(sample, "distances") is not None:
                edge_attr = to_tensor(getattr(sample, "distances"), dtype=torch.float).unsqueeze(-1)
            elif edge_attr is not None:
                edge_attr = to_tensor(edge_attr, dtype=torch.float)
            else:
                edge_attr = None
            y = getattr(sample, self.target_key, None)
            if y is not None:
                y = to_tensor(y, dtype=torch.float)
                if y.ndim == 0:
                    y = y.unsqueeze(0)

        # Always create a NEW Data object!
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
