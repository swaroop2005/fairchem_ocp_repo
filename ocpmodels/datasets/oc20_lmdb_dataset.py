import lmdb
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class OC20LmdbDataset(Dataset):
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
            self.keys = [k for k, _ in txn.cursor()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        with self.env.begin(write=False) as txn:
            sample = pickle.loads(txn.get(self.keys[idx]))
        z = sample.atomic_numbers.long()
        pos = sample.pos
        cell = sample.cell.squeeze()
        edge_index = sample.edge_index
        cell_offsets = sample.cell_offsets
        edge_attr = sample.edge_attr
        if edge_attr is None and hasattr(sample, "distances") and sample.distances is not None:
            edge_attr = sample.distances.unsqueeze(-1)
        elif edge_attr is None:
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
