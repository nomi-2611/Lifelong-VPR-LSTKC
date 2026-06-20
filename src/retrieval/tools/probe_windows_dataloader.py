import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    def __len__(self):
        return 256

    def __getitem__(self, index):
        return torch.tensor(index, dtype=torch.int64)


def main():
    for workers in [1, 2, 4, 8]:
        print(f"workers={workers}")
        try:
            loader = DataLoader(
                DummyDataset(),
                batch_size=16,
                num_workers=workers,
                pin_memory=True,
                persistent_workers=(workers > 0),
            )
            batch = next(iter(loader))
            print("ok", batch[:4].tolist())
        except Exception as exc:
            print("fail", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
