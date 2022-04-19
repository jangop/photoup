import torch.nn.functional
import torch.utils.data


class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset) * 4

    def __getitem__(self, idx):
        base_idx, rotation = divmod(idx, 4)
        sample, _ = self.dataset[base_idx]
        if rotation > 0:
            sample = torch.rot90(sample, rotation, [1, 2])
        label = torch.nn.functional.one_hot(torch.tensor(rotation), 4)
        return sample, label
