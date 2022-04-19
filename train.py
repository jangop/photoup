from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional
import torch.utils.data
import torchvision.datasets
import torchvision.models
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data import RotationDataset
from network import RotationModel


def train(
    batch_size: int,
    n_workers: int,
    device: str,
    n_devices: int,
    max_epochs: int,
    dataset: str,
):
    # Load base dataset.
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(128),
            torchvision.transforms.AutoAugment(),
            torchvision.transforms.ToTensor(),
        ]
    )
    if dataset == "places":
        path = Path("./data/train_256_places365standard.tar")
        base_dataset = torchvision.datasets.Places365(
            root="./data", small=True, download=~path.exists(), transform=transform
        )
    elif dataset == "people":
        base_dataset = torchvision.datasets.LFWPeople(
            root="./data", download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Split dataset into train and test.
    n_samples = len(base_dataset)
    n_samples_training = int(n_samples * 0.8)
    n_samples_validation = n_samples - n_samples_training
    base_dataset_training, base_dataset_validation = torch.utils.data.random_split(
        base_dataset, [n_samples_training, n_samples_validation]
    )

    # Prepare rotated dataset.
    rotation_dataset_training = RotationDataset(base_dataset_training)
    rotation_dataset_validation = RotationDataset(base_dataset_validation)

    # Prepare dataloader.
    data_loader_training = torch.utils.data.DataLoader(
        rotation_dataset_training,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    data_loader_validation = torch.utils.data.DataLoader(
        rotation_dataset_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )

    # Save a random batch of images.
    batch_images, batch_labels = next(iter(data_loader_training))
    grid = torchvision.utils.make_grid(batch_images)
    torchvision.utils.save_image(grid, "batch.png")

    # Prepare model.
    model = RotationModel()

    # Prepare trainer.
    tensor_board = TensorBoardLogger("tb_logs")
    trainer = pl.Trainer(
        logger=tensor_board,
        accelerator=device,
        devices=n_devices,
        max_epochs=max_epochs,
        val_check_interval=0.2,
        callbacks=[EarlyStopping(monitor="val_acc_average", mode="max")],
    )

    # Train model.
    trainer.validate(model, data_loader_validation)
    trainer.fit(
        model,
        train_dataloaders=data_loader_training,
        val_dataloaders=data_loader_validation,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device.",
    )
    parser.add_argument(
        "--n_devices", type=int, default=None, help="Number of devices."
    )
    parser.add_argument("--max_epochs", type=int, default=None, help="Max epochs.")
    parser.add_argument(
        "--dataset", type=str, default="places", choices=["places", "people"]
    )

    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        device=args.device,
        n_devices=args.n_devices,
        max_epochs=args.max_epochs,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
