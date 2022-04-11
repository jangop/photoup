import torch
import torch.utils.data
import torchvision.datasets
import pytorch_lightning as pl
from data import RotationDataset
import torch.nn.functional
import torchvision.models

from network import RotationModel


def train(
    batch_size: int, n_workers: int, device: str, n_devices: int, max_epochs: int
):
    # Load base dataset.
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(128),
            torchvision.transforms.AutoAugment(),
            torchvision.transforms.ToTensor(),
        ]
    )
    base_dataset = torchvision.datasets.LFWPeople(
        root="./data", download=True, transform=transform
    )

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
    trainer = pl.Trainer(accelerator=device, devices=n_devices, max_epochs=max_epochs)

    # Train model.
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

    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        device=args.device,
        n_devices=args.n_devices,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
