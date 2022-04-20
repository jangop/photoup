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


def train(args):
    # Load base dataset.
    if args.transforms == "auto":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(128),
                torchvision.transforms.AutoAugment(),
                torchvision.transforms.ToTensor(),
            ]
        )
    elif args.transforms == "none":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(128),
                torchvision.transforms.ToTensor(),
            ]
        )
    elif args.transforms == "soft":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(128),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),
                torchvision.transforms.ToTensor(),
            ]
        )
    elif args.transforms == "hard":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomPerspective(fill=128),
                torchvision.transforms.RandomResizedCrop(128),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),
                torchvision.transforms.RandomAdjustSharpness(
                    sharpness_factor=0.5, p=0.25
                ),
                torchvision.transforms.RandomAutocontrast(p=0.25),
                torchvision.transforms.RandomEqualize(p=0.25),
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        raise ValueError(f"Unknown transforms: {args.transforms}")

    if args.dataset == "places":
        path = Path("./data/data_256_standard")
        download = not path.exists()
        print(f"{download = }")
        base_dataset = torchvision.datasets.Places365(
            root="./data", small=True, download=download, transform=transform
        )
    elif args.dataset == "people":
        base_dataset = torchvision.datasets.LFWPeople(
            root="./data", download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    data_loader_validation = torch.utils.data.DataLoader(
        rotation_dataset_validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Save a random batch of images.
    batch_images, batch_labels = next(iter(data_loader_training))
    grid = torchvision.utils.make_grid(batch_images)
    torchvision.utils.save_image(grid, "batch.png")

    # Prepare model.
    model = RotationModel(args)

    # Prepare trainer.
    tensor_board = TensorBoardLogger("tb_logs")
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tensor_board,
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
        "--device", type=str, default="auto", help="Device.",
    )
    parser.add_argument(
        "--n_devices", type=int, default=None, help="Number of devices."
    )
    parser.add_argument(
        "--dataset", type=str, default="places", choices=["places", "people"]
    )

    parser = RotationModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
