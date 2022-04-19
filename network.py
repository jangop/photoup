import pytorch_lightning as pl
import torch
import torch.nn.functional
import torchmetrics
import torchvision


class RotationModel(pl.LightningModule):
    def __init__(self,):
        super().__init__()

        self.first_normalization = torch.nn.BatchNorm2d(3)
        self.first_convolution = torch.nn.Conv2d(3, 12, kernel_size=5, stride=2)
        self.second_normalization = torch.nn.BatchNorm2d(12)
        self.second_convolution = torch.nn.Conv2d(12, 24, kernel_size=5, stride=2)
        self.third_normalization = torch.nn.BatchNorm2d(24)
        self.third_convolution = torch.nn.Conv2d(24, 48, kernel_size=5, stride=2)
        self.fourth_normalization = torch.nn.BatchNorm2d(48)
        self.first_linear = torch.nn.Linear(48 * 2 * 2, 48 * 2)
        self.fifth_normalization = torch.nn.BatchNorm1d(48 * 2)
        self.second_linear = torch.nn.Linear(48 * 2, 4)

    def forward(self, x):
        x = self.first_normalization(x)
        x = self.first_convolution(x)
        x = torch.nn.functional.relu(x)
        x = self.second_normalization(x)
        x = self.second_convolution(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.third_normalization(x)
        x = self.third_convolution(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.fourth_normalization(x)
        x = x.view(-1, 48 * 2 * 2)
        x = self.first_linear(x)
        x = torch.nn.functional.relu(x)
        x = self.fifth_normalization(x)
        x = self.second_linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        loss = torch.nn.functional.cross_entropy(predictions, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Process data.
        x, y = batch
        predictions = self.forward(x)

        # Compute and log accuracy.
        predicted_labels = torch.argmax(predictions, dim=1)
        y_labels = torch.argmax(y, dim=1)

        validation_accuracy_per_class = torchmetrics.functional.accuracy(
            predicted_labels, y_labels, num_classes=4, average=None,
        )
        for i, accuracy in enumerate(validation_accuracy_per_class):
            self.log(f"val_acc_{i}", accuracy)
        validation_accuracy_averaged = torchmetrics.functional.accuracy(
            predicted_labels, y_labels
        )
        self.log(
            "val_acc_average", validation_accuracy_averaged,
        )

        # Compute and log loss.
        loss = torch.nn.functional.cross_entropy(predictions, y.float())
        self.log("val_loss", loss)

        # Return loss.
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
