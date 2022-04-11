import pytorch_lightning as pl
import torch
import torch.nn.functional


class RotationModel(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.first_convolution = torch.nn.Conv2d(3, 12, kernel_size=5, stride=2)
        self.second_convolution = torch.nn.Conv2d(12, 24, kernel_size=5, stride=2)
        self.third_convolution = torch.nn.Conv2d(24, 48, kernel_size=5, stride=2)

        self.first_linear = torch.nn.Linear(48 * 2 * 2, 48 * 2)
        self.second_linear = torch.nn.Linear(48 * 2, 4)

    def forward(self, x):
        x = self.first_convolution(x)
        x = torch.nn.functional.relu(x)
        x = self.second_convolution(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.third_convolution(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1, 48 * 2 * 2)
        x = self.first_linear(x)
        x = torch.nn.functional.relu(x)
        x = self.second_linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        predictions = self.forward(x)
        loss = torch.nn.functional.cross_entropy(predictions, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        predictions = self.forward(x)
        loss = torch.nn.functional.cross_entropy(predictions, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
