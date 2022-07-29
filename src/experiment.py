import pytorch_lightning as pl
import torch
from model import UNet
from data import RailSemDataset, make_collate_fn

class SegmentationExperiment(pl.LightningModule):
    def __init__(self, data_dir, learning_rate, batch_size, num_workers, test_steps, num_images, log_loss_rate):
        super().__init__()

        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_steps = test_steps
        self.num_images = num_images
        self.log_loss_rate = log_loss_rate

        self.model = UNet()
        self.loss = torch.nn.MSELoss()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, outputs = batch
        e_outputs = self(images)

        loss = self.loss(e_outputs, outputs)

        if batch_idx % self.log_loss_rate:
            self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, outputs = batch
        e_outputs = self(images)

        loss = self.loss(e_outputs, outputs)

        e_outputs = (e_outputs > 0.5).float()
        inter = torch.count_nonzero(torch.logical_and(outputs == 1, e_outputs == 1), (1, 2))
        union = torch.count_nonzero(torch.add(outputs, e_outputs), (1, 2))
        iou = inter / union

        return images, outputs, e_outputs, loss, iou

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def log_epoch_end(self, all_outputs, split):
        loss_sum = 0
        iou_sum = 0
        for _, _, _, loss, iou in all_outputs:
            loss_sum += loss
            iou_sum += iou
        num_outputs = len(all_outputs)

        self.log(f'{split}/mean_loss', loss_sum / num_outputs)
        self.log(f'{split}/mean_iou', iou_sum / num_outputs)

    def validation_epoch_end(self, all_outputs):
        self.log_epoch_end(all_outputs, 'eval')

    def test_epoch_end(self, all_outputs):
        self.log_epoch_end(all_outputs, 'test')

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            RailSemDataset(self.data_dir, range(1000, 8500)), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=make_collate_fn()
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            RailSemDataset(self.data_dir, range(0, 1000)), 
            batch_size=1, 
            num_workers=self.num_workers,
            collate_fn=make_collate_fn()
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            RailSemDataset(self.data_dir, range(1000, 1000 + self.test_steps)), 
            batch_size=1, 
            num_workers=self.num_workers,
            collate_fn=make_collate_fn()
        )