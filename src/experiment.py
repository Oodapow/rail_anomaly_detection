import pytorch_lightning as pl
import torch
from model import ResNet
from data import RailSemDataset, collate_fn, LABELS

class SegmentationExperiment(pl.LightningModule):
    def __init__(self, 
        data_dir, 
        learning_rate, 
        batch_size, 
        num_workers, 
        decoder_weight,
        lr_factor,
        lr_patience,
        decode_classes=['rail-embedded', 'rail-raised', 'rail-track', 'trackbed']
    ):
        super().__init__()

        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.decoder_weight = decoder_weight
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        self.model = ResNet()
        self.seg_loss = torch.nn.BCELoss()
        self.rec_loss = torch.nn.MSELoss()

        self.decode_index = [i for i, l in enumerate(LABELS) if l in decode_classes]


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        it, ot, iot = batch
        e_ot, e_iot = self(it)

        B, _, H, W = ot.shape

        mask = torch.zeros((B,1,H,W), device=ot.device)

        for i in self.decode_index:
            mask = mask + ot[:,i,:,:].unsqueeze(1)
        
        mask = torch.clamp(mask, 0, 1)

        e_iot = e_iot * mask

        seg_loss = self.seg_loss(e_ot, ot)
        rec_loss = self.rec_loss(e_iot, iot)
        loss = seg_loss + self.decoder_weight * rec_loss

        e_ot = (e_ot > 0.5).float()

        iou = {}
        for i,l in enumerate(LABELS):
            inter = torch.count_nonzero(torch.logical_and(ot[:,i,:,:] == 1, e_ot[:,i,:,:] == 1), (1, 2))
            union = torch.count_nonzero(torch.add(ot[:,i,:,:] , e_ot[:,i,:,:] ), (1, 2))
            iou[l] = (inter / (union + 1)).detach().cpu()

        return {'loss': loss, 'seg_loss': seg_loss.detach().cpu(), 'rec_loss': rec_loss.detach().cpu(), 'iou': iou}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def log_epoch_end(self, all_outputs, split):
        seg_loss_sum = 0
        rec_loss_sum = 0

        iou_sum = {l:0 for l in LABELS}
        iou_count = {l:0 for l in LABELS}
        for x in all_outputs:
            seg_loss, rec_loss, iou = x['seg_loss'], x['rec_loss'], x['iou']
            seg_loss_sum += seg_loss
            rec_loss_sum += rec_loss
            for k in LABELS:
                iou_sum[k] += iou[k].sum()
                iou_count[k] += iou[k].numel()

        num_outputs = len(all_outputs)

        for l in LABELS:
            self.log(f'iou/{split}/{l}', iou_sum[l] / iou_count[k])

        self.log(f'mean/{split}/seg_loss', seg_loss_sum / num_outputs)
        self.log(f'mean/{split}/rec_loss', rec_loss_sum / num_outputs)
        self.log(f'{split}_mean_iou', sum([iou_sum[l] / iou_count[l] for l in LABELS]) / len(LABELS))

    def training_epoch_end(self, all_outputs):
        self.log_epoch_end(all_outputs, 'train')

    def validation_epoch_end(self, all_outputs):
        self.log_epoch_end(all_outputs, 'eval')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    factor=self.lr_factor, 
                    patience=self.lr_patience, 
                    mode='max'
                ),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'eval_mean_iou',
                'strict': True,
                'name': 'learning_rate',
            },
        }

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            RailSemDataset(self.data_dir, range(1000, 8500)), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            RailSemDataset(self.data_dir, range(0, 1000)), 
            batch_size=1, 
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )