import pytorch_lightning as pl
import torch
from model import UNet, UNetTeacher
from data import RailSemDataset, collate_fn, LABELS, LABELS_WEIGHTS

class SegmentationExperiment(pl.LightningModule):
    def __init__(self, 
        data_path, 
        learning_rate, 
        batch_size, 
        num_workers, 
        decoder_weight,
        lr_factor,
        lr_patience,
        loss_weights,
        model,
        teacher_model,
        teacher_loss_weight,
        teacher_loss_temperature,
        teacher_state_path,
        decode_classes=['rail-embedded', 'rail-raised', 'rail-track', 'trackbed']
    ):
        super().__init__()

        self.data_path = data_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.decoder_weight = decoder_weight
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.teacher_loss_weight = teacher_loss_weight
        self.teacher_loss_temperature = teacher_loss_temperature

        self.model = eval(model)()

        if teacher_model:
            self.teacher_model = eval(teacher_model)()
            self.teacher_model.load_state_dict(torch.load(teacher_state_path))
            self.teacher_model = self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.teacher_model = None

        self.seg_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(LABELS_WEIGHTS) if loss_weights else None)
        self.rec_loss = torch.nn.MSELoss()
        self.teacher_loss = torch.nn.KLDivLoss(reduction='mean')

        self.decode_index = [i for i, l in enumerate(LABELS) if l in decode_classes]

    @torch.no_grad()
    def teacher_forward(self, x):
        return self.teacher_model(x)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        it, tgt, ot, iot = batch
        e_ot, e_iot = self(it)

        B, _, H, W = ot.shape

        mask = torch.zeros((B,1,H,W), device=ot.device)

        for i in self.decode_index:
            mask = mask + ot[:,i,:,:].unsqueeze(1)
        
        mask = torch.clamp(mask, 0, 1)

        e_iot = e_iot * mask

        seg_loss = self.seg_loss(e_ot, tgt)
        rec_loss = self.rec_loss(e_iot, iot)
        loss = seg_loss + self.decoder_weight * rec_loss

        e_tgt = torch.argmax(e_ot, dim=1)

        iou = {}
        for i,l in enumerate(LABELS):
            inter = torch.count_nonzero(torch.logical_and(tgt == i, e_tgt == i), (1, 2))
            union = torch.count_nonzero(torch.logical_or(tgt == i, e_tgt == i), (1, 2))
            iou[l] = (inter / (union + 1)).detach().cpu()

        res = {'loss': loss, 'seg_loss': seg_loss.detach().cpu(), 'rec_loss': rec_loss.detach().cpu(), 'iou': iou}

        if self.teacher_model:
            et_ot, _ = self.teacher_forward(it)

            T = self.teacher_loss_temperature
            W = self.teacher_loss_weight

            tch_loss = T * T * W * self.teacher_loss(
                torch.nn.functional.log_softmax(e_ot / T, 1), 
                torch.nn.functional.softmax(et_ot / T, 1)
            )

            res['loss'] = loss + tch_loss
            res['tch_loss'] = tch_loss.detach().cpu()

        return res

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def log_epoch_end(self, all_outputs, split):
        seg_loss_sum = 0
        rec_loss_sum = 0
        tch_loss_sum = 0

        iou_sum = {l:0 for l in LABELS}
        iou_count = {l:0 for l in LABELS}
        for x in all_outputs:
            seg_loss, rec_loss, tch_loss, iou,  = x['seg_loss'], x['rec_loss'], x.get('tch_loss', 0), x['iou']
            seg_loss_sum += seg_loss
            rec_loss_sum += rec_loss
            tch_loss_sum += tch_loss
            for k in LABELS:
                iou_sum[k] += iou[k].sum()
                iou_count[k] += iou[k].numel()

        num_outputs = len(all_outputs)

        for l in LABELS:
            self.log(f'iou/{split}/{l}', iou_sum[l] / iou_count[k])

        self.log(f'mean/{split}/seg_loss', seg_loss_sum / num_outputs)
        self.log(f'mean/{split}/rec_loss', rec_loss_sum / num_outputs)
        if self.teacher_model:
            self.log(f'mean/{split}/tch_loss', tch_loss_sum / num_outputs)
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
            RailSemDataset(self.data_path, range(1000, 8500)), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            RailSemDataset(self.data_path, range(0, 1000)), 
            batch_size=1, 
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )