import pytorch_lightning as pl
import torch 

from experiment import SegmentationExperiment
from params import make_parser

if __name__ == '__main__':
    parser = make_parser()

    args = parser.parse_args()

    logger = pl.loggers.wandb.WandbLogger(project='rail_anomaly_detection', name=args.name, version=args.version, log_model='all')

    experiment = SegmentationExperiment.load_from_checkpoint(
        args.ckpt_path,
        data_path=args.data_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        decoder_weight=args.decoder_weight,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        loss_weights=args.loss_weights,
        model=args.model,
        teacher_model=args.teacher_model,
        teacher_loss_weight=args.teacher_loss_weight,
        teacher_loss_temperature=args.teacher_loss_temperature,
        teacher_state_path=args.teacher_state_path,
    )

    in_t = torch.rand((1, 3, 540, 960))

    out_s, out_r = experiment.model(in_t)

    print(out_s.shape)
    print(out_r.shape)

    torch.onnx.export(
        experiment.model,
        in_t,
        args.model_export_path,
        opset_version=11,
        do_constant_folding=False,
    )