import pytorch_lightning as pl

from experiment import RealSegmentationExperiment
from params import make_parser

if __name__ == '__main__':
    parser = make_parser()

    args = parser.parse_args()

    logger = pl.loggers.wandb.WandbLogger(project='rail_anomaly_detection', name=args.name, version=args.version)

    logger.log_hyperparams(args)

    experiment = RealSegmentationExperiment(
        args.data_path,
        args.fake_data_path,
        args.learning_rate,
        args.batch_size,
        args.num_workers,
        args.lr_factor,
        args.lr_patience,
        args.model,
        args.model_state_path,
    )

    trainer = pl.Trainer(
        detect_anomaly=True,
        gradient_clip_val=10,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_top_k=2, monitor='eval_iou', filename='{epoch}_{eval_iou:.2f}', mode='max'),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
    )
    trainer.fit(experiment, ckpt_path=args.ckpt_path)