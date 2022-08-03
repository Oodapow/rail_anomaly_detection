import pytorch_lightning as pl
import argparse

from experiment import SegmentationExperiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/oodapow/data/rs19')
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=1)
    parser.add_argument('--name', type=str, default='rail_anomaly_detection')
    parser.add_argument('--version', type=str, default='')
    parser.add_argument('--decoder_weight', type=float, default=0.05)
    parser.add_argument('--lr_factor', type=float, default=0.3)
    parser.add_argument('--lr_patience', type=int, default=20)
    parser.add_argument('--default_root_dir', type=str, default='/home/oodapow/experiments/')
    args = parser.parse_args()

    logger = pl.loggers.wandb.WandbLogger(project='rail_anomaly_detection', name=args.name, version=args.version, log_model='all')

    logger.log_hyperparams(args)

    experiment = SegmentationExperiment(
        args.data_path,
        args.learning_rate,
        args.batch_size,
        args.num_workers,
        args.decoder_weight,
        args.lr_factor,
        args.lr_patience,
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_top_k=2, monitor='eval_mean_iou', filename='{epoch}_{eval_mean_iou:.2f}', mode='max'),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
    )
    trainer.fit(experiment)