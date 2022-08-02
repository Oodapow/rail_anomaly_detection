import pytorch_lightning as pl
import argparse

from experiment import SegmentationExperiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/oodapow/data/rs19')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--log_every_n_steps', type=int, default=5)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=1)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--decoder_weight', type=float, default=0.05)
    args = parser.parse_args()

    experiment = SegmentationExperiment(
        args.data_path,
        args.learning_rate,
        args.batch_size,
        args.num_workers,
        args.decoder_weight,
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        logger=pl.loggers.wandb.WandbLogger(project='rail_anomaly_detection', name=args.name, log_model=True),
        callbacks=[pl.callbacks.ModelCheckpoint(save_top_k=2, monitor="eval_mean_iou", filename='{epoch}_{eval_mean_iou:.2f}')],
    )
    trainer.fit(experiment)