import pytorch_lightning as pl
import argparse

from experiment import SegmentationExperiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='/home/oodapow/data/rs19')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-nw', '--num_workers', type=int, default=6)
    parser.add_argument('-ts', '--test_steps', type=int, default=1000)
    parser.add_argument('-ni', '--num_images', type=int, default=10)
    parser.add_argument('-ll', '--log_loss_rate', type=int, default=100)
    parser.add_argument('-g', '--gpus', type=int, default=1)
    parser.add_argument('-me', '--max_epochs', type=int, default=100)
    parser.add_argument('-pr', '--progress_bar_refresh_rate', type=int, default=1)
    parser.add_argument('-n', '--name', type=str, default='')
    args = parser.parse_args()

    experiment = SegmentationExperiment(
        args.data_path,
        args.learning_rate,
        args.batch_size,
        args.num_workers,
        args.test_steps,
        args.num_images,
        args.log_loss_rate,
    )
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        logger=pl.loggers.wandb.WandbLogger(project='rail_anomaly_detection', name=args.name, log_model=True),
        enable_checkpointing=True,
    )
    trainer.fit(experiment)