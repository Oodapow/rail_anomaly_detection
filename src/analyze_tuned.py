import pytorch_lightning as pl
import torch 
import cv2
import numpy as np

from experiment import RealSegmentationExperiment
from params import make_parser

if __name__ == '__main__':
    parser = make_parser()

    args = parser.parse_args()

    experiment = RealSegmentationExperiment.load_from_checkpoint(
        args.ckpt_path,
        data_path=args.data_path,
        fake_data_path=args.fake_data_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        model=args.model,
        model_state_path=args.model_state_path
    )

    image = cv2.imread("/home/oodapow/code/rail_anomaly_detection/src/net_res/0img.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image.astype('float32'), (960, 540))
    in_t = torch.stack([torch.tensor(image).permute(2, 0, 1).div(255.)])

    out_s = experiment.model(in_t)

    out_s = torch.argmax(out_s, dim=1)[0]
    out_s = torch.where(out_s == 0, 1, 0).detach().cpu().numpy()

    print(out_s.shape)
    print(image.shape)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image[:,:,1] = image[:,:,1] * out_s
    image[:,:,2] = image[:,:,2] * out_s

    cv2.imwrite("overlay.png", image)

    print(out_s.shape)