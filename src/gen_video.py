import pytorch_lightning as pl
import torch 
import cv2
import numpy as np
import os

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

    experiment.model.to('cuda:0')

    vid_path = '/home/oodapow/code/rail_anomaly_detection/src/alex-pls/sky_sun_rot.mp4'
    out_path = f'{os.path.splitext(vid_path)[0]}_model.mp4'

    frame_size = (960, 540)

    vid_capture = cv2.VideoCapture(vid_path)
    
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        fps = vid_capture.get(5)
        print('Frames per second : ', fps,'FPS')

        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)

    output = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size, True)

    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame.astype('float32'), frame_size)
            in_t = torch.stack([torch.tensor(frame, device='cuda:0').permute(2, 0, 1).div(255.)])

            out_s = experiment.model(in_t)
            out_s = torch.argmax(out_s, dim=1)[0]
            out_s = torch.where(out_s == 0, 1, 0).detach().cpu().numpy()

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame[:,:,1] = frame[:,:,1] * out_s
            frame[:,:,2] = frame[:,:,2] * out_s

            output.write(np.uint8(frame))
        else:
            break

    vid_capture.release()
    output.release()