import pytorch_lightning as pl
import torch 
import cv2

from experiment import SegmentationExperiment
from params import make_parser

if __name__ == '__main__':
    parser = make_parser()

    args = parser.parse_args()

    print(args.ae_classes_only)

    experiment = SegmentationExperiment.load_from_checkpoint(
        args.ckpt_path,
        strict=False,
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
        ae_classes_only=args.ae_classes_only,
    )

    image = cv2.imread("/home/oodapow/code/rail_anomaly_detection/src/895.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    in_t = torch.stack([torch.tensor(cv2.resize(image.astype('float32'), (960, 540))).permute(2, 0, 1).div(255.)])

    out_s, out_r = experiment.model(in_t)

    out_s, out_r = torch.argmax(out_s, dim=1)[0], out_r[0].permute(1, 2, 0).detach().cpu().numpy() * 255

    #out_s = torch.where(torch.logical_or(out_s == 17, out_s == 12), 255, 0).detach().cpu().numpy()
    out_s = torch.where(out_s > 0, 255, 0).detach().cpu().numpy()

    cv2.imwrite("seg.png", out_s)
    cv2.imwrite("rec.png", out_r)

    print(out_s.shape)
    print(out_r.shape)

    # torch.onnx.export(
    #     experiment.model,
    #     in_t,
    #     args.model_export_path,
    #     opset_version=11,
    #     do_constant_folding=False,
    # )