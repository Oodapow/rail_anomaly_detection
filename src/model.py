import torch
import torchvision

class ResNet(torch.nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.transform = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        backbone = torchvision.models.resnet18(pretrained=True, norm_layer=torchvision.ops.misc.FrozenBatchNorm2d)

        self.backbone = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )

        self.seg_head =  torch.nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
        self.rec_head =  torch.nn.Conv2d(64, 3, kernel_size=1, bias=False)

        self.sigmoid = torch.nn.Sigmoid()

    
    def forward(self, x):
        x = self.transform(x)

        x = self.backbone(x)
        
        x_seg = self.seg_head(x)
        x_rec = self.rec_head(x)

        x_seg = self.sigmoid(x_seg)
        x_rec = self.sigmoid(x_rec)

        return x_seg, x_rec
        
if __name__ == '__main__':

    model = ResNet()

    in_t = torch.rand((1, 3, 1080, 1920))

    out_s, out_r = model(in_t)

    print(out_s.shape)
    print(out_r.shape)

    torch.onnx.export(
        model,
        in_t,
        'model.onnx',
        opset_version=11,
        do_constant_folding=False,
    )
