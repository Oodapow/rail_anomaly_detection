import torch
import torchvision

from torch import Tensor
from typing import List

class UNet(torch.nn.Module):
    def __init__(self, downsample_count=1, channels=8, classes=19):
        super().__init__()
        assert downsample_count > 0 and downsample_count < 5

        self.downsample_count = downsample_count
        self.classes = classes

        self.transform = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.down_stages = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(3 if i == 0 else 2 ** (i - 1) * channels, 2 ** i * channels, kernel_size=3, padding=1, bias=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(2 ** i * channels, 2 ** i * channels, kernel_size=3, padding=1, bias=False),
                    torch.nn.ReLU(inplace=True),
                )
                for i in range(downsample_count + 1)
            ]
        )

        self.up_stages = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(2 ** (downsample_count - i - 1) * channels * 2, 2 ** (downsample_count - i - 1) * channels, kernel_size=3, padding=1, bias=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(2 ** (downsample_count - i - 1) * channels, 2 ** (downsample_count - i - 1) * channels, kernel_size=3, padding=1, bias=False),
                    torch.nn.ReLU(inplace=True),
                )
                for i in range(downsample_count)
            ]
        )

        self.up_convs = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(2 ** (downsample_count - i) * channels, 2 ** (downsample_count - i - 1) * channels, kernel_size=3, padding=1, bias=False)
                for i in range(downsample_count)
            ]
        )

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(channels, classes, kernel_size=1, bias=False),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.transform(x)

        fmaps : List[Tensor] = []

        for i in range(self.downsample_count + 1):
            x = self.down_stages[i](x)
            if i != self.downsample_count:
                fmaps.append(x)
                x = self.maxpool(x)
        
        for i in range(self.downsample_count):
            x = self.upsample(x)
            x = self.up_convs[i](x)
            x = torch.cat([x, fmaps.pop()], axis=1)
            x = self.up_stages[i](x)
        
        x = self.head(x)

        return x
        
if __name__ == '__main__':

    model = UNet()

    in_t = torch.rand((1, 3, 1080, 1920))

    out_t = model(in_t)

    print(out_t.shape)

    torch.onnx.export(
        model,
        in_t,
        'model.onnx',
        opset_version=11,
        do_constant_folding=False,
    )
