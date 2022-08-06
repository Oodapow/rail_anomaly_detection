import torch
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class UNet(torch.nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.transform = torchvision.transforms.Normalize(MEAN, STD)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        self.unpool1 = torch.nn.Upsample(scale_factor=2)
        
        self.upblock1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )

        self.seg_head =  torch.nn.Conv2d(32, num_classes, kernel_size=1, bias=False)
        self.rec_head =  torch.nn.Conv2d(32, 3, kernel_size=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.transform(x)

        x1 = self.block1(x)
        
        x = self.pool1(x1)
        x = self.block2(x)
        
        x = self.unpool1(x)
        x = torch.cat([x, x1], axis=1)
        x = self.upblock1(x)
        
        x_seg = self.seg_head(x)
        x_rec = self.rec_head(x)

        x_rec = self.sigmoid(x_rec)

        return x_seg, x_rec

class UNetTeacher(torch.nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.transform = torchvision.transforms.Normalize(MEAN, STD)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )

        self.unpool1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, kernel_size=1, bias=False),
        )
        self.upblock1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        
        self.unpool2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, 32, kernel_size=1, bias=False),
        )
        self.upblock2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
        )

        self.seg_head =  torch.nn.Conv2d(32, num_classes, kernel_size=1, bias=False)
        self.rec_head =  torch.nn.Conv2d(32, 3, kernel_size=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.transform(x)

        x2 = self.block1(x)
        x = self.pool1(x2)

        x1 = self.block2(x)
        x = self.pool2(x1)

        x = self.block3(x)

        x = self.unpool1(x)
        x = torch.cat([x, x1], axis=1)
        x = self.upblock1(x)

        x = self.unpool2(x)
        x = torch.cat([x, x2], axis=1)
        x = self.upblock2(x)
        
        x_seg = self.seg_head(x)
        x_rec = self.rec_head(x)

        x_rec = self.sigmoid(x_rec)

        return x_seg, x_rec
        
if __name__ == '__main__':

    model = UNet()

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
