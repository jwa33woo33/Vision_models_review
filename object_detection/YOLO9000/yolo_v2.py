import torch
import torch.nn as nn

class YoLo_v2(nn.Module):
    def __init__(self, backbone, num_classes=20, 
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)],):
        super(YoLo_v2, self).__init__()
        self.anchors = anchors

        self.num_classes=num_classes

        self.backbone = backbone[:43] # Exclude fc layer
        self.resconv = backbone[43:]
        
        self.regression = nn.Sequential(
            # Skip Connection 1024 + 2048
            nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=(len(anchors) * (5 + num_classes)), kernel_size=1, stride=1, padding=0),
            # Activation = Linear
        )
            
        self.init_weight(self.regression)

    def forward(self, i):
        r = self.backbone(i)
        batch_size, num_channels, height, width = r.shape
        o = self.resconv(r)
        r = r.view(-1, num_channels*4, height//2, width//2).contiguous()
        x = torch.cat((r, o), 1)
        x = self.regression(x)
        return x

    def init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)