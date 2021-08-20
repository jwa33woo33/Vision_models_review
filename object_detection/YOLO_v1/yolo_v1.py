import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from darknet import DarkNet
from util_layers import flatten_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YOLOv1(nn.Module):
    def __init__(self, features, num_bboxes=2, num_classes=20, bn=True):
        super().__init__()

        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.features = features # Get DarkNet feature
        self.conv_layers = self._make_conv_layers(bn)
        self.fc_layers = self._make_fc_layers()


    def _make_conv_layers(self, bn):
        if bn:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
            )
    
        else:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
            )

        return net

    def _make_fc_layers(self):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        net = nn.Sequential(
            flatten_layer(), # x.view(x.size(0), -1)
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

        return net

    def forward(self, x):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        x = self.features(x) # This feature is directly from DarkNet
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        #  S(그리드) = 7, B(후보박스갯수) = 2, C(클래스갯수) = 20
        x = x.view(-1, S, S, 5 * B + C) # output(x.shape): torch.Size([1, 7, 7, 30])

        return x

def test():
    # Build model with randomly initialized weights
    darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
    yolo = YOLOv1(darknet.features).to(device)

    # Prepare a dummy image to input (YOLO_v1의 input image 사이즈는 448x448)
    image = torch.rand(1, 3, 448, 448)
    
    # Forward
    output = yolo(image.to(device))
    # Check output tensor size, which should be [10, 7, 7, 30]
    # print(summary(yolo, (3,448,448)))
    print(output.size())


if __name__ == '__main__':
    test()
