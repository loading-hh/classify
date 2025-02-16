import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self, pretrain, **kwargs):
        super().__init__(**kwargs)
        if type == 11:
            self.conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        ###pytorch官方没有NiN的预训练模型
        if pretrain == True:
            self.net = nn.Sequential(self.nin_block(1, 96, 11, 4, 0),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2),
                                     self.nin_block(96, 256, 5, 1, 2),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2),
                                     self.nin_block(256, 384, 3, 1, 1),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2), nn.Dropout(0.5),
                                     self.nin_block(384, 10, 3, 1, 1),
                                     nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten())
        else:
            self.net = nn.Sequential(self.nin_block(1, 96, 11, 4, 0),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2),
                                     self.nin_block(96, 256, 5, 1, 2),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2),
                                     self.nin_block(256, 384, 3, 1, 1),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2), nn.Dropout(0.5),
                                     self.nin_block(384, 10, 3, 1, 1),
                                     nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten())

    def forward(self, x):
        y = self.net(x)
        return y

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        net = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=1, stride=1, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=1, stride=1, padding=0),
                            nn.ReLU())
        return net

if __name__ == "__main__":
    net = Net(False)
    print(net)
