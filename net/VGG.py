import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self, pretrain, type = 11, **kwargs):
        super().__init__(**kwargs)
        if type == 11:
            self.conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        ###pytorch官方没有LeNet的预训练模型
        if pretrain == True:
            if type == 11:
                self.net = torchvision.models.vgg11(weights = torchvision.models.VGG11_Weights.DEFAULT)
        else:
            if type == 11:
                self.conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
                self.net = self.vgg(self.conv_arch)

    def forward(self, x):
        y = self.net(x)
        return y

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=3, padding=1, stride=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        return nn.Sequential(*layers)

    def vgg(self, conv_arch):
        conv_blks = []
        in_channels = 1
        for num_convs, out_channels in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(*conv_blks, nn.Flatten(),
                             # 全连接层部分
                             nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(4096, 4096), nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(4096, 10))

if __name__ == "__main__":
    net = Net(False)
    print(net)
