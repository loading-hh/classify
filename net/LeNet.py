import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, pretrain, **kwargs):
        super().__init__(**kwargs)
        ###pytorch官方没有LeNet的预训练模型
        if pretrain == True:
            self.net = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 2, stride = 1),
                                     nn.Sigmoid (),
                                     nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                     nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, padding = 0, stride = 1),
                                     nn.Sigmoid (),
                                     nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                     nn.Flatten(),
                                     nn.Linear(16 * 5 * 5, 120), nn.Sigmoid (),
                                     nn.Linear(120, 84), nn.Sigmoid (),
                                     nn.Linear(84, 10))
        else:
            self.net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1),
                                     nn.Sigmoid(),
                                     nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
                                     nn.Sigmoid(),
                                     nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                     nn.Flatten(),
                                     nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                                     nn.Linear(120, 84), nn.Sigmoid(),
                                     nn.Linear(84, 10))

    def forward(self, x):
        y = self.net(x)
        return y
if __name__ == "__main__":
    net = Net(True)
    print(net)
