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
            self.net = torchvision.models.alexnet(weights = torchvision.models.AlexNet_Weights.DEFAULT)
        else:
            self.net = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 11, padding = 1, stride = 4), nn.ReLU (),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),
                                     nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, padding = 2, stride = 1), nn.ReLU (),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),
                                     nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1, stride = 1), nn.ReLU (),
                                     nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1, stride = 1), nn.ReLU (),
                                     nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1, stride = 1), nn.ReLU (),
                                     nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),
                                     nn.Flatten(),
                                     nn.Linear(6400, 4096), nn.ReLU (),
                                     nn.Dropout(p = 0.5),
                                     nn.Linear(4096, 4096), nn.ReLU (),
                                     nn.Dropout(p = 0.5),
                                     nn.Linear(4096, 10))

    def forward(self, x):
        y = self.net(x)
        return y

if __name__ == "__main__":
    net = Net(False)
    print(net)
