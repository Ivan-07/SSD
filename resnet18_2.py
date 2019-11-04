from .basic_module import BasicModule
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.layer(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)

class ResNet18(BasicModule):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        self.model_name = 'resnet18'

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # self.conv2 = self.make_layer(64, 64, 2, pad=1, stride=1)
        self.conv3 = self.make_layer(64, 128, 2)
        # 一层里第一个block的stride为2，第二个1
        self.conv4 = self.make_layer(128, 256, 2)
        self.conv5 = self.make_layer(256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def make_layer(self, inchannel, outchannel, block_num, stride=2):
        # shortcut应该是原输入，但因为channel不同，所以用kernel_size=1的卷积来改变channel数量
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers= []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            # 同一类block的outchannel相同，所以shortcut就是输入，因此不需要第一个block的shortcut
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    # def get_optimizer(self, lr=3e-3, momentum=0.9):
    #     return optim.Adam(self.parameters(), lr=lr  )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x
