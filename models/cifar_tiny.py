import torch.nn as nn
import torch.nn.functional as F

class Cifar_Tiny(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar_Tiny, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def get_features(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))

        return out
