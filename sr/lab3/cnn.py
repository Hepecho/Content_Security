from torch import nn
from torchsummary import summary


# VGG分类器
class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #  4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        #  对于torch.nn.Flatten()，因为其被用在神经网络中，
        # 输入为一批数据，第一维为batch，通常要把一个数据拉成一维，而不是将一
        # 批数据拉为一维。所以torch.nn.Flatten()默认从第二维开始平坦化。
        self.linear = nn.Linear(128*5*4, 10)  # 128:最后一个卷积层的输出，5：
        self.softmax = nn.Softmax(dim=1)  # 对linear层输出的结果进行归一化

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        prediction = self.softmax(logits)
        return prediction


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 64, 44))
    # 1:输入的通道数量，64是频率轴， 44是时间轴
