from torch import nn
import torch


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = ConvolutionLayer(in_channels, 64)  # 三通道拓展至32通道
        self.down1 = DownSample()  # 下采样至1/2
        self.conv2 = ConvolutionLayer(64, 128)  # 32通道==>64通道
        self.down2 = DownSample()  # 下采样至1/4
        self.conv3 = ConvolutionLayer(128, 256)  # 64通道==>128通道
        self.down3 = DownSample()  # 下采样至1/8
        self.conv4 = ConvolutionLayer(256, 512)  # 128通道==>256通道
        self.down4 = DownSample()  # 下采样至1/16
        self.conv5 = ConvolutionLayer(512, 1024)  # 256通道==>512通道
        self.up1 = UpSample(1024)  # 上采样至1/8
        self.conv6 = ConvolutionLayer(1024, 512)  # 512通道==>256通道
        self.up2 = UpSample(512)  # 上采样至1/4
        self.conv7 = ConvolutionLayer(512, 256)  # 256通道==>128通道
        self.up3 = UpSample(256)  # 上采样至1/2
        self.conv8 = ConvolutionLayer(256, 128)  # 128通道==>64通道
        self.up4 = UpSample(128)  # 上采样至1/1
        self.conv9 = ConvolutionLayer(128, 64)  # 64通道==>32通道
        self.predict = nn.Sequential(  # 输出层，由sigmoid函数激活
            nn.Conv2d(64, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """下采样"""
        x1 = self.conv1(x)  # ===> 1/1 64
        d1 = self.down1(x1)  # ===> 1/2 64

        x2 = self.conv2(d1)  # ===> 1/2 128
        d2 = self.down2(x2)  # ===> 1/4 128

        x3 = self.conv3(d2)  # ===> 1/4 256
        d3 = self.down3(x3)  # ===> 1/8 256

        x4 = self.conv4(d3)  # ===> 1/8 512
        d4 = self.down4(x4)  # ===> 1/16 512

        x5 = self.conv5(d4)  # ===> 1/16 1024
        """上采样"""
        up1 = self.up1(x5)  # ===> 1/8 512

        x6 = self.conv6(torch.cat((x4, up1), dim=1))  # ===> 1/8 512
        up2 = self.up2(x6)  # ===> 1/4 256

        x7 = self.conv7(torch.cat((x3, up2), dim=1))  # ===> 1/4 256
        up3 = self.up3(x7)  # ===> 1/2 128

        x8 = self.conv8(torch.cat((x2, up3), dim=1))  # ===> 1/2 128
        up4 = self.up4(x8)  # ===> 1/1 64

        x9 = self.conv9(torch.cat((x1, up4), dim=1))  # ===> 1/1 64
        mask = self.predict(x9)  # ===> 1/1 out_channels
        return mask


class ConvolutionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        卷积层
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        """
        super(ConvolutionLayer, self).__init__()
        self.layer = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # BN层
            nn.ReLU(),  # 激活
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):

    def __init__(self, ):
        """
        最大池化层构成的下采样，池化窗口为2×2
        """
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):

    def __init__(self, in_channels):
        """
        反卷积，上采样，通道数将会减半，
        :param in_channels: 输入通道数
        """
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)
