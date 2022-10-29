
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cov1 = self.block1(1, 16, 5, 2, 2, 2)  # 1*512*512->16*256*256
        self.cov2 = self.block1(16, 64, 3, 1, 2, 2)  # 16*256*256->64*128*128
        self.cov3 = self.block1(64, 256, 3, 1, 2, 2)  # 64*128*128->256*64*64
        self.cov4 = self.block1(256, 256, 1, 0, 1, 1)  # 256*64*64->256*64*64
        self.rcov1 = self.block2(256, 64, 2, 0, 2)  # 256*64*64->64*128*128
        self.rcov2 = self.block2(64, 16, 2, 0, 2)  # 64*128*128->16*256*256
        self.rcov3 = self.block2(16, 1, 2, 0, 2)  # 16*256*256->1*512*512

    def forward(self, x):
        ret = self.cov1(x)
        # x_4_256=ret.clone()
        ret = self.cov2(ret)
        # x_16_128=ret.clone()
        ret = self.cov3(ret)
        # x_32_64=ret.clone()
        ret = self.cov4(ret)
        ret = self.rcov1(ret)
        # ret=ret+x_16_128
        ret = self.rcov2(ret)
        # ret=ret+x_4_256
        ret = self.rcov3(ret)
        out = nn.Sigmoid()(ret)
        return out

    def block1(self, in_size, out_size, kfilter, padding, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kfilter, padding=padding),
            nn.MaxPool2d(kernel_size, stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True)
        )

    def block2(self, in_size, out_size, kfilter, padding, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kfilter, stride, padding, bias=False),
            nn.BatchNorm2d(out_size)
        )

