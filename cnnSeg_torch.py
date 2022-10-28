import torch
import cv2
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchsummary import summary
from unet_model import UNet

# https://blog.csdn.net/weixin_36411839/article/details/
# 105088883?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-5.
# pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.4&utm_relevant_index=8
# output(输出图像大小) = (input(输入图像大小) – 1) * stride + outputpadding(输出图像padding) – 2 * padding + kernelsize
# https://blog.csdn.net/weixin_40244676/article/details/117258128 Pytorch的padding值确定方法
# 一般来讲，根据卷积的原理，输入的大小和输出的大小之间的关系由如下公式表示：
# out_size=(input_size - kernerl_size + 2*padding)/stride +1
# 180705 Pytorch查看模型每层的输出形状https://blog.csdn.net/qq_33039859/article/details/80934060
# https://www.baidu.com/link?url=qE8bc5K09pwKK-4aeE6N0946rupj4RM6JneBchCqqSrIwFxVY0cZ5FOAgTwRu4mP2IiFdL04GZ-RKRBakjOJT_&wd=&eqid=9707214f0034ff1a0000000562754f40


class Mydataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.target = []
        data_y = []
        data_x = []
        image_dir = os.path.join(file_path)
        list_image = os.listdir(image_dir)
        for image in list_image:
            if image[4:8] == 'mask':
                data_y_path = os.path.join(image_dir, image)
                datay = cv2.imread(data_y_path, cv2.IMREAD_GRAYSCALE)
                datay = cv2.resize(datay, (128, 128))
                datay = torch.Tensor(datay/255)
                datay = datay.view(1, 128, 128)
                data_y.append(datay)
            else:
                data_x_path = os.path.join(image_dir, image)
                datax = cv2.imread(data_x_path, cv2.IMREAD_GRAYSCALE)
                datax = cv2.resize(datax, (128, 128))
                datax = torch.Tensor(datax/255)
                datax = datax.view(1, 128, 128)
                data_x.append(datax)
        self.data = torch.stack(data_x)
        self.target = torch.stack(data_y)
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.target[index, :, :, :]

    def __len__(self):
        return self.len


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=3, out_ch=2):
        super(U_Net, self).__init__()
        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out = nn.Sigmoid()(self.Conv(d2))
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = self.conv_sequential(1, 16, 3, 1, 2, 2)  # 1表示通道数，16个filter,每个小是3*3，padding为1，maxpooling
        # 的filter为2*2，stride=2
        self.layer2 = self.conv_sequential(16, 64, 3, 1, 2, 2)
        self.layer3 = self.conv_sequential(64, 128, 3, 1, 2, 2)
        self.layer4 = self.conv_sequential(128, 256, 3, 1, 2, 2)
        self.layer5 = self.conv_sequential(256, 512, 3, 1, 2, 2)
        self.transpose_layer2 = self.transpose_conv_sequential(1, 1, 4, 2, 1)
        self.transpose_layer8 = self.transpose_conv_sequential(1, 1, 16, 8, 4)
        self.ravel_layer32 = nn.Sequential(
            nn.Conv2d(512, 1, 1),  # 1个filter,每个大小是1*1
            nn.ReLU(True)
        )
        self.ravel_layer16 = nn.Sequential(
            nn.Conv2d(256, 1, 1),  # 1个filter,每个大小是1*1
            nn.ReLU(True)
        )
        self.ravel_layer8 = nn.Sequential(
            nn.Conv2d(128, 1, 1),  # 1个filter,每个大小是1*1
            nn.ReLU(True)
        )

    def forward(self, x):
        ret = self.layer1(x)  # 此时网络应该输出16个大小为64*64的特征图
        ret = self.layer2(ret)  # 此时网络应该输出64个大小为32*32的特征图
        ret = self.layer3(ret)  # 此时网络应该输出128个大小为16*16的特征图
        x8 = ret
        ret = self.layer4(ret)  # 此时网络应该输出256个大小为8*8的特征图
        x16 = ret
        ret = self.layer5(ret)  # 此时网络应该输出512个大小为4*4的特征图
        x32 = ret
        x32 = self.ravel_layer32(x32)  # 此时网络应该输出1个大小为4*4的特征图
        x16 = self.ravel_layer16(x16)  # 应该输出1个大小为8*8的特征图
        x8 = self.ravel_layer8(x8)  # 应该输出1个大小为16*16的特征图
        x32 = self.transpose_layer2(x32)  # 应该输出1个大小为8*8的特征图
        x16 = x16 + x32
        x16 = self.transpose_layer2(x16)  # 应该输出1个大小为16*16的特征图
        x8 = x8 + x16
        result = self.transpose_layer8(x8)  # 应该输出1个大小为128*128的特征图
        out = nn.Sigmoid()(result)
        return out

    def conv_sequential(self, in_size, out_size, kfilter, padding, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kfilter, padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size, stride)
        )

    def transpose_conv_sequential(self, in_size, out_size, kfilter, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kfilter, stride, padding, bias=False),
            nn.BatchNorm2d(out_size)
        )


def train():
    batch_size = 4
    trainfile_path = './data_list/train'
    traindataset = Mydataset(trainfile_path)
    train_dataloader = DataLoader(
        dataset=traindataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    # model = UNet(1, 2)
    model = Net()
    # 查看每层的输出大小
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model, input_size=(1, 512, 512))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.BCELoss()
    num_size = 150
    for num in range(num_size):
        Loss = 0
        print('----第%d轮迭代-----' % num)
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            prediction = model(inputs)
            loss = loss_function(prediction, labels)
            Loss += loss
            loss.backward()
            optimizer.step()
        print('损失为%f' % (Loss/100.0).item())
    return model, optimizer


def test(model):
    testfile_path = './data_list/test'
    testdataset = Mydataset(testfile_path)
    test_dataloaders = DataLoader(dataset=testdataset, batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i, data in enumerate(test_dataloaders):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            prediction = model(inputs)
            img_y = torch.reshape(prediction, (128, 128)).detach().cpu().numpy()  # 取出数值部分，原大小是1*1*128*128
            # 对应的二值图
            img = np.round(img_y)  # 预测标签
            img = img * 255  # *255
            im = Image.fromarray(img)  # numpy 转 image类
            im = np.array(im, dtype='uint8')
            Image.fromarray(im, 'L').save("result/%03d.png" % i)
            i = i + 1
            plt.pause(0.01)


if __name__ == '__main__':

    net, optimizer = train()
    # 模型保存
    torch.save(net, 'model/cnn.pth')

    model = torch.load("model/cnn.pth")
    test(model)
    print('done')
