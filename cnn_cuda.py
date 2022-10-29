import torch
import cv2
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchsummary import summary


class MyDataSet(Dataset):
    def __init__(self, path):
        self.data_x = []
        self.data_y = []
        data_x = []
        data_y = []
        image_list = os.listdir(path)
        for image in image_list:
            if image[4:8] == 'mask':
                data_y_path = os.path.join(path, image)
                data = cv2.imread(data_y_path, cv2.IMREAD_GRAYSCALE)
                # data = cv2.resize(data, (512, 512))
                data = torch.Tensor(data / 255)  # 归一
                data = data.view(1, 512, 512)
                data_y.append(data)
            elif image[4:] == "png":
                data_x_path = os.path.join(path, image)
                data = cv2.imread(data_x_path, cv2.IMREAD_GRAYSCALE)
                # data = cv2.resize(data, (512, 512))
                data = torch.Tensor(data / 255)  # 归一
                data = data.view(1, 512, 512)
                data_x.append(data)
        self.data_x = torch.stack(data_x)
        self.data_y = torch.stack(data_y)
        # print(data_x)
        self.len = len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index, :, :], self.data_y[index, :, :]

    def __len__(self):
        return self.len


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


def train():
    batch_size = 8
    path = 'mydataset'
    data_set = MyDataSet(path)
    dataloader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    net = Net()
    net = net.to(device)
    summary(net, input_size=(1, 512, 512))
    loss_function = nn.MSELoss()
    loss_function = loss_function.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    count = 100

    for cnt in range(count):
        Loss = 0
        print('第%d轮迭代' % cnt)
        for i, data in enumerate(dataloader):
            input_data, labels = data
            input_data = input_data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # 梯度置零
            predict = net(input_data)  # 数据输入网络输出预测值
            # labels=labels.view(5,1,512,512)
            loss = loss_function(predict, labels)  # 通过预测值与标签算出误差
            loss.backward()  # 误差逆传播
            optimizer.step()  # 通过梯度调整参数
            Loss += loss
        # os.system("nvidia-smi")
        print(Loss.item())
    return net


def test(net):
    path = 'mydataset_test'
    data_set = MyDataSet(path)
    dataloader = DataLoader(
        dataset=data_set,
        batch_size=1
    )
    with torch.no_grad():
        dices = np.array([])
        for i, data in enumerate(dataloader):
            input_data, labels = data
            input_data = input_data.to(device)
            predict = net(input_data)
            img_pre = torch.reshape(predict, (512, 512)).cpu().detach().numpy()
            img_lab = torch.reshape(labels, (512, 512)).detach().numpy()
            img_pre = np.round(img_pre)
            img_lab = np.round(img_lab)
            temp = dice(img_pre, img_lab)
            # print(temp)
            dices = np.append(dices,temp)
            img_pre = img_pre * 255
            im = Image.fromarray(img_pre)
            im = np.array(im, dtype='uint8')
            Image.fromarray(im, 'L').save("./result/%03d.png" % i)
            print("average:", dices.sum() / len(dataloader))


def dice(predict, label):
    same = 0
    all = label.sum() + predict.sum()
    for x in range(predict.shape[0]):
        for y in range(predict.shape[1]):
            if predict[x][y] == 1 and label[x][y] == 1:
                same += 1
    return same * 2 / all


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = train()
    torch.save(net, 'model/cnn.pt')
    test(net)
    print("finished")
