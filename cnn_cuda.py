import torch
import cv2
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from UNet import UNet


class MyDataSet(Dataset):
    def __init__(self, path):
        self.data_x = []
        self.data_y = []
        data_x = []
        data_y = []
        image_list = os.listdir(path)
        for image in image_list:
            if image[-8:-4] == 'mask':
                data_y_path = os.path.join(path, image)
                data = cv2.imread(data_y_path, cv2.IMREAD_GRAYSCALE)
                # data = cv2.resize(data, (512, 512))
                # temp = (data / 255)
                data = torch.Tensor(data / 255)  # 归一
                data = data.view(1, 512, 512)
                data_y.append(data)
                # print(data_y_path)
            elif image[-3:] == "png":
                data_x_path = os.path.join(path, image)
                data = cv2.imread(data_x_path, cv2.IMREAD_GRAYSCALE)
                # data = cv2.resize(data, (512, 512))
                # temp = (data/255)
                data = torch.Tensor(data / 255)  # 归一
                data = data.view(1, 512, 512)
                data_x.append(data)
                # print(data_x_path)
        self.data_x = torch.stack(data_x)
        self.data_y = torch.stack(data_y)
        # print(data_x)
        self.len = len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index, :, :], self.data_y[index, :, :]

    def __len__(self):
        return self.len


def train():
    batch_size = 1
    path = 'mydataset_expand'
    data_set = MyDataSet(path)
    dataloader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=True,
    )
    net = torch.load("projectFiles/model/cnn.pt", map_location='cuda' if torch.cuda.is_available() else 'cpu')
    # net = UNet(1, 1)
    net = net.to(device)
    loss_function = nn.BCELoss()
    loss_function = loss_function.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00000001)
    count = 50

    for cnt in range(count):
        Loss = 0
        print('第%d轮迭代' % cnt)
        for i, data in enumerate(dataloader):
            input_data, labels = data
            input_data = input_data.to(device)
            labels = labels.to(device)
            # cv2.imwrite("Test/input_data_%03d.png" % i,torch.reshape(input_data, (512, 512)).cpu().detach().numpy() * 255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
            # cv2.imwrite("Test/labels_%03d.png" % i,torch.reshape(labels, (512, 512)).cpu().detach().numpy() * 255,[cv2.IMWRITE_PNG_COMPRESSION, 0])
            optimizer.zero_grad()  # 梯度置零
            predict = net(input_data)  # 数据输入网络输出预测值
            loss = loss_function(predict, labels)  # 通过预测值与标签算出误差
            loss.backward()  # 误差逆传播
            optimizer.step()  # 通过梯度调整参数
            Loss += loss
        # os.system("nvidia-smi")
        print(Loss.item())
    return net


def test(net):
    path = 'Test/test'
    data_set = MyDataSet(path)
    dataloader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=False,
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
            dices = np.append(dices, temp)
            img_pre = img_pre * 255
            im = Image.fromarray(img_pre)
            im = np.array(im, dtype='uint8')
            cv2.imwrite("./result/%03d.png" % i, im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print("dice:", temp)
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
    # print(device)
    # net = train()
    net = torch.load('./projectFiles/model/cnn_24.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
    # torch.save(net, 'projectFiles/model/cnn24.pt')
    torch.save(net.state_dict(), './projectFiles/model/cnn_24.pth')
    # test(net)
    print("finished")
