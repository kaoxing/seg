import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
from myDataSetPre import MyDataSetPre
from myDataSetTra import MyDataSetTra
from abc import abstractmethod
import importlib
import sys


class Model:
    """模型"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.predict_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loss = 0

    def load_model(self, model_path, net_path):
        """加载模型,参数（模型路径，网络路径）"""
        # self.models = MyModel(model_path)
        # models.load_state_dict(torch.load(PATH))
        # models.eval()
        # 动态导入模块
        sys.path.append(os.path.abspath(os.path.dirname(net_path)))
        net_name = os.path.basename(net_path)[:-3]
        metaclass = importlib.import_module(net_name)   # 获取模块实例
        Net = getattr(metaclass, net_name)  # 获取构造函数
        self.model: nn.Module = Net(1,1)
        self.model_path = model_path
        self.model.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def load_predict_data(self, data_path):
        """"加载数据,参数（数据路径）"""
        self.predict_dataset = MyDataSetPre(data_path)
        self.predict_dataset = DataLoader(
            dataset=self.predict_dataset,
            batch_size=1,
            shuffle=False,
        )

    def load_test_data(self, test_path):
        """加载测试集,参数（测试集数据）"""
        self.test_dataset = MyDataSetTra(test_path)
        self.test_dataset = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
        )

    def load_train_data(self, data_path, mask_path):
        """加载标签,参数（标签路径）"""
        self.train_dataset = MyDataSetTra(data_path, mask_path)

    def run_model(self, predict_result_path):
        """运行模型，参数(结果保存路径)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i, data in enumerate(self.predict_dataset):
            print(i)
            data = data.to(device)
            predict = self.model(data)
            predict = torch.reshape(predict, (512, 512))
            predict = predict.cpu()
            predict = predict.detach()
            predict = predict.numpy()
            predict = predict.round()
            predict = predict * 255
            img = np.array(predict, dtype='uint8')
            cv2.imwrite("{0}/{1:0>3d}.png".format(predict_result_path, i), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def train_model(self, epoch, batch_size, learning_rate=0.000001,
                    shuffle=True, optim="Adam", loss_func="BCELoss"):
        """训练模型,参数（训练轮数,训练批次大小,学习率,数据集是否打乱,优化器,），若新model名为空则将覆盖原model"""
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if loss_func == "BCELoss":
            loss_func = nn.BCELoss()
        elif loss_func == "CrossEntropyLoss":
            loss_func = nn.CrossEntropyLoss()
        elif loss_func == "MSELoss":
            loss_func = nn.MSELoss()
        elif loss_func == "NLLoss2d":
            loss_func = nn.NLLLoss2d()
        loss_func = loss_func.to(device)
        if optim == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optim == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optim == "RMSProp":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        for cnt in range(epoch):
            Loss = 0
            # print('第%d轮迭代' % cnt)
            for i, data in enumerate(dataloader):
                input_data, labels = data
                optimizer.zero_grad()  # 梯度置零
                predict = self.model(input_data)  # 数据输入网络输出预测值
                loss = loss_func(predict, labels)  # 通过预测值与标签算出误差
                loss.backward()  # 误差逆传播
                optimizer.step()  # 通过梯度调整参数
                Loss += loss
            # print(Loss.item())
            self.train_loss = Loss
            self.state_change()

    @abstractmethod
    def state_change(self):
        pass


if __name__ == '__main__':
    model = Model()
    model.load_model("./models/UNet/cnn_24.pth", "./models/UNet/UNet.py")
    model.load_predict_data("./data")
    model.run_model("./result")
