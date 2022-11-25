import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
from myDataSetPre import MyDataSetPre
from myDataSetTra import MyDataSetTra

class Model:
    """模型"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.predict_dataset = None
        self.train_dataset = None

    def load_model(self, load_path):
        """加载模型,参数（模型路径）"""
        self.model_path = load_path
        self.model = torch.load(load_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    def load_predict_data(self, data_path):
        """"加载数据,参数（数据路径）"""
        self.predict_dataset = MyDataSetPre(data_path)
        self.predict_dataset = DataLoader(
            dataset=self.myDataSet,
            batch_size=1,
            shuffle=False,
        )

    def load_train_data(self, data_path, mask_path):
        """加载标签,参数（标签路径）"""
        self.predict_dataset = MyDataSetTra(data_path, mask_path)

    def run_model(self, predict_result_path):
        """运行模型，参数(结果保存路径)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i, data in enumerate(self.myDataSet):
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

    def train(self, train_result_path, epoch, batch_size, learning_rate=0.00000001,
              shuffle=True, optim="Adam", model_name=None):
        """训练模型,参数（训练结果,训练轮数,训练批次大小,学习率,数据集是否打乱,model保存名），若新model名为空则将覆盖原model"""
        dataloader = DataLoader(
            dataset=self.predict_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_function = nn.BCELoss()
        loss_function = loss_function.to(device)
        optimizer = None
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
                loss = loss_function(predict, labels)  # 通过预测值与标签算出误差
                loss.backward()  # 误差逆传播
                optimizer.step()  # 通过梯度调整参数
                Loss += loss
            # print(Loss.item())
        if model_name is None:
            torch.save(self.model, self.model_path)
        else:

            torch.save(self.model, self.model_path)


if __name__ == '__main__':
    model = Model()
    model.load_model("model/cnn_24.pt")
    model.load_data("mydataset_test")
    model.run_model("result")
