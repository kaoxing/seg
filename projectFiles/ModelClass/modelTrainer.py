import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from ModelClass.myDataSetTra import MyDataSetTra
from abc import abstractmethod
from ModelClass.myModel import Model


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model: Model = None):
        self.model: Model = model
        self.train_dataset = None
        self.train_loss = 0

    def set_model(self, model):
        self.model = model

    def load_train_data(self, data_path, mask_path):
        """加载标签,参数（标签路径）"""
        self.train_dataset = MyDataSetTra(data_path, mask_path)

    def save_model(self, save_path):
        self.model.save_model(save_path)

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
            for i, data in enumerate(dataloader):
                input_data, labels = data
                optimizer.zero_grad()  # 梯度置零
                predict = self.model(input_data)  # 数据输入网络输出预测值
                loss = loss_func(predict, labels)  # 通过预测值与标签算出误差
                loss.backward()  # 误差逆传播
                optimizer.step()  # 通过梯度调整参数
                Loss += loss.item()
                print("loss.item():", loss.item())
            print("Loss:", Loss)
            self.train_loss = Loss
            self.state_change()

    @abstractmethod
    def state_change(self):
        pass
