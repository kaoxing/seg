import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from ModelClass.myDataSetTra import MyDataSetTra
from abc import abstractmethod
from ModelClass.myModel import Model


class ModelTrainer:
    """模型训练器"""

    def __init__(self):
        self.model: Model = None
        self.train_dataset = None
        self.train_loss = 0
        self.flag = True

    def set_model(self, model):
        self.model = model.get_model()

    def load_train_data(self, data_path, mask_path):
        """加载标签,参数（标签路径）"""
        self.train_dataset = MyDataSetTra(data_path, mask_path)

    def train_model(self, epoch, batch_size, learning_rate=0.000001,
                    shuffle=True, optim="Adam", loss_func="BCELoss",
                    num_workers=-1, multiple_gpu=False, pin_memory=True):
        """
        训练模型,参数（训练轮数,训练批次大小,学习率,数据集是否打乱,优化器,数据装载线程数,多GPU模式,内存优化）
        若新model名为空则将覆盖原model
        数据装载线程数表示装载tensor数据的线程数，若为默认值-1则会自动安排一个较为合理的数量
        多GPU模式可能会导致模型精度变差
        """
        if num_workers == -1:
            num_workers = torch.cuda.device_count() * 4 + 2
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        if torch.cuda.device_count() > 1 and multiple_gpu:
            print("On", torch.cuda.device_count(), "GPU")
            self.model = nn.DataParallel(self.model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
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
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate)
        elif optim == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate)
        elif optim == "RMSProp":
            optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=learning_rate)
        for cnt in range(epoch):
            if not self.flag:
                break
            Loss = 0
            for i, data in enumerate(dataloader):
                input_data, labels = data
                input_data = input_data.to(device)
                labels = labels.to(device)
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
