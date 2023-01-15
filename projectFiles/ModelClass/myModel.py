import os
import torch
import torch.nn as nn
import importlib
import sys


class Model:
    """模型"""

    def __init__(self):
        self.model = None
        self.model_path = None

    def load_model(self, model_path, net_path):
        """加载模型,参数（模型路径，网络路径）"""
        # self.models = MyModel(model_path)
        # models.load_state_dict(torch.load(PATH))
        # models.eval()
        # 动态导入模块
        sys.path.append(os.path.abspath(os.path.dirname(net_path)))
        net_name = os.path.basename(net_path)[:-3]
        metaclass = importlib.import_module(net_name)  # 获取模块实例
        Net = getattr(metaclass, net_name)  # 获取构造函数
        self.model: nn.Module = Net(1, 1)
        self.model_path = model_path
        self.model.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


if __name__ == '__main__':
    model = Model()
    model.load_model("./models/UNet/cnn_24.pth", "./models/UNet/UNet.py")
    model.load_predict_data("./data")
    model.run_model("./result")
