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

    def get_model(self):
        return self.model

    def load_model(self, net_path, has_net=False, model_path=None):
        """加载模型,参数（模型路径，网络路径）"""
        # self.models = MyModel(model_path)
        # models.load_state_dict(torch.load(PATH))
        # models.eval()
        # 动态导入模块
        # print(net_path)
        sys.path.append(os.path.abspath(os.path.dirname(net_path)))
        net_name = os.path.basename(net_path)[:-3]
        # print(net_name)
        metaclass = importlib.import_module(net_name)  # 获取模块实例
        # print(metaclass.__file__)
        Net = getattr(metaclass, net_name)  # 获取构造函数
        # print(net_name,metaclass,Net)
        self.model: nn.Module = Net()
        self.model_path = model_path
        if has_net:
            self.model.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)



