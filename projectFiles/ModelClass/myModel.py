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

    def load_model(self, model_path, net_path, *args):
        """
        加载模型,参数（模型.pth路径，网络.py路径，模型初始化所需参数）
        若因文件与类名不相同返回false
        调用示例如下
        net = Model()
        net.load_model(".\\UNet.pth", ".\\UNet.py", 1, 1)
        """
        # self.models = MyModel(model_path)
        # models.load_state_dict(torch.load(PATH))
        # 动态导入模块
        sys.path.append(os.path.abspath(os.path.dirname(net_path)))
        net_name = os.path.basename(net_path)[:-3]
        metaclass = importlib.import_module(net_name)  # 获取模块实例
        Net = getattr(metaclass, net_name)  # 获取构造函数
        if Net is None:
            return False
        # 确实找不到别的好办法来解决这个参数的问题，这里只好采用eval
        self.model: nn.Module = eval(f"Net{args}")
        self.model_path = model_path
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        return True

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


# if __name__ == '__main__':
    # metaclass = importlib.import_module("NetModel")
    # Net = getattr(metaclass, "A")
    # print(net)
    # net = Model()
    # net.load_model(".\\UNet.pth", ".\\UNet.py", 1, 1)
