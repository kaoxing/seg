import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from myDataSet import MyDataSet


class Model:
    """模型"""

    def __init__(self):
        self.model = None
        self.myDataSet = None

    def load_model(self, load_path):
        """加载模型,参数（模型路径）"""
        self.model = torch.load(load_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self, data_path):
        """"加载数据,参数（数据路径）"""
        self.myDataSet = MyDataSet(data_path)
        self.myDataSet = DataLoader(
            dataset=self.myDataSet,
            batch_size=1,
            shuffle=False,
        )

    def run_model(self, result_path):
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
            cv2.imwrite("{0}/{1:0>3d}.png".format(result_path, i), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    model = Model()
    model.load_model("model/cnn_24.pt")
    model.load_data("data")
    model.run_model("result")
