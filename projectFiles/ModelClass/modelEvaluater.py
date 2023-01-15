import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from myDataSetPre import MyDataSetPre
from myModel import Model


class ModelEvaluater:
    """模型评估器"""

    def __init__(self, model: Model):
        self.model: Model = model
        self.predict_dataset = None

    def load_predict_data(self, data_path):
        """"加载数据,参数（数据路径）"""
        self.predict_dataset = MyDataSetPre(data_path)
        self.predict_dataset = DataLoader(
            dataset=self.predict_dataset,
            batch_size=1,
            shuffle=False,
        )

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


