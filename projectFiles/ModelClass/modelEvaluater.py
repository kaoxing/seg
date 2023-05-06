import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from ModelClass.imgDataSetPre import MyDataSetPre
from ModelClass.myModel import Model
from abc import abstractmethod


class ModelEvaluater:
    """模型评估器"""

    def __init__(self):
        self.model: Model = None
        self.predict_dataset = None
        self.filename = None

    def set_model(self, model: Model):
        self.model = model.get_model()

    def load_predict_data(self, data_path):
        """"加载数据,参数（数据路径）"""
        num_workers = torch.cuda.device_count() * 4 + 2
        self.predict_dataset = MyDataSetPre(data_path)
        self.predict_dataset = DataLoader(
            dataset=self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )

    def run_model(self, predict_result_path):
        """运行模型，参数(结果保存路径)"""
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for i, data in enumerate(self.predict_dataset):
                data = data.to(device)
                predict = self.model(data)
                predict = torch.reshape(predict, (512, 512))
                predict = predict.cpu()
                predict = predict.detach()
                predict = predict.numpy()
                predict = predict.round()
                predict = predict * 255
                filename = "{0}/{1:0>3d}.png".format(predict_result_path, i)
                img = np.array(predict, dtype='uint8')
                cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                self.filename = filename
                self.state_change()

    @abstractmethod
    def state_change(self):
        pass
