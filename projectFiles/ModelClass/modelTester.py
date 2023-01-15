import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from myDataSetTra import MyDataSetTra
from abc import abstractmethod
from myModel import Model


class ModelTester:
    """模型测试器"""

    def __init__(self, model: Model):
        self.model: Model = model
        self.test_dataset = None
        self.test_dice = 0

    def load_test_data(self, data_path, mask_path):
        """加载测试集,参数（测试集数据）"""
        self.test_dataset = MyDataSetTra(data_path, mask_path)

    def test_model(self, result_path):
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # dices = np.array([])
        for i, data in enumerate(dataloader):
            input_data, labels = data
            input_data = input_data.to(device)
            predict = self.model(input_data)
            img_pre = torch.reshape(predict, (512, 512)).cpu().detach().numpy()
            img_lab = torch.reshape(labels, (512, 512)).detach().numpy()
            img_pre = np.round(img_pre)
            img_lab = np.round(img_lab)
            self.test_dice = self.__dice(img_pre, img_lab)
            img_pre = img_pre * 255
            im = Image.fromarray(img_pre)
            im = np.array(im, dtype='uint8')
            cv2.imwrite(os.path.join(result_path, "%03d.png" % i), im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.state_change()
            # print("dice:", self.test_dice)
        # print("average:", dices.sum() / len(dataloader))

    @staticmethod
    def __dice(predict, label):
        same = 0
        cnt = label.sum() + predict.sum()
        for x in range(predict.shape[0]):
            for y in range(predict.shape[1]):
                if predict[x][y] == 1 and label[x][y] == 1:
                    same += 1
        return same * 2 / cnt

    @abstractmethod
    def state_change(self):
        pass


if __name__ == '__main__':
    model = Model()
    model.load_model("./models/UNet/cnn_24.pth", "./models/UNet/UNet.py")
    model.load_predict_data("./data")
    model.run_model("./result")
