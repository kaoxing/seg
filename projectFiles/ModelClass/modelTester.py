import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from ModelClass.imgDataSetTra import MyDataSetTra
from abc import abstractmethod
from ModelClass.myModel import Model
from ModelClass.NewNiiDataset import CreateNiiDataset


class ModelTester:
    """模型测试器"""

    def __init__(self):
        self.model: Model = None
        self.test_dataset = None
        self.test_dice = 0
        self.filename = None

    def set_model(self, model):
        self.model = model.get_model()

    def load_test_data(self, data_path, mask_path, data_type="img", max_size=0, remove_black=False):
        """加载测试集,参数（测试集数据）"""
        if data_type == "img":
            self.test_dataset = MyDataSetTra(data_path, mask_path)
        elif data_type == "nii":
            self.test_dataset = CreateNiiDataset(data_path, mask_path, False, max_size,remove_black=remove_black)

    def test_model(self, result_path, num_workers=0):
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # dices = np.array([])
        self.model = self.model.to(device)
        Dice = 0
        imgavg = 0
        imgcnt = 0
        blackavg = 0
        blackcnt = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input_data, labels = data
                input_data = input_data.to(device)
                predict = self.model(input_data)
                img_pre = torch.reshape(predict, (384, 384)).cpu().detach().numpy()
                img_lab = torch.reshape(labels, (384, 384)).detach().numpy()
                # imgcnt += np.sum(img_pre)
                # if np.max(img_lab) == 0:
                #     temp = img_pre >= 0.5
                #     blackavg += np.sum(img_pre*temp)
                #     blackcnt += np.sum(img_pre >= 0.5)
                # else:
                #     temp = img_pre >= 0.5
                #     imgavg += np.sum(img_pre*temp)
                #     imgcnt += np.sum(img_pre >= 0.5)
                img_pre = (img_pre >= 0.5)*1
                self.test_dice = self.__dice(img_pre, img_lab)
                img_pre = img_pre * 255
                im = Image.fromarray(np.uint8(img_pre))
                im = np.array(im, dtype='uint8')
                predictname = os.path.join(result_path, "%04d.png" % i)
                cv2.imwrite(predictname, im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                labelname = os.path.join(result_path, "%04d_label.png" % i)
                img_lab = img_lab * 255
                im = Image.fromarray(np.uint8(img_lab))
                im = np.array(im, dtype="uint8")
                cv2.imwrite(labelname, im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print("dice:", self.test_dice)
                Dice += self.test_dice
            print("average:", Dice / len(dataloader))
            # print(blackavg / blackcnt)
            # print(imgavg / imgcnt)

    @staticmethod
    def __dice(predict, label):
        smooth = 1.  # 用于防止分母为0
        label = label.flatten()
        predict = predict.flatten()
        And = np.sum(label * predict)
        return (2 * And + smooth) / (np.sum(label) + np.sum(predict) + smooth)

    @abstractmethod
    def state_change(self):
        pass
