import os
import cv2
import torch
from torch.utils.data import Dataset


class MyDataSetPre(Dataset):
    def __init__(self, path):
        self.data_x = []
        data = []
        image_list = os.listdir(path)
        for image in image_list:
            if image[-3:] == "png":
                data_path = os.path.join(path, image)
                img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
                img = torch.Tensor(img / 255)  # 归一
                img = img.view(1, 512, 512)
                data.append(img)
        self.data_x = torch.stack(data)
        self.len = len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index, :, :]

    def __len__(self):
        return self.len
