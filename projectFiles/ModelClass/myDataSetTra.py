import imghdr
import os
import cv2
import torch
from torch.utils.data import Dataset


class MyDataSetTra(Dataset):
    def __init__(self, data_path, mask_path):
        # print(data_path)
        self.data_x = []
        self.data_y = []
        data_x = []
        data_y = []
        mask_list = os.listdir(mask_path)
        imgType_list = {'jpg', 'bmp', 'png', 'jpeg', 'jfif'}
        for image in mask_list:
            # print(image)
            if imghdr.what(mask_path + image) in imgType_list:
                data_y_path = os.path.join(mask_path, image)
                data = cv2.imread(data_y_path, cv2.IMREAD_GRAYSCALE)
                data = torch.Tensor(data / 255)  # 归一
                data = data.view(1, 512, 512)
                data_y.append(data)
                # print(data_y_path)
        raw_list = os.listdir(data_path)
        for image in raw_list:
            # print(image)
            if imghdr.what(data_path + image) in imgType_list:
                data_x_path = os.path.join(data_path, image)
                data = cv2.imread(data_x_path, cv2.IMREAD_GRAYSCALE)
                data = torch.Tensor(data / 255)  # 归一
                data = data.view(1, 512, 512)
                data_x.append(data)
                # print(data_x_path)
        self.data_x = torch.stack(data_x)
        self.data_y = torch.stack(data_y)
        # print(data_x)
        self.len = len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index, :, :], self.data_y[index, :, :]

    def __len__(self):
        return self.len
