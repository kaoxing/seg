import imghdr
import os
import cv2
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np


class CreateNiiDataset(Dataset):
    def __init__(self, path_raw, path_label, train=True, max_size=0, remove_black=False, equalize=True):
        self.path_raw = path_raw
        self.path_label = path_label
        self.file_raw = os.listdir(path_raw)
        self.file_label = os.listdir(path_label)
        if len(self.file_label) != len(self.file_raw):
            raise ValueError("The number of labels is not equal to the number of raw")
        self.raw_tensor = []
        self.label_tensor = []
        for i in range(len(self.file_raw)):
            if max_size != 0 and i == max_size:
                break
            img1 = sitk.ReadImage(os.path.join(self.path_raw, self.file_raw[i]))
            img2 = sitk.ReadImage(os.path.join(self.path_label, self.file_label[i]))
            data1 = sitk.GetArrayFromImage(img1)
            data2 = sitk.GetArrayFromImage(img2)

            data1 = cv2.resize(data1, (384, 384))
            data2 = cv2.resize(data2, (384, 384))

            if train is True:
                data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
            if remove_black is True and np.max(data2) == 0:  # 当为训练集时除黑
                continue
            if equalize is True:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl1 = clahe.apply(img)
            data2 = data2 / 255
            data1 = data1[np.newaxis, :, :]
            # data1_tensor = torch.from_numpy(np.concatenate([data1, data1, data1], 1))
            data1 = torch.from_numpy(data1)
            data1_tensor = data1.type(torch.FloatTensor)

            data2 = data2[np.newaxis, :, :]
            # data2_tensor = torch.from_numpy(np.concatenate([data2, data2, data2], 1))
            data2 = torch.from_numpy(data2)
            data2_tensor = data2.type(torch.FloatTensor)
            self.raw_tensor.append(data1_tensor)
            self.label_tensor.append(data2_tensor)
        self.raw_tensor = torch.stack(self.raw_tensor)
        self.label_tensor = torch.stack(self.label_tensor)

    # def crop(self, image, size):
    #     shp = image.shape
    #     scl = [int((shp[0] - crop_size[0]) / 2), int((shp[1] - crop_size[1]) / 2)]
    #     image_crop = image[scl[0]:scl[0] + crop_size[0], scl[1]:scl[1] + crop_size[1]]
    #     return image_crop

    def __getitem__(self, item):
        return self.raw_tensor[item], self.label_tensor[item]

    def load_data(self):
        return self

    def __len__(self):
        if len(self.raw_tensor) < len(self.label_tensor):
            return len(self.raw_tensor)
        else:
            return len(self.label_tensor)
