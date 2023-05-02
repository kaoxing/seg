import imghdr
import os
import cv2
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np


class CreateNiiDataset(Dataset):
    def __init__(self, path_raw, path_label):
        self.path_raw = path_raw
        self.path_label = path_label
        lines = os.listdir(path_raw)
        lines.sort()
        self.file_raw = []
        for line in lines:
            self.file_raw.append(line)
        self.file_label = []
        lines = os.listdir(path_label)
        for line in lines:
            self.file_label.append(line)
        if len(self.file_label) != len(self.file_raw):
            raise ValueError("The number of labels is not equal to the number of raw")

    # def crop(self, image, size):
    #     shp = image.shape
    #     scl = [int((shp[0] - crop_size[0]) / 2), int((shp[1] - crop_size[1]) / 2)]
    #     image_crop = image[scl[0]:scl[0] + crop_size[0], scl[1]:scl[1] + crop_size[1]]
    #     return image_crop

    def __getitem__(self, item):
        img1 = sitk.ReadImage(os.path.join(self.path_raw, self.file_raw))
        img2 = sitk.ReadImage(os.path.join(self.path_label, self.file_label))
        data1 = sitk.GetArrayFromImage(img1)
        data2 = sitk.GetArrayFromImage(img2)

        # if data1.shape[0] >= 256:
        #     data1 = self.crop(data1, [256, 256])
        #     data2 = self.crop(data2, [256, 256])
        # if self.transform is not None:
        #     data1 = self.transform(data1)
        #     data2 = self.transform(data2)
        data1 = cv2.resize(data1, (512, 512))
        data2 = cv2.resize(data2, (512, 512))

        if np.min(data1) < 0:
            data1 = data1 - np.min(data1)
            # data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))

        if np.min(data2) < 0:
            data2 = data2 - np.min(data2)
            # data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))

        data1 = data1[np.newaxis, np.newaxis, :, :]
        data1_tensor = torch.from_numpy(np.concatenate([data1, data1, data1], 1))
        data1_tensor = data1_tensor.type(torch.FloatTensor)

        data2 = data2[np.newaxis, np.newaxis, :, :]
        data2_tensor = torch.from_numpy(np.concatenate([data2, data2, data2], 1))
        data2_tensor = data2_tensor.type(torch.FloatTensor)
        return data1_tensor, data2_tensor

    def load_data(self):
        return self

    def __len__(self):
        if len(self.file_raw) < len(self.file_label):
            return len(self.file_raw)
        else:
            return len(self.file_label)


if __name__ == '__main__':
    data = CreateNiiDataset()