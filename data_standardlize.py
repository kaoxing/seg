import cv2 as cv
import numpy as np

# 导入os模块
import os

path = "mydataset/"
file_name_list = os.listdir(path)
for image in file_name_list:
    if image[4:8] == 'mask':
        img_path = os.path.join(path, image)
        img_org = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        # 2.二值化处理
        ret, img_bin = cv.threshold(img_org, 1, 255, 0)
        cv.imwrite(img_path, img_bin, [cv.IMWRITE_PNG_COMPRESSION, 0])
