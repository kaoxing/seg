import cv2 as cv
import numpy as np

# 导入os模块
import os

# path定义要获取的文件名称的目录
path = "mydataset/"
# os.listdir()方法获取文件夹名字，返回数组
file_name_list = os.listdir(path)

cnt = 0
# 1.导入图片
for image in file_name_list:
    if image[4:8] == 'mask':
        img_path = os.path.join(path, image)
        img_org = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        # 2.二值化处理
        ret, img_bin = cv.threshold(img_org, 30, 255, 0)
        # 3.细化处理
        img_thinning = cv.ximgproc.thinning(img_bin, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
        img_bin = img_bin - img_thinning
        cv.imwrite('mydataset/thin_result/{:0>3}_result.png'.format(cnt), img_bin, [cv.IMWRITE_PNG_COMPRESSION, 0])
        cnt = cnt+1

# print(np.count_nonzero(img_thinning))