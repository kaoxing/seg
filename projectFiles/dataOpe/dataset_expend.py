import cv2 as cv
import numpy as np
# 导入os模块
import os
path = "mydataset/"
file_name_list = os.listdir(path)
i = 0
j = 0
for image in file_name_list:
    if image[4:8] == 'mask':
        img_path = os.path.join(path, image)
        img_org = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        # x轴镜像翻转
        img_x = cv.flip(img_org, 0)
        # y轴镜像翻转
        img_y = cv.flip(img_org, 1)
        # 180度镜像翻转
        img_z = cv.flip(img_org, -1)
        cv.imwrite("mydataset_expand/{:0>3d}_0_mask.png".format(i), img_org, [cv.IMWRITE_PNG_COMPRESSION, 0])
        cv.imwrite("mydataset_expand/{:0>3d}_1_mask.png".format(i), img_x, [cv.IMWRITE_PNG_COMPRESSION, 0])
        cv.imwrite("mydataset_expand/{:0>3d}_2_mask.png".format(i), img_y, [cv.IMWRITE_PNG_COMPRESSION, 0])
        cv.imwrite("mydataset_expand/{:0>3d}_3_mask.png".format(i), img_z, [cv.IMWRITE_PNG_COMPRESSION, 0])
        i += 1
    elif image[-3:] == "png":
        img_path = os.path.join(path, image)
        img_org = cv.imread(img_path)
        # x轴镜像翻转
        img_x = cv.flip(img_org, 0)
        # y轴镜像翻转
        img_y = cv.flip(img_org, 1)
        # 180度镜像翻转
        img_z = cv.flip(img_org, -1)
        cv.imwrite("mydataset_expand/{:0>3d}_0.png".format(j), img_org, [cv.IMWRITE_PNG_COMPRESSION, 0])
        cv.imwrite("mydataset_expand/{:0>3d}_1.png".format(j), img_x, [cv.IMWRITE_PNG_COMPRESSION, 0])
        cv.imwrite("mydataset_expand/{:0>3d}_2.png".format(j), img_y, [cv.IMWRITE_PNG_COMPRESSION, 0])
        cv.imwrite("mydataset_expand/{:0>3d}_3.png".format(j), img_z, [cv.IMWRITE_PNG_COMPRESSION, 0])
        j += 1
