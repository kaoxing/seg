import imghdr
import cv2 as cv
import numpy as np
import os


def standardize(path: str):
    """对path路径下所有图片进行二值化处理"""
    file_name_list = os.listdir(path)
    imgType_list = {'jpg', 'bmp', 'png', 'jpeg', 'jfif'}
    for image in file_name_list:
        if imghdr.what(path + image) in imgType_list:
            img_path = os.path.join(path, image)
            img_org = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            # 二值化处理
            ret, img_bin = cv.threshold(img_org, 1, 255, 0)
            cv.imwrite(img_path, img_bin, [cv.IMWRITE_PNG_COMPRESSION, 0])


def expend(spath: str, rpath: str):
    """对spath路径下的图片进行旋转扩展，并存入rpath中"""
    file_name_list = os.listdir(spath)
    i = 0
    n = len(str(len(file_name_list)))  # 获取文件个数位数
    imgType_list = {'jpg', 'bmp', 'png', 'jpeg', 'jfif'}
    for image in file_name_list:
        img_path = os.path.join(spath, image)
        img_type = imghdr.what(img_path)
        if img_type in imgType_list:
            img_org = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            # x轴镜像翻转
            img_x = cv.flip(img_org, 0)
            # y轴镜像翻转
            img_y = cv.flip(img_org, 1)
            # 180度镜像翻转
            img_z = cv.flip(img_org, -1)
            cv.imwrite(os.path.join(rpath, "{i:0>{n}d}".format(i=i, n=n) + img_type), img_org,
                       [cv.IMWRITE_PNG_COMPRESSION, 0])
            cv.imwrite(os.path.join(rpath, "{i:0>{n}d}".format(i=i + 1, n=n) + img_type), img_org,
                       [cv.IMWRITE_PNG_COMPRESSION, 0])
            cv.imwrite(os.path.join(rpath, "{i:0>{n}d}".format(i=i + 2, n=n) + img_type), img_org,
                       [cv.IMWRITE_PNG_COMPRESSION, 0])
            cv.imwrite(os.path.join(rpath, "{i:0>{n}d}".format(i=i + 3, n=n) + img_type), img_org,
                       [cv.IMWRITE_PNG_COMPRESSION, 0])
            i += 4
