import cv2 as cv
import numpy as np

# 图像显示



def pixGray(img, max):  # 直方图统计
    h = img.shape[0]
    w = img.shape[1]

    gray_level = np.zeros(max)
    gray_level2 = np.zeros(max)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            gray_level[img[i, j]] += 1  # 统计灰度级为img_gray[i,j]的个数

    for i in range(1, max):
        gray_level2[i] = gray_level2[i - 1] + gray_level[i]  # 统计灰度级小于img_gray[i,j]的个数
    return gray_level2


def imgEqualize(img, max):  # 直方图均衡化
    h, w = img.shape
    gray_level2 = pixGray(img)
    lut = np.zeros(max)
    for i in range(max):
        lut[i] = (max-1) / (h * w) * gray_level2[i]  # 得到新的灰度级
    lut = np.uint8(lut + 0.5)
    out = cv.LUT(img, lut)
    print('图像直方图均衡化 处理完毕')
    return out


if __name__ == "__main__":
    pic = cv.imread('hua.jpg', 0)
    title = 'mulan'
    picture = np.hstack((pic, imgEqualize(pic)))
    picShow(title, picture)

