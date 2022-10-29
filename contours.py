import os
import cv2
import numpy as np

# path定义要获取的文件名称的目录
path = "mydataset/"
# os.listdir()方法获取文件夹名字，返回数组
file_name_list = os.listdir(path)
count = 0
for image in file_name_list:
    if image[4:8] == 'mask':
        groundtruth_path = os.path.join(path, image)
        groundtruth = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)
        h1, w1 = groundtruth.shape
        contours, cnt = cv2.findContours(groundtruth.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros([h1, w1], dtype=groundtruth.dtype)
        for contour in contours:
            M = cv2.moments(contour)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            cv2.circle(groundtruth, (center_x, center_y), 0, 0, -1)
            cv2.circle(image, (center_x, center_y), 0, 255, -1)
        cv2.imwrite('mydataset/thin/{:0>3}_result.png'.format(count), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite('mydataset/thin_result/{:0>3}_result.png'.format(count), groundtruth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count = count + 1
