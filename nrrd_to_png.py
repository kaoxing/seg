import numpy as np
import nrrd
import os
import cv2


def nrrd_to_jpg(nrrd_filename, save_path, patient_id):
    nrrd_filename = nrrd_filename
    nrrd_data, nrrd_options = nrrd.read(nrrd_filename)
    h, w, slides_num = nrrd_data.shape
    print(nrrd_data)
    print(nrrd_options)
    for i in range(slides_num):
        img = nrrd_data[:, :, slides_num - i - 1] * 255
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        cv2.imwrite(save_path + '/' + patient_id + '_' + str(i + 1) + '.png', img)


def rotation(path):
    image = cv2.imread(path)
    # image = cv2.transpose(image)
    cv2.imwrite(path, image)


if __name__ == '__main__':
    nrrd_to_jpg("D:\\slicer_save\\CTACardio.nrrd", "D:\\machine_learning\\cnn\\centerline_data", "05")
    # path = "D:/machine_learning/cnn/centerline_data"
    # image_list = os.listdir(path)
    # for image in image_list:
    #     string = path + "/" + image
    #     rotation(string)
    # rotation("D:/machine_learning/cnn/centerline_data/05_1.jpg")
