import sys

import cv2
from torch.utils.data import DataLoader

myData = cv2.imread("mydataset/000.png", cv2.IMREAD_GRAYSCALE)
data = cv2.imread("data/train/000.png", cv2.IMREAD_GRAYSCALE)

print(myData)
print(data)

from untitled import Ui_Dialog
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog


class Class(Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Class, self).__init__(parent)
        self.setupUi(self)

    def accept(self) -> None:
        print("accept")


if __name__ == '__main__':
    # app = QtWidgets.QApplication(sys.argv)
    # abc = Class()
    # abc.show()
    # sys.exit(app.exec_())
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    batch_size
    from cnn_cuda import MyDataSet
    path = 'mydataset_test'
    data_set = MyDataSet(path)
    dataloader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=True,
    )
