import os
import typing
from PyQt5.QtWidgets import QApplication, QVBoxLayout
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
import pyqtgraph as pg
import numpy as np
import cv2
import imghdr


class Image3dWidget(QWidget):
    def __init__(self, parent=None):
        super(Image3dWidget, self).__init__(parent)
        self.graphWidget = Image3d()
        # self.plotWidget = gl.GLViewWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.graphWidget)
        self.setLayout(self.layout)
        self.graphWidget.show()

    def load_images(self, file_path):
        self.graphWidget.loadGraph(file_path)

    def set_images(self, images):
        self.graphWidget.setImages(images)

    def show_images(self):
        self.graphWidget.showGraph()


class Image3d(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        pg.setConfigOptions(antialias=True, foreground='y', background='w')
        self.setCameraPosition(distance=800)
        self.images = np.ndarray(shape=(0, 512, 512, 4))
        self.thread = QThread()

    def loadGraph(self, file_path):
        imgType_list = {'jpg', 'bmp', 'png', 'jpeg', 'jfif'}
        image_list = os.listdir(file_path)

        for image in image_list:
            if imghdr.what(os.path.join(file_path, image)) in imgType_list:
                data_path = os.path.join(file_path, image)
                image = cv2.imread(data_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                image = cv2.resize(image, (512, 512))
                image = image[np.newaxis, :, :, :]
                self.images = np.append(self.images, image, 0)

    def setImages(self, images):
        self.images = images
        self.showGraph()

    def showGraph(self):
        self.clear()
        length = -int(self.images.shape[0] / 2)
        for i, image in enumerate(self.images):
            item = gl.GLImageItem(image, smooth=True)
            item.translate(-len(image[0]) / 2, -
                           len(image) / 2, i + length)  # 这是为了使模型居中
            self.addItem(item)
        self.images = np.ndarray(shape=(0, 512, 512, 4))


if __name__ == '__main__':
    app = QApplication([])
    main_window = Image3dWidget()
    main_window.show()
    main_window.load_images(
        "E:\\CodeField\\workdir_20230205_140223\\result\\20230206_213335")
    main_window.show_images()
    app.exec_()
