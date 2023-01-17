import os
from PyQt5.QtWidgets import QApplication, QVBoxLayout
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QWidget
import pyqtgraph as pg
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

    def show_images(self):
        self.graphWidget.showGraph()


class Image3d(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        pg.setConfigOptions(antialias=True,foreground='y')
        self.setCameraPosition(distance=800)
        self.images = []

    def loadGraph(self, file_path):
        imgType_list = {'jpg', 'bmp', 'png', 'jpeg', 'jfif'}
        image_list = os.listdir(file_path)
        for image in image_list:
            if imghdr.what(os.path.join(file_path, image)) in imgType_list:
                data_path = os.path.join(file_path, image)
                image = cv2.imread(data_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                self.images.append(image)

    def showGraph(self):
        self.clear()
        length = -int(len(self.images) / 2)
        for i, image in enumerate(self.images):
            item = gl.GLImageItem(image, smooth=True)
            item.translate(-len(image[0])/2, -len(image)/2, i+length)  # 这是为了使模型居中
            self.addItem(item)
        self.images.clear()

if __name__ == '__main__':
    app = QApplication([])
    main_window = Image3dWidget()
    main_window.show()
    main_window.load_images("D:\\BigProject\\Seg\\Seg\\mydataset")
    main_window.show_images()
    app.exec_()

