from time import sleep
import numpy as np
import imghdr
import os
import cv2
from PyQt5.QtCore import pyqtSlot,QThread,QObject,pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog
from UI.static.modelingTab import Ui_ModelingTab
import logging


logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)

class ModelingTab(Ui_ModelingTab,QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.file_path = None
        self.images = np.ndarray(shape=(0, 512, 512,4))
        self.load_graph_thread = LoadGraphThread()
        self.load_graph_thread.sig.connect(self.widget_3d.set_images)
        self.load_graph_thread.sig.connect(self.set_images)

    @pyqtSlot()
    def on_pushButton_Browse_clicked(self):
        _temp = QFileDialog.getExistingDirectory()
        if _temp != "":
            self.lineEdit.setText(_temp)
            self.file_path = _temp

    @pyqtSlot()
    def on_pushButton_Modeling_clicked(self):
        if self.file_path is None:
            return
        self.pushButton_Browse.setEnabled(False)
        self.pushButton_Modeling.setEnabled(False)
        self.pushButton_Modeling.setText("Waiting...")
        self.load_graph_thread.set_file_path(self.file_path)
        self.load_graph_thread.start()

    def set_images(self,images):
        self.images = images
        self.spinBox_x.setMaximum(self.images.shape[1]-1)
        self.spinBox_y.setMaximum(self.images.shape[2]-1)
        self.spinBox_z.setMaximum(self.images.shape[0]-1)
        self.label_x.setText(str(self.images.shape[1]-1))
        self.label_y.setText(str(self.images.shape[2]-1))
        self.label_z.setText(str(self.images.shape[0]-1))
        self.pushButton_Modeling.setText("Modeling")
        self.pushButton_Browse.setEnabled(True)
        self.pushButton_Modeling.setEnabled(True)

    @pyqtSlot(int)
    def on_spinBox_x_valueChanged(self,value):
        self.widget_X.set_2darray(self.images[:,value,:])

    @pyqtSlot(int)
    def on_spinBox_y_valueChanged(self,value):
        self.widget_Y.set_2darray(self.images[:,:,value])

    @pyqtSlot(int)
    def on_spinBox_z_valueChanged(self,value):
        self.widget_Z.set_2darray(self.images[value,:,:])
    

class LoadGraphThread(QThread):
    sig = pyqtSignal(np.ndarray)
    def __init__(self, parent = None):
        super().__init__(parent)
        self.file_path = None

    def set_file_path(self,file_path):
        self.file_path = file_path

    def run(self):
        imgType_list = {'jpg', 'bmp', 'png', 'jpeg', 'jfif'}
        image_list = os.listdir(self.file_path)
        images = np.ndarray(shape=(0, 512, 512, 4))
        for image in image_list:
            if imghdr.what(os.path.join(self.file_path, image)) in imgType_list:
                data_path = os.path.join(self.file_path, image)
                image = cv2.imread(data_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                image = cv2.resize(image, (512, 512))
                image = image[np.newaxis, :, :, :]
                images = np.append(images, image, 0)
        self.sig.emit(images)