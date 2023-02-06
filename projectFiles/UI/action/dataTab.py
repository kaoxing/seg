from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog, QLineEdit
from UI.static.dataTab import Ui_dataTab
import dataOpe.dataOpeMethod as dom
import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)


class dataTab(Ui_dataTab, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    @pyqtSlot()
    def on_pushButton_standardize_clicked(self):
        """
        图片二值化
        """
        _path = self.lineEdit_standardize_folder.text()
        dom.standardize(_path)

    @pyqtSlot()
    def on_pushButton_expend_clicked(self):
        """
        数据扩充，图片选择
        """
        _spath = self.lineEdit_input_folder.text()
        _rpath = self.lineEdit_output_folder.text()
        dom.expend(_spath, _rpath)

    # 下面几个方法都是用来选择文件夹的(START)
    def select_folder(self, item: QLineEdit):
        temp = QFileDialog.getExistingDirectory()
        if temp != "":
            item.setText(temp)

    @pyqtSlot()
    def on_pushButton_standardize_browse_clicked(self):
        self.select_folder(self.lineEdit_standardize_folder)

    @pyqtSlot()
    def on_pushButton_input_browse_clicked(self):
        self.select_folder(self.lineEdit_input_folder)

    @pyqtSlot()
    def on_pushButton_output_browse_clicked(self):
        self.select_folder(self.lineEdit_output_folder)
    # 上面几个方法都是用来选择文件夹的(END)
