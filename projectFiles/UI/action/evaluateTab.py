from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog
from UI.static.evaluateTab import Ui_evaluateTab
from UI.threads.EvaluateThread import EvaluateThread
from workspace import Workspace
import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)


class evaluateTab(Ui_evaluateTab, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.lineEdit_loaded_model.setText("to be loaded")
        self.lineEdit_status.setText("waiting...")
        self.evaluate_thread = EvaluateThread()
        self.set_threads()
        self.widget_3d

    def set_workspace(self, workspace: Workspace):
        """
        设置工作区
        """
        self.workspace = workspace
        input_folder = workspace.get_evaluate_folder()
        self.lineEdit_evaluate_folder.setText(input_folder)
        self.widget_input.load_images(input_folder)

    def set_threads(self):
        """
        设置线程
        """
        self.evaluate_thread.img_sig.connect(self.widget_result.add_image)
        self.evaluate_thread.finished.connect(self.evaluate_finished)

    @pyqtSlot()
    def on_pushButton_evaluate_clicked(self):
        """
        选择输入文件夹
        """
        temp = QFileDialog.getExistingDirectory()
        if temp != "":
            self.lineEdit_evaluate_folder.setText(temp)
            self.widget_input.clear()
            self.widget_input.load_images(temp)
            self.workspace.set_evaluate_folder(temp)
            self.workspace.save_project()

    @pyqtSlot()
    def on_pushButton_start_clicked(self):
        """
        start evaluate按钮
        """
        self.lineEdit_status.setText("evaluating...")
        self.widget_result.clear()
        self.evaluate_thread.set_workspace(self.workspace)
        self.evaluate_thread.start()

    def evaluate_finished(self):
        """
        预测结束
        """
        self.lineEdit_status.setText("evaluate finished")
        logging.info("evaluate finished")
