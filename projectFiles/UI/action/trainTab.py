from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget,QFileDialog
from UI.static.trainTab import Ui_trainTab
from UI.threads.RunThread import RunThread
from workspace import Workspace


class trainTab(Ui_trainTab,QWidget):
    def __init__(self):
        print("3")
        super().__init__()
        self.setupUi(self)

    def set_workspace(self, workspace: Workspace):
        """
        设置工作区
        """
        self.workspace = workspace
    
    def set_threads(self,run_thread:RunThread):
        """
        设置线程
        """
        self.run_thread = run_thread

    @pyqtSlot()
    def on_pushButton_run_clicked(self):
        self.run_thread.set_workspace(self.workspace)
        self.run_thread.start()

    @pyqtSlot()
    def on_pushButton_save_clicked(self):
        file, _ = QFileDialog.getSaveFileName(self, "save model", filter="pth file(*.pth)")
        print(file)