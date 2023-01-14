from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog
from UI.static.evaluateTab import Ui_evaluateTab
from UI.threads.EvaluateThread import EvaluateThread
from workspace import Workspace


class evaluateTab(Ui_evaluateTab, QWidget):
    def __init__(self):
        print("4")
        super().__init__()
        self.setupUi(self)

    def set_workspace(self, workspace: Workspace):
        """
        设置工作区
        """
        self.workspace = workspace
        self.lineEdit_evaluate_folder.setText(workspace.get_evaluate_folder())
    
    def set_threads(self,evaluate_thread:EvaluateThread):
        """
        设置线程
        """
        self.evaluate_thread = evaluate_thread

    @pyqtSlot()
    def on_pushButton_evaluate_clicked(self):
        temp = QFileDialog.getExistingDirectory()
        if temp != "":
            self.lineEdit_evaluate_folder.setText(temp)
            self.workspace.set_evaluate_folder(temp)
            self.workspace.save_project()

    @pyqtSlot()
    def on_pushButton_start_clicked(self):
        self.evaluate_thread.set_workspace(self.workspace)
        self.evaluate_thread.start()
