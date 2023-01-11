from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget
from UI.static.modelTab import Ui_modelTab
from workspace import Workspace


class modelTab(Ui_modelTab, QWidget):
    def __init__(self):
        print("2")
        super().__init__()
        self.setupUi(self)

    def set_workspace(self, workspace: Workspace):
        """
        设置工作区
        """
        self.workspace = workspace

    @pyqtSlot(int)
    def on_comboBox_model_currentIndexChanged(self, index: int):
        dic = {
            0: "请选择一个模型",
            1: "U-net的描述"
        }
        self.textEdit_model_explaination.setText(dic[index])
        self.workspace.set_model_index(index)
        self.workspace.save_project()
