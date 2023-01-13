import os
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
        self.comboBox_model.setCurrentIndex(workspace.get_model_index())
        self.init_my_model()

    def init_my_model(self):
        model_path = f"{self.workspace.workdir}/models"
        if os.path.exists(model_path):
            for model in os.listdir(model_path):
                self.listWidget.addItem(model.strip())

    @pyqtSlot(int)
    def on_comboBox_model_currentIndexChanged(self, index: int):
        dic = {
            0: "default",
            1: "UNet"
        }
        if index in dic.keys():
            md_path = f"./models/{dic[index]}/{dic[index]}.md"
        else:
            # TODO
            return
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding="UTF-8") as fp:
                md_content = fp.read()
        print(md_content)
        self.textBrowser_model_explaination.setMarkdown(md_content)
        self.workspace.set_model_index(index)
        self.workspace.save_project()
