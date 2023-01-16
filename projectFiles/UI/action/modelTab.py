import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QMessageBox
from ModelClass.myModel import Model
from UI.static.modelTab import Ui_modelTab
from workspace import Workspace
import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)


class modelTab(Ui_modelTab, QWidget):
    model_loaded_sig = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.sys_model_path = "./models"
        self.my_model_path = None
        self.sys_model_list = []
        self.my_model_list = []

    def set_workspace(self, workspace: Workspace):
        """
        设置工作区
        """
        self.workspace = workspace
        self.my_model_path = f"{workspace.workdir}/models"
        self.checkBox.setChecked(workspace.get_from_my_model())
        try:
            self.comboBox_model.setCurrentIndex(workspace.get_model_index())
        except:
            logging.error("error")
        

    def init_model_list(self):
        """
        初始化模型列表
        """
        if os.path.exists(self.sys_model_path):
            self.sys_model_list = os.listdir(self.sys_model_path)
            logging.info(f"sys_model_list:\n{self.sys_model_list}")
        if os.path.exists(self.my_model_path):
            self.my_model_list = os.listdir(self.my_model_path)
            logging.info(f"my_model_list:\n{self.my_model_list}")

    @pyqtSlot(int)
    def on_checkBox_stateChanged(self, state: int):
        """
        勾选框，确定系统模型/我的模型
        """
        logging.info(f"on_checkBox_stateChanged:{state}")
        self.init_model_list()
        if state == 0:
            self.comboBox_model.clear()
            self.comboBox_model.addItems(self.sys_model_list)
            self.workspace.set_from_my_model(False)
        elif state == 2:
            self.comboBox_model.clear()
            self.comboBox_model.addItems(self.my_model_list)
            self.workspace.set_from_my_model(True)

    @pyqtSlot(int)
    def on_comboBox_model_currentIndexChanged(self, index: int):
        """
        下拉框，选择模型
        """
        logging.info(f"下拉框:{index}")
        self.workspace.set_model_index(index)
        text = self.comboBox_model.currentText()
        md_path = f"./models/{text}/{text}.md"
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding="UTF-8") as fp:
                md_content = fp.read()
            self.textBrowser_model_explaination.setMarkdown(md_content)

    @pyqtSlot()
    def on_pushButton_load_clicked(self):
        """
        按下load按钮，加载模型
        """
        # 处理路径
        model_name = self.comboBox_model.currentText()
        if self.checkBox.isChecked():
            dir_path = os.path.join(self.my_model_path, model_name)
        else:
            dir_path = os.path.join(self.sys_model_path, model_name)
        model_path = os.path.join(dir_path, f"{model_name}.pth")
        net_path = os.path.join(dir_path, f"{model_name}.py")
        # 检查
        if os.path.exists(model_path) and os.path.exists(net_path):
            # 把模型载入工作区
            model = Model()
            model.load_model(model_path, net_path)
            self.workspace.set_pretrain_model(model_name)
            self.workspace.set_model(model)
            self.workspace.save_project()
            # 发送信号
            self.model_loaded_sig.emit()
            QMessageBox.information(
                self,
                "load model",
                "model loaded successfully!"
            )
            logging.info(f"model loaded:{self.workspace.get_model()}")
        else:
            print(type(self))
            QMessageBox.critical(
                self,
                "load model",
                "model broken!"
            )
            logging.error("model broken")
