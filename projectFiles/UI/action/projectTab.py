import os
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog, QListWidgetItem, QLineEdit
from UI.static.projectTab import Ui_projectTab
from workspace import Workspace


class projectTab(Ui_projectTab, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # sav = "./projectList/projectList.sav"
        # if os.path.exists(sav):
        #     with open(sav, 'r') as f:
        #         # 逐行读取文件内容
        #         for line in f:
        #             # 将该行内容添加到 QListWidget 组件中
        #             self.listWidget.addItem(line.strip())
        #     self.stackedWidget_state.setCurrentIndex(1)
        # else:
        #     pass
        #     # TODO 创建sav文件

    def set_workspace(self, workspace: Workspace):
        """
        设置工作区
        """
        self.workspace = workspace
        self.lineEdit_project_name.setText(workspace.get_project_name())
        self.lineEdit_image_folder.setText(workspace.get_image_folder())
        self.lineEdit_label_folder.setText(workspace.get_label_folder())
        self.lineEdit_test_folder.setText(workspace.get_test_folder())
        self.lineEdit_result_folder.setText(workspace.get_result_folder())

    # 下面几个方法都是用来选择文件夹的(START)
    def select_folder(self, item: QLineEdit):
        temp = QFileDialog.getExistingDirectory()
        if temp != "":
            item.setText(temp)

    @pyqtSlot()
    def on_pushButton_image_clicked(self):
        self.select_folder(self.lineEdit_image_folder)
        # TODO self.set_tree_view(self.treeView_image, temp)

    @pyqtSlot()
    def on_pushButton_label_clicked(self):
        self.select_folder(self.lineEdit_label_folder)

    @pyqtSlot()
    def on_pushButton_test_clicked(self):
        self.select_folder(self.lineEdit_test_folder)

    @pyqtSlot()
    def on_pushButton_result_clicked(self):
        self.select_folder(self.lineEdit_result_folder)
    # 上面几个方法都是用来选择文件夹的(END)

    @pyqtSlot()
    def on_pushButton_confirm_clicked(self):
        self.workspace.set_project_name(self.lineEdit_project_name.text())
        self.workspace.set_image_folder(self.lineEdit_image_folder.text())
        self.workspace.set_label_folder(self.lineEdit_label_folder.text())
        self.workspace.set_test_folder(self.lineEdit_test_folder.text())
        self.workspace.set_result_folder(self.lineEdit_result_folder.text())
        self.workspace.save_project()



    # @pyqtSlot()
    # def on_pushButton_create_clicked(self):
    #     """
    #     创建新项目
    #     """
    #     temp = [self.lineEdit_project_name.text(
    #     ), self.lineEdit_image_folder.text()]
    #     if temp[0] != "" and temp[1] != "":
    #         # 把界面信息写入工作区，并保存项目
    #         self.workspace.set_project_name(temp[0])
    #         self.workspace.set_image_folder(temp[1])
    #         self.workspace.set_label_folder(self.lineEdit_label_folder.text())
    #         self.workspace.save_project()
    #         # 更新界面
    #         self.stackedWidget_state.setCurrentIndex(1)
    #         # TODO self.tabWidget.setCurrentIndex(1)
    #     else:
    #         print("请填写完整信息")
    #         # TODO 弹窗提醒

    # @pyqtSlot()
    # def on_pushButton_test_clicked(self):
    #     temp = QFileDialog.getExistingDirectory()
    #     if temp == "":
    #         self.lineEdit_test_folder.clear()
    #         self.workspace.set_test_folder(None)
    #     else:
    #         self.lineEdit_test_folder.setText(temp)
    #         self.workspace.set_test_folder(temp)
    #         self.workspace.save_project()

    # @pyqtSlot(QListWidgetItem)
    # def on_listWidget_itemDoubleClicked(self, item: QListWidgetItem):
    #     if self.workspace.load_project(item.text()):
    #         # self.lineEdit_result_folder.setText(self.workspace.get_result_folder())
    #         self.lineEdit_project_name.setText(
    #             self.workspace.get_project_name())
    #         self.lineEdit_image_folder.setText(
    #             self.workspace.get_image_folder())
    #         self.lineEdit_label_folder.setText(
    #             self.workspace.get_label_folder())
    #         # self.comboBox_model.setCurrentIndex(self.workspace.get_model_index())
    #         # self.set_tree_view(self.treeView_image, self.workspace.get_image_folder())
    #         # self.set_tree_view(self.treeView_evaluate, self.workspace.get_result_folder())
    #         # TODO 改下被注释的句子
