import os
import sys
import datetime
from time import strftime

from PyQt5.QtWidgets import QMessageBox, QMainWindow, QSpinBox, QApplication, QFileSystemModel, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from UI.mainWindow import Ui_MainWindow
from workspace import Workspace
from myModel import Model


class EvaluateThread(QThread):
    # sig = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(EvaluateThread, self).__init__(parent)

    def __del__(self):
        self.wait()

    def set_workspace(self, workspace:Workspace):
        self.image_folder = workspace.get_image_folder()
        self.result_folder = workspace.get_result_folder()
        self.model_index = workspace.get_model_index()

    def run(self):
        model = Model()
        if self.model_index == 1:
            model.load_model("model/cnn_24.pt")
        model.load_data(self.image_folder)
        model.run_model(self.result_folder)
        # self.sig.emit(True)

class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.workspace = Workspace()
        # self.workdir = None
        # self.treeView.hide()
        # self.widget.hide()

    def q_action(self, q):
        """
        菜单栏事件
        @para:
        q：事件
        """
        return
        if q == self.action_open_workdir or q == self.action_new_workdir:
            # 打开/新建工作区
            temp = QFileDialog.getExistingDirectory()
            if temp == "":
                return
            else:
                self.workdir = temp
            if q == self.action_new_workdir:
                # 新建工作区
                self.workdir = self.workdir + "/workdir_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # 初始化工作区
                os.mkdir(self.workdir)
                os.mkdir(self.workdir + "/data")
                # os.mkdir(self.workdir + "/data/train")
                # os.mkdir(self.workdir + "/data/test")
                os.mkdir(self.workdir + "/model")
                os.mkdir(self.workdir + "/result")
            # 更新工作区视图
            print(self.workdir)
            tree_model = QFileSystemModel()
            tree_model.setRootPath(self.workdir)
            self.treeView.setModel(tree_model)
            self.treeView.setRootIndex(tree_model.index(self.workdir))
            self.treeView.setColumnHidden(1, True)
            self.treeView.setColumnHidden(2, True)
            self.treeView.setColumnHidden(3, True)
            self.treeView.show()
            self.widget.show()

        elif q == self.action_close_workdir:
            # 关闭工作区
            self.workdir = None
            self.treeView.hide()
            self.widget.hide()

    def select_image_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            return
        else:
            self.lineEdit_new_image_folder.setText(temp)
            self.workspace.set_image_folder(temp)

    def select_label_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            return
        else:
            self.lineEdit_new_label_folder.setText(temp)
            self.workspace.set_label_folder(temp)

    def create_project(self):
        """
        创建新项目
        """
        temp = self.lineEdit_new_project_name.text()
        if temp != "" and self.workspace.check():
            self.workspace.set_project_name(temp)
            self.lineEdit_project_name.setText(self.workspace.get_project_name())
            self.lineEdit_image_folder.setText(self.workspace.get_image_folder())
            self.lineEdit_label_folder.setText(self.workspace.get_label_folder())
            self.stackedWidget.setCurrentIndex(1)
            self.workspace.save_project()
        else:
            print("请填写完整信息")
            # TODO 弹窗提醒

    def modify_project_name(self):
        temp = self.lineEdit_project_name.text()
        if temp != "":
            self.workspace.set_project_name(temp)

    def modify_image_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            return
        else:
            self.lineEdit_image_folder.setText(temp)
            self.workspace.set_image_folder(temp)

    def modify_label_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            return
        else:
            self.lineEdit_label_folder.setText(temp)
            self.workspace.set_label_folder(temp)

    def model_chosen(self,index:int):
        if index == 0:
            self.textEdit_model_explaination.setText("请选择一个模型")
        elif index == 1:
            self.textEdit_model_explaination.setText("U-net的描述")
        self.workspace.set_model_index(index)

    def select_result_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            return
        else:
            self.lineEdit_result_folder.setText(temp)
            self.workspace.set_result_folder(temp)


    def evaluate(self):
        evaluate_thread.set_workspace(self.workspace)
        evaluate_thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    evaluate_thread = EvaluateThread()
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
