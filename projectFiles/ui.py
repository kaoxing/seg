import os
import sys
import datetime
from time import strftime

from PyQt5.QtWidgets import QMessageBox, QMainWindow, QSpinBox, QApplication, QFileSystemModel, QFileDialog, QTreeView
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

    def set_workspace(self, workspace: Workspace):
        self.image_folder = workspace.get_image_folder()
        self.result_folder = workspace.get_result_folder()
        self.model_index = workspace.get_model_index()

    def run(self):
        model = Model()
        if self.model_index == 1:
            localpath = "E:/大创/Seg/projectFiles"
            model.load_model(localpath+"/model/cnn_32.pt")
        model.load_predict_data(self.image_folder)
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
                self.workdir = self.workdir + "/workdir_" + \
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

    def set_tree_view(self,view:QTreeView,path:str):
        tree_model = QFileSystemModel()
        tree_model.setRootPath(path)
        view.setModel(tree_model)
        view.setRootIndex(tree_model.index(path))
        view.setColumnHidden(1, True)
        view.setColumnHidden(2, True)
        view.setColumnHidden(3, True)

    def select_image_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            self.lineEdit_image_folder.clear()
        else:
            self.lineEdit_image_folder.setText(temp)
            self.set_tree_view(self.treeView_image,temp)


    def select_label_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            self.lineEdit_label_folder.clear()
        else:
            self.lineEdit_label_folder.setText(temp)

    def create_project(self):
        """
        创建新项目
        """
        temp = [self.lineEdit_project_name.text(), self.lineEdit_image_folder.text()]
        if temp[0] != "" and temp[1] != "":
            # 把界面信息写入工作区，并保存项目
            self.workspace.set_project_name(temp[0])
            self.workspace.set_image_folder(temp[1])
            self.workspace.set_label_folder(self.lineEdit_label_folder.text())
            self.workspace.save_project()
            # 更新界面
            self.stackedWidget_state.setCurrentIndex(1)
            self.tabWidget.setCurrentIndex(1)
        else:
            print("请填写完整信息")
            # TODO 弹窗提醒

    def model_chosen(self, index: int):
        dic = {
            0: "请选择一个模型",
            1: "U-net的描述"
        }
        self.textEdit_model_explaination.setText(dic[index])
        self.workspace.set_model_index(index)

    def select_result_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            self.lineEdit_result_folder.clear()
            self.workspace.set_result_folder(None)
        else:
            self.lineEdit_result_folder.setText(temp)
            self.workspace.set_result_folder(temp)
            self.set_tree_view(self.treeView_evaluate,temp)

    def evaluate(self):
        evaluate_thread.set_workspace(self.workspace)
        evaluate_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    evaluate_thread = EvaluateThread()
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
