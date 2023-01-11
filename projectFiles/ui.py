import os
import sys
import datetime
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileSystemModel, QFileDialog, QTreeView
from UI.mainWindow import Ui_MainWindow
from workspace import Workspace


class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

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
                os.mkdir(self.workdir + "/models")
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

    def set_tree_view(self, view: QTreeView, path: str):
        tree_model = QFileSystemModel()
        tree_model.setRootPath(path)
        view.setModel(tree_model)
        view.setRootIndex(tree_model.index(path))
        view.setColumnHidden(1, True)
        view.setColumnHidden(2, True)
        view.setColumnHidden(3, True)

