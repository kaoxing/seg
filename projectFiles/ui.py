import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileSystemModel, QFileDialog, QTreeView, QAction
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from UI.mainWindow import Ui_MainWindow
from workspace import Workspace


class MainWindow(Ui_MainWindow, QMainWindow):
    new_workspace_sig = pyqtSignal()  # 新建工作区

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.workspace = None
        self.tabWidget.hide()

    def init_workspace(self, workdir: str):
        """
        初始化工作区
        """
        if os.path.exists(workdir):
            self.workspace = Workspace(workdir)
            self.tab.set_workspace(self.workspace)
            self.tab_2.set_workspace(self.workspace)
            self.tab_3.set_workspace(self.workspace)
            self.tab_4.set_workspace(self.workspace)
            self.tabWidget.show()

    def new_workspace(self, workdir: str):
        os.mkdir(workdir)
        os.mkdir(workdir + "/data")
        os.mkdir(workdir + "/data/image")
        os.mkdir(workdir + "/data/label")
        os.mkdir(workdir + "/data/test")
        os.mkdir(workdir + "/models")
        os.mkdir(workdir + "/result")
        self.init_workspace(workdir)

    @pyqtSlot(QAction)
    def on_menubar_triggered(self, q_action: QAction):
        """
        菜单栏事件
        """
        if q_action == self.action_open_workdir:
            # 打开工作区
            workdir = QFileDialog.getExistingDirectory()
            self.init_workspace(workdir)
            return
            # 更新工作区视图
            tree_model = QFileSystemModel()
            tree_model.setRootPath(workdir)
            self.treeView.setModel(tree_model)
            self.treeView.setRootIndex(tree_model.index(workdir))
            self.treeView.setColumnHidden(1, True)
            self.treeView.setColumnHidden(2, True)
            self.treeView.setColumnHidden(3, True)
            self.treeView.show()
            self.widget.show()
        elif q_action == self.action_new_workdir:
            # 新建工作区
            self.new_workspace_sig.emit()
        elif q_action == self.action_close_workdir:
            # 关闭工作区
            self.workspace = None
            self.tabWidget.hide()


    def set_tree_view(self, view: QTreeView, path: str):
        tree_model = QFileSystemModel()
        tree_model.setRootPath(path)
        view.setModel(tree_model)
        view.setRootIndex(tree_model.index(path))
        view.setColumnHidden(1, True)
        view.setColumnHidden(2, True)
        view.setColumnHidden(3, True)
