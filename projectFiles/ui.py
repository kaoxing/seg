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
        self.tab.workspace_change_sig.connect(self.tab_3.reset_widget_label)
        self.tab_2.model_loaded_sig.connect(self.model_loaded)
        self.tab_3.change_tab_sig.connect(self.change_tab)

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
        os.mkdir(workdir + "/data/train")
        os.mkdir(workdir + "/data/train/image")
        os.mkdir(workdir + "/data/train/label")
        os.mkdir(workdir + "/data/test")
        os.mkdir(workdir + "/data/test/image")
        os.mkdir(workdir + "/data/test/label")
        os.mkdir(workdir + "/data/test/result")
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

    def model_loaded(self):
        model_name = self.workspace.get_pretrain_model()
        self.tab_3.lineEdit_pretrain_model.setText(model_name)
        self.tab_4.lineEdit_loaded_model.setText(model_name)

    def change_tab(self, page: int):
        self.tabWidget.setCurrentIndex(page)
