import os
import sys
import datetime
from time import strftime

from PyQt5.QtWidgets import QMessageBox, QMainWindow, QSpinBox, QApplication, QFileSystemModel, QFileDialog, QTreeView,QListWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal,pyqtSlot
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
            model.load_model("./models/cnn_24.pt","./net/Unet.py")
        model.load_predict_data(self.image_folder)
        model.run_model(self.result_folder)
        # self.sig.emit(True)

class RunThread(QThread):
    # sig = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(RunThread, self).__init__(parent)

    def __del__(self):
        self.wait()

    def set_workspace(self, workspace: Workspace):
        self.image_folder = workspace.get_image_folder()
        self.label_folder = workspace.get_label_folder()
        self.test_folder = workspace.get_test_folder()
        self.model_index = workspace.get_model_index()

    def run(self):
        model = Model()
        if self.model_index == 1:
            model.load_model("./model/cnn_24.pt","./net/Unet.py")
        model.load_train_data()
        # model.load_predict_data(self.image_folder)
        # model.run_model(self.result_folder)
        pass

class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.workspace = Workspace()
        sav = "./projectList/projectList.sav"
        if os.path.exists(sav):
            with open(sav, 'r') as f:
            # 逐行读取文件内容
                for line in f:
                    # 将该行内容添加到 QListWidget 组件中
                    self.listWidget.addItem(line.strip())
            self.stackedWidget_state.setCurrentIndex(1)

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
        self.workspace.save_project()

    def select_result_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            self.lineEdit_result_folder.clear()
            self.workspace.set_result_folder(None)
        else:
            self.lineEdit_result_folder.setText(temp)
            self.workspace.set_result_folder(temp)
            self.workspace.save_project()
            self.set_tree_view(self.treeView_evaluate,temp)


    def select_test_folder(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            self.lineEdit_test_folder.clear()
            self.workspace.set_test_folder(None)
        else:
            self.lineEdit_test_folder.setText(temp)
            self.workspace.set_test_folder(temp)
            self.workspace.save_project()

    def evaluate(self):
        evaluate_thread.set_workspace(self.workspace)
        evaluate_thread.start()

    @pyqtSlot()
    def on_pushButton_run_clicked(self):
        run_thread.set_workspace(self.workspace)
        run_thread.start()
    
    @pyqtSlot()
    def on_pushButton_save_clicked(self):
        file, _ = QFileDialog.getSaveFileName(self,"save model",filter="pth file(*.pth)")
        print(file)

    def modify_project(self):
        pass # TODO

    def load_recent_project(self,item:QListWidgetItem):
        if self.workspace.load_project(item.text()):
            self.lineEdit_result_folder.setText(self.workspace.get_result_folder())
            self.lineEdit_project_name.setText(self.workspace.get_project_name())
            self.lineEdit_image_folder.setText(self.workspace.get_image_folder())
            self.lineEdit_label_folder.setText(self.workspace.get_label_folder())
            self.comboBox_model.setCurrentIndex(self.workspace.get_model_index())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    evaluate_thread = EvaluateThread()
    run_thread = RunThread()
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
