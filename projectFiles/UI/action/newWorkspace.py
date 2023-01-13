from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget
from UI.static.newWorkspace import Ui_newWorkspace
from workspace import Workspace
import datetime
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileSystemModel, QFileDialog, QTreeView, QAction



class newWorkspace(Ui_newWorkspace, QWidget):
    sig = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def show(self):
        super().show()
        default_text = "workdir_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lineEdit.setText(default_text)

    @pyqtSlot()
    def on_pushButton_clicked(self):
        temp = QFileDialog.getExistingDirectory()
        if temp == "":
            return
        else:
            name = self.lineEdit.text()
            if len(name)>0:
                workdir = f"{temp}/{name}"
                print(workdir)
                self.sig.emit(workdir)
                self.hide()
