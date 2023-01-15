from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog
from UI.static.trainTab import Ui_trainTab
from UI.threads.RunThread import RunThread
from workspace import Workspace
import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)

class trainTab(Ui_trainTab, QWidget):
    change_tab_sig = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.run_thread = None

    def set_workspace(self, workspace: Workspace):
        """
        设置工作区
        """
        self.workspace = workspace
        settings = workspace.get_settings()
        self.spinBox_epochs.setValue(settings["epochs"])
        self.spinBox_batch_size.setValue(settings["batch_size"])
        self.comboBox_loss_function.setCurrentText(settings["loss_function"])
        self.doubleSpinBox_lr.setValue(settings["lr"])
        self.comboBox_optimizer.setCurrentText(settings["optimizer"])
        self.lineEdit_pretrain_model.setText(settings["pretrain_model"])
        self.lineEdit_status.setText(settings["status"])

    def set_threads(self, run_thread: RunThread):
        """
        设置线程
        """
        self.run_thread = run_thread
        self.run_thread.loss_sig.connect(self.widget_loss.loss_plot)
        self.run_thread.finished.connect(self.train_finished)

    def update_settings(self):
        """
        更新设置
        """
        settings = {
            "epochs": self.spinBox_epochs.value(),
            "batch_size": self.spinBox_batch_size.value(),
            "loss_function": self.comboBox_loss_function.currentText(),
            "lr": self.doubleSpinBox_lr.value(),
            "optimizer": self.comboBox_optimizer.currentText(),
            "pretrain_model": self.lineEdit_pretrain_model.text(),
            "status": self.lineEdit_status.text(),
        }
        self.workspace.set_settings(settings)
        self.workspace.save_project()
        logging.info("setting updated")
        print(settings)


    @pyqtSlot()
    def on_pushButton_change_clicked(self):
        self.change_tab_sig.emit(1)

    @pyqtSlot()
    def on_pushButton_train_clicked(self):
        self.update_settings()
        self.lineEdit_status.setText("running...")
        self.widget_loss.reset_plot_item()
        self.run_thread = RunThread()
        self.run_thread.set_workspace(self.workspace)
        self.run_thread.start()

    def train_finished(self):
        logging.info("train finished")
        self.lineEdit_status.setText("train finished")

    @pyqtSlot()
    def on_pushButton_test_clicked(self):
        logging.info("pushButton_test clicked")

    @pyqtSlot()
    def on_pushButton_save_clicked(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "save model", filter="pth file(*.pth)")
        print(file)
