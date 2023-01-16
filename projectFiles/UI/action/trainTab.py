import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog
from UI.static.trainTab import Ui_trainTab
from UI.threads.RunThread import RunThread
from UI.threads.TestThread import TestThread
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
        self.run_thread = RunThread()
        self.test_thread = TestThread()
        self.set_threads()

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
        self.lineEdit_pretrain_model.setText(workspace.get_pretrain_model())
        self.lineEdit_status.setText(workspace.get_status())
        self.reset_widget_label()

    def reset_widget_label(self):
        self.widget_label.load_images(os.path.join(self.workspace.get_test_folder(),"label"))

    def set_threads(self):
        """
        设置线程
        """
        self.run_thread.loss_sig.connect(self.widget_loss.loss_plot)
        self.run_thread.finished.connect(self.train_finished)
        self.test_thread.dice_sig.connect(self.widget_dice.dice_plot)
        self.test_thread.img_sig.connect(self.widget_result.add_image)
        self.test_thread.finished.connect(self.test_finished)

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
        """
        训练按钮
        """
        self.update_settings()
        self.lineEdit_status.setText("trainning...")
        self.widget_loss.reset_plot_item()
        self.run_thread.set_workspace(self.workspace)
        self.run_thread.start()

    def train_finished(self):
        logging.info("train finished")
        self.lineEdit_status.setText("train finished")

    @pyqtSlot()
    def on_pushButton_test_clicked(self):
        """
        测试按钮
        """
        self.lineEdit_status.setText("testing...")
        self.widget_dice.reset_plot_item()
        self.widget_result.clear()
        self.test_thread.set_workspace(self.workspace)
        self.test_thread.start()

    def test_finished(self):
        logging.info("test finished")
        self.lineEdit_status.setText("test finished")

    @pyqtSlot()
    def on_pushButton_save_clicked(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "save model", filter="pth file(*.pth)")
        if os.path.exists(file):
            self.workspace.get_model().save_model(file)
