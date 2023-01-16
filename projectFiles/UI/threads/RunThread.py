from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject
from workspace import Workspace
from ModelClass.modelTrainer import ModelTrainer

import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)


class RunThread(QThread, ModelTrainer):
    loss_sig = pyqtSignal(float)

    def __init__(self, parent=None):
        super(RunThread, self).__init__(parent)
        self.epoch = 0

    def __del__(self):
        # self.wait()
        pass

    def set_workspace(self, workspace: Workspace):
        self.train_folder = workspace.get_train_folder()
        self.model = workspace.get_model()
        self.settings = workspace.get_settings()

    def state_change(self):
        self.loss_sig.emit(self.train_loss)
        logging.info(f"loss sig emitted,loss:{self.train_loss}")
        return super().state_change()

    def run(self):
        logging.info("train thread is running")
        data_path = f"{self.train_folder}/image"
        mask_path = f"{self.train_folder}/label"
        self.load_train_data(data_path, mask_path)
        self.set_model(self.model)
        self.train_model(
            epoch=self.settings["epochs"],
            batch_size=self.settings["batch_size"],
            learning_rate=self.settings["lr"],
            shuffle=True,
            optim=self.settings["optimizer"],
            loss_func=self.settings["loss_function"],
        )
        return
