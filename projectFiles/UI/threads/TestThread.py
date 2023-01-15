from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject
from workspace import Workspace
from ModelClass.modelTester import ModelTester

import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)


class TestThread(QThread, ModelTester):
    dice_sig = pyqtSignal(float)

    def __init__(self, parent=None):
        super(TestThread, self).__init__(parent)

    def __del__(self):
        # self.wait()
        pass

    def set_workspace(self, workspace: Workspace):
        self.test_folder = workspace.get_test_folder()
        self.model = workspace.get_model()
        self.settings = workspace.get_settings()

    def state_change(self):
        self.dice_sig.emit(self.test_dice)
        logging.info(f"dice sig emitted,loss:{self.self.test_dice}")
        return super().state_change()

    def run(self):
        logging.info("run thread is running")
        data_path = f"{self.test_folder}/image"
        mask_path = f"{self.test_folder}/label"
        result_path = f"{self.test_folder}/result"
        self.load_test_data(data_path, mask_path)
        self.set_model(self.model)
        self.test_model(result_path)
        return
