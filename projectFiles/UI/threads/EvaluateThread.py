import datetime
import os
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from workspace import Workspace
from ModelClass.myModel import Model
from ModelClass.modelEvaluater import ModelEvaluater
import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)

class EvaluateThread(QThread, ModelEvaluater):
    img_sig = pyqtSignal(str)

    def __init__(self, parent=None):
        super(EvaluateThread, self).__init__(parent)

    def __del__(self):
        # self.wait()
        pass

    def set_workspace(self, workspace: Workspace):
        self.evaluate_folder = workspace.get_evaluate_folder()
        result_folder = workspace.get_result_folder()
        sub_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_folder = os.path.join(result_folder,sub_folder)
        os.mkdir(self.result_folder)
        self.model = workspace.get_model()

    def state_change(self):
        self.img_sig.emit(self.filename)
        return super().state_change()

    def run(self):
        self.set_model(self.model)
        self.load_predict_data(self.evaluate_folder)
        self.run_model(self.result_folder)
        return