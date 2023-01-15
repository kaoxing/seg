from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from workspace import Workspace
from ModelClass.myModel import Model
from ModelClass.modelEvaluater import ModelEvaluater


class EvaluateThread(QThread, ModelEvaluater):
    # sig = pyqtSignal(bool)

    def __init__(self, model: Model, parent=None):
        super(EvaluateThread, self).__init__(parent)
        super(ModelEvaluater, self).__init__(model)

    def __del__(self):
        # self.wait()
        pass

    def set_workspace(self, workspace: Workspace):
        self.evaluate_folder = workspace.get_evaluate_folder()
        self.result_folder = workspace.get_result_folder()
        self.model_index = workspace.get_model_index()

    def run(self):
        model = Model()
        if self.model_index == 1:
            model.load_model("./models/UNet/cnn_24.pth", "./models/UNet/UNet.py")
        model.load_predict_data(self.evaluate_folder)
        model.run_model(self.result_folder)

        # self.sig.emit(True)
