from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from workspace import Workspace
from myModel import Model


class RunThread(QThread):
    # sig = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(RunThread, self).__init__(parent)

    def __del__(self):
        # self.wait()
        pass

    def set_workspace(self, workspace: Workspace):
        self.image_folder = workspace.get_image_folder()
        self.label_folder = workspace.get_label_folder()
        self.test_folder = workspace.get_test_folder()
        self.model_index = workspace.get_model_index()

    def run(self):
        model = Model()
        if self.model_index == 1:
            model.__load_model("./models/UNet/cnn_24.pth",
                             "./models/UNet/UNet.py")
        # model.load_predict_data(self.image_folder)
        # model.run_model(self.result_folder)
        model.load_train_data(self.image_folder, self.label_folder)
        model.train_model(2, 1)
