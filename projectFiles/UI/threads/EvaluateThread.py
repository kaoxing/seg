from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from workspace import Workspace
from myModel import Model


class EvaluateThread(QThread):
    # sig = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(EvaluateThread, self).__init__(parent)

    def __del__(self):
        # self.wait()
        pass

    def set_workspace(self, workspace: Workspace):
        self.image_folder = workspace.get_image_folder()
        self.result_folder = workspace.get_result_folder()
        self.model_index = workspace.get_model_index()

    def run(self):
        model = Model()
        if self.model_index == 1:
            model.load_model("./models/UNet/cnn_24.pth", "./models/UNet/UNet.py")
        model.load_predict_data(self.image_folder)
        model.run_model(self.result_folder)

        # self.sig.emit(True)


# import os
# import sys

# current_directory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_directory)

# if __name__ == "__main__":
#     # print(os.path.abspath("./"))
#     print(current_directory)