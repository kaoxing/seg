from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from workspace import Workspace
from myModel import Model

class RealtimeModel(Model):
    loss_sig = pyqtSignal(int,int)
    def __init__(self):
        super().__init__()
        self.epoch = 0

    def state_change(self):
        self.loss_sig.emit(self.epoch,self.train_loss)
        return super().state_change()
        
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
        self.settings = workspace.get_settings()

    def run(self):
        model = RealtimeModel()
        if self.model_index == 1:
            model.load_model("./models/UNet/cnn_24.pth",
                             "./models/UNet/UNet.py")
        else:
            return
        model.load_train_data(self.image_folder, self.label_folder)
        model.train_model(
            epoch=self.settings["epochs"],
            batch_size=self.settings["batch_size"],
            learning_rate=self.settings["lr"],
            shuffle=True,
            optim=self.settings["optimizer"],
            loss_func=self.settings["loss_function"],
        )
