import json
import os
import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)


class Workspace:
    """
    工作区，用于存放和操作各种项目数据
    """

    def __init__(self, workdir: str):
        self.workdir = workdir
        config = workdir+"/.config"
        # 根据.config文件初始化
        if os.path.exists(config):
            self.load_project()
        else:
            self.project_name = os.path.basename(workdir)
            self.image_folder = workdir + "/data/image"
            self.label_folder = workdir + "/data/label"
            self.test_folder = workdir + "/data/test"
            self.result_folder = workdir + "/result"
            self.model_index = 0
            self.epochs = 0
            self.batch_size = 0
            self.loss_function = "Cross-Entropy"
            self.lr = 0.000001
            self.optimizer = "Adam"
            self.prtrain_model = "to be loaded"
            self.status = "waiting..."
            self.save_project()

    def set_project_name(self, project_name: str):
        self.project_name = project_name

    def set_image_folder(self, image_folder: str):
        self.image_folder = image_folder

    def set_label_folder(self, label_folder: str):
        self.label_folder = label_folder

    def set_result_folder(self, result_folder: str):
        self.result_folder = result_folder

    def set_test_folder(self, test_folder: str):
        self.test_folder = test_folder

    def set_model_index(self, model_index: int):
        self.model_index = model_index

    def set_settings(self, settings: dict):
        self.epochs = settings["epochs"]
        self.batch_size = settings["batch_size"]
        self.loss_function = settings["loss_function"]
        self.lr = settings["lr"]
        self.optimizer = settings["optimizer"]
        self.prtrain_model = settings["prtrain_model"]
        self.status = settings["status"]

    def get_project_name(self):
        return self.project_name

    def get_image_folder(self):
        return self.image_folder

    def get_label_folder(self):
        return self.label_folder

    def get_result_folder(self):
        return self.result_folder

    def get_test_folder(self):
        return self.test_folder

    def get_model_index(self):
        return self.model_index

    def get_settings(self):
        settings = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "prtrain_model": self.prtrain_model,
            "status": self.status,
        }
        return settings

    def save_project(self):
        dic = {
            "project_name": self.project_name,
            "image_folder": self.image_folder,
            "label_folder": self.label_folder,
            "test_folder": self.test_folder,
            "result_folder": self.result_folder,
            "model_index": self.model_index,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "prtrain_model": self.prtrain_model,
            "status": self.status,
        }
        path = self.workdir+"/.config"
        with open(path, 'w') as file:
            json.dump(dic, file)
            logging.info(f"project saved in {path}")
        with open("./projectList/projectList.sav", 'r+') as file:
            content = file.read()
            if content.count(path) == 0:
                file.write(f"{self.project_name} {self.workdir}\n")

    def load_project(self):
        path = self.workdir+"/.config"
        if os.path.exists(path):
            with open(path) as file:
                dic = json.load(file)
            self.project_name = dic["project_name"]
            self.image_folder = dic["image_folder"]
            self.label_folder = dic["label_folder"]
            self.test_folder = dic["test_folder"]
            self.result_folder = dic["result_folder"]
            self.model_index = dic["model_index"]
            self.epochs = dic["epochs"]
            self.batch_size = dic["batch_size"]
            self.loss_function = dic["loss_function"]
            self.lr = dic["lr"]
            self.optimizer = dic["optimizer"]
            self.prtrain_model = dic["prtrain_model"]
            self.status = dic["status"]
            logging.info(f"project loaded from {path}")
            return True
        else:
            return False

    def check(self):
        if self.image_folder is not None:
            return True
        else:
            return False


if __name__ == "__main__":
    a = os.path.abspath("./")
    print(a)
