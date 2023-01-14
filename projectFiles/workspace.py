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
            self.train_folder = workdir + "/data/train"
            self.test_folder = workdir + "/data/test"
            self.evaluate_folder = ""
            self.result_folder = workdir + "/result"
            self.model_index = 0
            self.epochs = 0
            self.batch_size = 0
            self.loss_function = "Cross-Entropy"
            self.lr = 0.000001
            self.optimizer = "Adam"
            self.pretrain_model = "to be loaded"
            self.status = "waiting..."
            self.save_project()

    # 下面几个方法都是SET系列的(START)
    def set_project_name(self, project_name: str):
        self.project_name = project_name

    def set_train_folder(self, train_folder: str):
        self.train_folder = train_folder

    def set_test_folder(self, test_folder: str):
        self.test_folder = test_folder

    def set_evaluate_folder(self, evaluate_folder: str):
        self.evaluate_folder = evaluate_folder

    def set_result_folder(self, result_folder: str):
        self.result_folder = result_folder

    def set_model_index(self, model_index: int):
        self.model_index = model_index

    def set_settings(self, settings: dict):
        self.epochs = settings["epochs"]
        self.batch_size = settings["batch_size"]
        self.loss_function = settings["loss_function"]
        self.lr = settings["lr"]
        self.optimizer = settings["optimizer"]
        self.pretrain_model = settings["pretrain_model"]
        self.status = settings["status"]
    # 上面几个方法都是SET系列的(END)

    # 下面几个方法都是GET系列的(START)
    def get_project_name(self):
        return self.project_name

    def get_train_folder(self):
        return self.train_folder

    def get_test_folder(self):
        return self.test_folder

    def get_evaluate_folder(self):
        return self.evaluate_folder

    def get_result_folder(self):
        return self.result_folder

    def get_model_index(self):
        return self.model_index

    def get_settings(self):
        settings = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "pretrain_model": self.pretrain_model,
            "status": self.status,
        }
        return settings
    # 上面几个方法都是GET系列的(END)

    def save_project(self):
        """
        保存项目
        """
        dic = {
            "project_name": self.project_name,
            "train_folder": self.train_folder,
            "test_folder": self.test_folder,
            "evaluate_folder": self.evaluate_folder,
            "result_folder": self.result_folder,
            "model_index": self.model_index,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "pretrain_model": self.pretrain_model,
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
        """
        载入项目
        """
        path = self.workdir+"/.config"
        if os.path.exists(path):
            with open(path) as file:
                dic = json.load(file)
            self.project_name = dic["project_name"]
            self.train_folder = dic["train_folder"]
            self.test_folder = dic["test_folder"]
            self.evaluate_folder = dic["evaluate_folder"]
            self.result_folder = dic["result_folder"]
            self.model_index = dic["model_index"]
            self.epochs = dic["epochs"]
            self.batch_size = dic["batch_size"]
            self.loss_function = dic["loss_function"]
            self.lr = dic["lr"]
            self.optimizer = dic["optimizer"]
            self.pretrain_model = dic["pretrain_model"]
            self.status = dic["status"]
            logging.info(f"project loaded from {path}")
            return True
        else:
            return False


if __name__ == "__main__":
    a = os.path.abspath("./")
    print(a)
