import json
import os


class Workspace:
    """
    工作区，用于存放和操作各种项目数据
    """

    def __init__(self):
        self.test_folder = None
        self.result_folder = None
        self.project_name = None
        self.image_folder = None
        self.label_folder = None
        self.model_index = 0

    def set_project_name(self, project_name: str):
        self.project_name = project_name
        print(project_name)

    def set_image_folder(self, image_folder: str):
        self.image_folder = image_folder
        print(self.image_folder)

    def set_label_folder(self, label_folder: str):
        self.label_folder = label_folder
        print(self.label_folder)

    def set_result_folder(self, result_folder: str):
        self.result_folder = result_folder
        print(self.result_folder)

    def set_test_folder(self, test_folder: str):
        self.test_folder = test_folder
        print(self.result_folder)

    def set_model_index(self, model_index: int):
        self.model_index = model_index

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

    def save_project(self):
        dic = {
            "result_folder": self.result_folder,
            "project_name": self.project_name,
            "image_folder": self.image_folder,
            "label_folder": self.label_folder,
            "test_folder": self.test_folder,
            "model_index": self.model_index,
        }
        with open("./projectList/"+self.project_name+".proj", 'w') as file:
            json.dump(dic, file)
        with open("./projectList/projectList.sav", 'r+') as file:
            content = file.read()
            if content.count(self.project_name) == 0:
                file.write(self.project_name+"\n")

    def load_project(self, file: str):
        path = f"./projectList/{file}.proj"
        if os.path.exists(path):
            with open(path) as file:
                dic = json.load(file)
            self.result_folder = dic["result_folder"]
            self.project_name = dic["project_name"]
            self.image_folder = dic["image_folder"]
            self.label_folder = dic["label_folder"]
            self.test_folder = dic["test_folder"]
            self.model_index = dic["model_index"]
            print(dic)
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
