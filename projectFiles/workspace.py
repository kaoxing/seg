class Workspace:
    """
    工作区，用于存放和操作各种项目数据
    """

    def __init__(self):
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

    def get_model_index(self):
        return self.model_index

    def save_project(self):
        pass
        # TODO

    def load_project(self):
        pass
        # TODO

    def check(self):
        if self.image_folder is not None:
            return True
        else:
            return False
