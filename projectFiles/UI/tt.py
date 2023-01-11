from PyQt5.QtWidgets import QWidget
from UI.testTab import Ui_testTab

class testTab(Ui_testTab,QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)