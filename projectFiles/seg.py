import os
import sys
import datetime
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileSystemModel, QFileDialog, QTreeView
from ui import MainWindow
from UI.threads.EvaluateThread import EvaluateThread
from UI.threads.RunThread import RunThread
from workspace import Workspace



if __name__ == "__main__":
    app = QApplication(sys.argv)
    evaluate_thread = EvaluateThread()
    run_thread = RunThread()
    main_window = MainWindow()
    workspace = Workspace()
    main_window.tab.set_workspace(workspace)
    main_window.tab_2.set_workspace(workspace)
    main_window.tab_3.set_workspace(workspace)
    main_window.tab_4.set_workspace(workspace)
    main_window.tab_3.set_threads(run_thread)
    main_window.tab_4.set_threads(evaluate_thread)
    main_window.show()
    sys.exit(app.exec_())