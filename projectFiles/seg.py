import os
import sys
import datetime
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileSystemModel, QFileDialog, QTreeView
from ui import MainWindow
from UI.action.newWorkspace import newWorkspace
from UI.threads.EvaluateThread import EvaluateThread
from UI.threads.RunThread import RunThread
from workspace import Workspace



if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 窗口
    main_window = MainWindow()
    new_workspace_widget = newWorkspace()
    # 线程
    evaluate_thread = EvaluateThread()
    run_thread = RunThread()
    main_window.tab_3.set_threads(run_thread)
    main_window.tab_4.set_threads(evaluate_thread)
    # 信号与槽
    main_window.new_workspace_sig.connect(new_workspace_widget.show)
    new_workspace_widget.sig.connect(main_window.new_workspace)
    main_window.show()
    sys.exit(app.exec_())