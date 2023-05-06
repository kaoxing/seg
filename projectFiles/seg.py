import sys
import sysPath
import os
from PyQt5.QtWidgets import QApplication
from ui import MainWindow
from UI.action.newWorkspace import newWorkspace
from UI.action.dataTab import dataTab

if __name__ == "__main__":
    app = QApplication(sys.argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])
    # 窗口
    main_window = MainWindow()
    new_workspace_widget = newWorkspace()
    data_tab = dataTab()
    # 信号与槽
    main_window.new_workspace_sig.connect(new_workspace_widget.show)
    new_workspace_widget.sig.connect(main_window.new_workspace)
    main_window.data_tab_sig.connect(data_tab.show)
    main_window.show()
    sys.exit(app.exec_())
