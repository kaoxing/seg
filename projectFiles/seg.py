import sys
import sysPath
from PyQt5.QtWidgets import QApplication
from ui import MainWindow
from UI.action.newWorkspace import newWorkspace

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 窗口
    main_window = MainWindow()
    new_workspace_widget = newWorkspace()
    # 信号与槽
    main_window.new_workspace_sig.connect(new_workspace_widget.show)
    new_workspace_widget.sig.connect(main_window.new_workspace)
    main_window.show()
    sys.exit(app.exec_())
