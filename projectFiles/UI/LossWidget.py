import sys
import random
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

class lossWidget(QWidget):
    def __init__(self,parent=None):
        super(lossWidget, self).__init__(parent)
        # self.resize(600, 600)

        # 1
        pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # 2
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3','d', '+', 'x', 'p', 'h', 'star'])
        r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])

        # 3
        self.pw = pg.PlotWidget(self)
        self.plot_data = self.pw.plot(x, y, pen=None, symbol=r_symbol, symbolBrush=r_color)

        # 4
        self.plot_btn = QPushButton('Replot', self)
        self.plot_btn.clicked.connect(self.plot_slot)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.pw)
        self.v_layout.addWidget(self.plot_btn)
        self.setLayout(self.v_layout)

    def plot_slot(self):
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        print(x)
        r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star'])
        r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])
        self.plot_data.setData(x, y, pen=None, symbol=r_symbol, symbolBrush=r_color)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = LossWidget()
    demo.show()
    sys.exit(app.exec_())