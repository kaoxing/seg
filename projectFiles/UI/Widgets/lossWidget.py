import sys
import random
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import logging
logging.basicConfig(
    # filename='./log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s:%(message)s'
)


class lossWidget(QWidget):
    def __init__(self, parent=None):
        super(lossWidget, self).__init__(parent)
        pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.pw = pg.PlotWidget(self)
        self.plot_item = self.pw.getPlotItem()
        self.init_plot_item()
        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.pw)
        self.setLayout(self.v_layout)
        self.loss_list = []

    def init_plot_item(self):
        """
        初始化绘图区
        """
        self.plot_item.setLabel("left", "loss")
        self.plot_item.setLabel("bottom", "epoch")
        self.plot_item.showGrid(x=True, y=True)

    # def plot_slot(self):
    #     x = np.random.normal(size=1000)
    #     y = np.random.normal(size=1000)
    #     r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star'])
    #     r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])
    #     self.plot_data.setData(x, y, pen=None, symbol=r_symbol, symbolBrush=r_color)

    def loss_plot(self, loss: float):
        """
        往后绘制一个loss值
        """
        logging.info(f"loss_plot() received loss:{loss}")
        self.loss_list.append(loss)
        x = range(len(self.loss_list))
        self.plot_item.plot().setData(
            x,
            self.loss_list,
            symbol="o",
            symbolBrush=("b"),
            pen="r"
        )

    def reset_plot_item(self):
        """
        重置绘图区
        """
        self.plot_item.clear()
        self.loss_list = []

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = lossWidget()
    demo.show()
    demo.loss_plot(2)
    demo.loss_plot(3)
    demo.loss_plot(4)
    sys.exit(app.exec_())
