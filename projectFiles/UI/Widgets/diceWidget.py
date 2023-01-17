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


class diceWidget(QWidget):
    def __init__(self, parent=None):
        super(diceWidget, self).__init__(parent)
        pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.pw = pg.PlotWidget(self)
        self.plot_item = self.pw.getPlotItem()
        self.init_plot_item()
        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(self.pw)
        self.setLayout(self.v_layout)
        self.dice_list = []

    def init_plot_item(self):
        """
        初始化绘图区
        """
        self.plot_item.setLabel("left", "dice")
        self.plot_item.setLabel("bottom", "number")
        self.plot_item.showGrid(x=True, y=True)

    # def plot_slot(self):
    #     x = np.random.normal(size=1000)
    #     y = np.random.normal(size=1000)
    #     r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star'])
    #     r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])
    #     self.plot_data.setData(x, y, pen=None, symbol=r_symbol, symbolBrush=r_color)

    def dice_plot(self, dice: float):
        """
        往后绘制一个dice值
        """
        logging.info(f"dice_plot() received dice:{dice}")
        self.dice_list.append(dice)
        x = range(len(self.dice_list))
        self.plot_item.plot().setData(
            x,
            self.dice_list,
            symbol="o",
            symbolBrush=("b"),
            pen="r"
        )

    def reset_plot_item(self):
        """
        重置绘图区
        """
        self.plot_item.clear()
        self.dice_list = []


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = diceWidget()
    demo.show()
    demo.dice_plot(2)
    demo.dice_plot(3)
    demo.dice_plot(4)
    sys.exit(app.exec_())
