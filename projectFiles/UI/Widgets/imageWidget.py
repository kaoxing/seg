import os
import time

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMenu, QAbstractItemView, QListWidgetItem, QListView
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from UI.Widgets.singleImageView import SingleImageView

class ImageListWidget(QtWidgets.QListWidget):
    show_single_image_sig = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_image_view = SingleImageView()
        self.show_single_image_sig.connect(self.single_image_view.set_image)
        self.show_single_image_sig.connect(self.single_image_view.show)
        self.image_cmp_widget = None
        self.single_image = None
        self.setWindowTitle('All Images')
        self.resize(1400, 700)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        # 创建QMenu信号事件
        # self.customContextMenuRequested.connect(self.showMenu)
        # self.contextMenu = QMenu(self)
        # self.CMP = self.contextMenu.addAction('比较')
        # # self.CP = self.contextMenu.addAction('复制')
        # self.DL = self.contextMenu.addAction('删除')
        # # self.CP.triggered.connect(self.copy)
        # self.DL.triggered.connect(self.del_text)

        # 设置每个item size
        self.setGridSize(QtCore.QSize(220, 190))
        # 设置横向list
        self.setFlow(QListView.LeftToRight)
        # 设置换行
        self.setWrapping(True)
        # 窗口size 变化后重新计算列数
        self.setResizeMode(QtWidgets.QListView.Adjust)
        # 设置选择模式
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setIconSize(QSize(200, 150))

    # 显示右键菜单
    # def showMenu(self, pos):
    #     # pos 鼠标位置
    #     # 菜单显示前,将它移动到鼠标点击的位置
    #     self.contextMenu.exec_(QCursor.pos())  # 在鼠标位置显示

    # 获取选择行的内容
    # def selected_text(self):
    #     try:
    #         selected = self.selectedItems()
    #         texts = ''
    #         for item in selected:
    #             if texts:
    #                 texts = texts + '\n' + item.text()
    #             else:
    #                 texts = item.text()
    #     except BaseException as e:
    #         print(e)
    #         return
    #     print('selected_text texts', texts)
    #     return texts

    # def copy(self):
    #     text = self.selected_text()
    #     if text:
    #         clipboard = QApplication.clipboard()
    #         clipboard.setText(text)

    # def del_text(self):
    #     try:
    #         index = self.selectedIndexes()
    #         row = []

    #         for i in index:
    #             r = i.row()
    #             row.append(r)
    #         for i in sorted(row, reverse=True):
    #             self.takeItem(i)
    #     except BaseException as e:
    #         print(e)
    #         return
        # self.signal.emit(row)

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mouseDoubleClickEvent(e)
        selected = self.selectedItems()
        img_path = ''
        for item in selected:
            img_path = item.get_image_path()
        if os.path.exists(img_path):
            # 打开新窗口显示单张图片
            self.show_single_image_sig.emit(img_path)

    def load_images(self, image_folder:str):
        for image in os.listdir(image_folder):
            self.add_image(os.path.join(image_folder,image))

    def add_image(self,path:str):
        if path.endswith(".png") or path.endswith(".jpg"):
            img_item = ImageQListWidgetItem(path)
            self.addItem(img_item)
            self.setItemWidget(img_item, img_item.widget)
            # 刷新界面
            QApplication.processEvents()


# 自定义的item 继承自QListWidgetItem
class ImageQListWidgetItem(QListWidgetItem):
    def __init__(self,img_path:str):
        super().__init__()

        self.img_path = img_path
        # 自定义item中的widget 用来显示自定义的内容
        self.widget = QWidget()
        # 用来显示name
        self.nameLabel = QLabel()
        self.nameLabel.setText(os.path.basename(img_path))
        # 用来显示avator(图像)
        self.avatorLabel = QLabel()
        # 设置图像源 和 图像大小
        img_obg = QPixmap(img_path)
        width = img_obg.width()
        height = img_obg.height()
        scale_size = QSize(200, 150)
        if width < height:
            scale_size = QSize(150, 200)
        self.avatorLabel.setPixmap(QPixmap(img_path).scaled(scale_size))
        # 图像自适应窗口大小
        self.avatorLabel.setScaledContents(True)
        # 设置布局用来对nameLabel和avatorLabel进行布局
        self.hbox = QVBoxLayout()
        self.hbox.addWidget(self.avatorLabel)
        self.hbox.addWidget(self.nameLabel)
        self.hbox.addStretch(1)
        # 设置widget的布局
        self.widget.setLayout(self.hbox)
        # 设置自定义的QListWidgetItem的sizeHint，不然无法显示
        self.setSizeHint(self.widget.sizeHint())

    def get_name(self):
        return self.nameLabel.text()

    def get_image_path(self):
        return self.img_path


if __name__ == '__main__':
    print('main layout show')
    now = time.time()
    app = QApplication([])
    main_window = ImageListWidget()
    main_window.show()


    # 数据扩充
    main_window.load_images("E:/CodeField/workdir_20230114_180927/data/test/image")

    # 绑定点击槽函数 点击显示对应item中的name
    # main_window.itemClicked.connect(lambda item: print('clicked item label:', item.nameLabel.text()))
    print("ImageListWidget 耗时: {:.2f}秒".format(time.time() - now))

    app.exec_()
