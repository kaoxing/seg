# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'evaluateTab.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_evaluateTab(object):
    def setupUi(self, evaluateTab):
        evaluateTab.setObjectName("evaluateTab")
        evaluateTab.resize(1133, 609)
        self.verticalLayout = QtWidgets.QVBoxLayout(evaluateTab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.label_25 = QtWidgets.QLabel(evaluateTab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.horizontalLayout_23.addWidget(self.label_25)
        self.lineEdit_evaluate_folder = QtWidgets.QLineEdit(evaluateTab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit_evaluate_folder.setFont(font)
        self.lineEdit_evaluate_folder.setInputMask("")
        self.lineEdit_evaluate_folder.setText("")
        self.lineEdit_evaluate_folder.setReadOnly(True)
        self.lineEdit_evaluate_folder.setObjectName("lineEdit_evaluate_folder")
        self.horizontalLayout_23.addWidget(self.lineEdit_evaluate_folder)
        self.pushButton_evaluate = QtWidgets.QPushButton(evaluateTab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_evaluate.setFont(font)
        self.pushButton_evaluate.setObjectName("pushButton_evaluate")
        self.horizontalLayout_23.addWidget(self.pushButton_evaluate)
        self.verticalLayout.addLayout(self.horizontalLayout_23)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_20 = QtWidgets.QLabel(evaluateTab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.horizontalLayout.addWidget(self.label_20)
        self.lineEdit_loaded_model = QtWidgets.QLineEdit(evaluateTab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.lineEdit_loaded_model.setFont(font)
        self.lineEdit_loaded_model.setReadOnly(True)
        self.lineEdit_loaded_model.setObjectName("lineEdit_loaded_model")
        self.horizontalLayout.addWidget(self.lineEdit_loaded_model)
        self.label_21 = QtWidgets.QLabel(evaluateTab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.horizontalLayout.addWidget(self.label_21)
        self.lineEdit_status = QtWidgets.QLineEdit(evaluateTab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.lineEdit_status.setFont(font)
        self.lineEdit_status.setText("")
        self.lineEdit_status.setReadOnly(True)
        self.lineEdit_status.setObjectName("lineEdit_status")
        self.horizontalLayout.addWidget(self.lineEdit_status)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.frame_3d = QtWidgets.QFrame(evaluateTab)
        self.frame_3d.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3d.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_3d.setObjectName("frame_3d")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_3d)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget_3d = Image3dWidget(self.frame_3d)
        self.widget_3d.setMinimumSize(QtCore.QSize(0, 0))
        self.widget_3d.setObjectName("widget_3d")
        self.horizontalLayout_2.addWidget(self.widget_3d)
        self.verticalLayout.addWidget(self.frame_3d)
        self.widget_13 = QtWidgets.QWidget(evaluateTab)
        self.widget_13.setMinimumSize(QtCore.QSize(0, 200))
        self.widget_13.setMaximumSize(QtCore.QSize(16777215, 250))
        self.widget_13.setObjectName("widget_13")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget_13)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.splitter_5 = QtWidgets.QSplitter(self.widget_13)
        self.splitter_5.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_5.setObjectName("splitter_5")
        self.layoutWidget = QtWidgets.QWidget(self.splitter_5)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_26 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.verticalLayout_9.addWidget(self.label_26)
        self.widget_input = ImageListWidget(self.layoutWidget)
        self.widget_input.setObjectName("widget_input")
        self.verticalLayout_9.addWidget(self.widget_input)
        self.layoutWidget_2 = QtWidgets.QWidget(self.splitter_5)
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_27 = QtWidgets.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.verticalLayout_10.addWidget(self.label_27)
        self.widget_result = ImageListWidget(self.layoutWidget_2)
        self.widget_result.setObjectName("widget_result")
        self.verticalLayout_10.addWidget(self.widget_result)
        self.gridLayout_5.addWidget(self.splitter_5, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.widget_13)
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_24.addItem(spacerItem)
        self.pushButton_start = QtWidgets.QPushButton(evaluateTab)
        self.pushButton_start.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.horizontalLayout_24.addWidget(self.pushButton_start)
        self.pushButton_modeling = QtWidgets.QPushButton(evaluateTab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_modeling.setFont(font)
        self.pushButton_modeling.setObjectName("pushButton_modeling")
        self.horizontalLayout_24.addWidget(self.pushButton_modeling)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_24.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_24)

        self.retranslateUi(evaluateTab)
        QtCore.QMetaObject.connectSlotsByName(evaluateTab)

    def retranslateUi(self, evaluateTab):
        _translate = QtCore.QCoreApplication.translate
        evaluateTab.setWindowTitle(_translate("evaluateTab", "evaluateTab"))
        self.label_25.setText(_translate("evaluateTab", " Input Folder："))
        self.lineEdit_evaluate_folder.setPlaceholderText(_translate("evaluateTab", "select input folder"))
        self.pushButton_evaluate.setText(_translate("evaluateTab", "Browse"))
        self.label_20.setText(_translate("evaluateTab", "Loaded Model:"))
        self.label_21.setText(_translate("evaluateTab", "status:"))
        self.label_26.setText(_translate("evaluateTab", " input："))
        self.label_27.setText(_translate("evaluateTab", "result："))
        self.pushButton_start.setText(_translate("evaluateTab", "start"))
        self.pushButton_modeling.setText(_translate("evaluateTab", "Modeling"))
from UI.Widgets.image3dWidget import Image3dWidget
from UI.Widgets.imageWidget import ImageListWidget
