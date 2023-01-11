# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'projectTab.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_projectTab(object):
    def setupUi(self, projectTab):
        projectTab.setObjectName("projectTab")
        projectTab.resize(1152, 614)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(projectTab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.stackedWidget_state = QtWidgets.QStackedWidget(projectTab)
        self.stackedWidget_state.setObjectName("stackedWidget_state")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.page)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem)
        self.label_12 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_12.addWidget(self.label_12)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem1)
        self.horizontalLayout_13.addLayout(self.horizontalLayout_12)
        self.stackedWidget_state.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.page_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.label_4 = QtWidgets.QLabel(self.page_2)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_5)
        self.stackedWidget_state.addWidget(self.page_2)
        self.verticalLayout_2.addWidget(self.stackedWidget_state)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.lineEdit_project_name = QtWidgets.QLineEdit(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit_project_name.setFont(font)
        self.lineEdit_project_name.setInputMask("")
        self.lineEdit_project_name.setText("")
        self.lineEdit_project_name.setObjectName("lineEdit_project_name")
        self.gridLayout_3.addWidget(self.lineEdit_project_name, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 2, 0, 1, 1)
        self.pushButton_label = QtWidgets.QPushButton(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_label.setFont(font)
        self.pushButton_label.setObjectName("pushButton_label")
        self.gridLayout_3.addWidget(self.pushButton_label, 2, 2, 1, 1)
        self.lineEdit_image_folder = QtWidgets.QLineEdit(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit_image_folder.setFont(font)
        self.lineEdit_image_folder.setInputMask("")
        self.lineEdit_image_folder.setText("")
        self.lineEdit_image_folder.setReadOnly(True)
        self.lineEdit_image_folder.setObjectName("lineEdit_image_folder")
        self.gridLayout_3.addWidget(self.lineEdit_image_folder, 1, 1, 1, 1)
        self.pushButton_image = QtWidgets.QPushButton(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_image.setFont(font)
        self.pushButton_image.setObjectName("pushButton_image")
        self.gridLayout_3.addWidget(self.pushButton_image, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.lineEdit_label_folder = QtWidgets.QLineEdit(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit_label_folder.setFont(font)
        self.lineEdit_label_folder.setInputMask("")
        self.lineEdit_label_folder.setText("")
        self.lineEdit_label_folder.setReadOnly(True)
        self.lineEdit_label_folder.setObjectName("lineEdit_label_folder")
        self.gridLayout_3.addWidget(self.lineEdit_label_folder, 2, 1, 1, 1)
        self.label_20 = QtWidgets.QLabel(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.gridLayout_3.addWidget(self.label_20, 3, 0, 1, 1)
        self.lineEdit_test_folder = QtWidgets.QLineEdit(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit_test_folder.setFont(font)
        self.lineEdit_test_folder.setObjectName("lineEdit_test_folder")
        self.gridLayout_3.addWidget(self.lineEdit_test_folder, 3, 1, 1, 1)
        self.pushButton_test = QtWidgets.QPushButton(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_test.setFont(font)
        self.pushButton_test.setObjectName("pushButton_test")
        self.gridLayout_3.addWidget(self.pushButton_test, 3, 2, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_3)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 4)
        self.horizontalLayout.setStretch(2, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.stackedWidget = QtWidgets.QStackedWidget(projectTab)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_new = QtWidgets.QWidget()
        self.page_new.setObjectName("page_new")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.page_new)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.widget = QtWidgets.QWidget(self.page_new)
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.textEdit = QtWidgets.QTextEdit(self.widget)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout_2.addWidget(self.textEdit)
        self.verticalLayout_5.addWidget(self.widget)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_back = QtWidgets.QPushButton(self.page_new)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_back.setFont(font)
        self.pushButton_back.setObjectName("pushButton_back")
        self.horizontalLayout_4.addWidget(self.pushButton_back)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem6)
        self.pushButton_create = QtWidgets.QPushButton(self.page_new)
        self.pushButton_create.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_create.setFont(font)
        self.pushButton_create.setObjectName("pushButton_create")
        self.horizontalLayout_4.addWidget(self.pushButton_create)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.stackedWidget.addWidget(self.page_new)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.page_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem7)
        self.pushButton_new = QtWidgets.QPushButton(self.page_3)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_new.setFont(font)
        self.pushButton_new.setObjectName("pushButton_new")
        self.horizontalLayout_7.addWidget(self.pushButton_new)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem8)
        self.pushButton_confirm = QtWidgets.QPushButton(self.page_3)
        self.pushButton_confirm.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_confirm.setFont(font)
        self.pushButton_confirm.setObjectName("pushButton_confirm")
        self.horizontalLayout_7.addWidget(self.pushButton_confirm)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem9)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.line = QtWidgets.QFrame(self.page_3)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem10)
        self.label_13 = QtWidgets.QLabel(self.page_3)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_14.addWidget(self.label_13)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem11)
        self.verticalLayout.addLayout(self.horizontalLayout_14)
        self.listWidget = QtWidgets.QListWidget(self.page_3)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)
        self.stackedWidget.addWidget(self.page_3)
        self.verticalLayout_2.addWidget(self.stackedWidget)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 7)
        self.label_3.setBuddy(self.lineEdit_label_folder)
        self.label_2.setBuddy(self.lineEdit_image_folder)
        self.label.setBuddy(self.lineEdit_project_name)
        self.label_20.setBuddy(self.lineEdit_test_folder)

        self.retranslateUi(projectTab)
        self.stackedWidget_state.setCurrentIndex(0)
        self.stackedWidget.setCurrentIndex(0)
        self.stackedWidget_state.currentChanged['int'].connect(self.stackedWidget.setCurrentIndex)
        QtCore.QMetaObject.connectSlotsByName(projectTab)

    def retranslateUi(self, projectTab):
        _translate = QtCore.QCoreApplication.translate
        projectTab.setWindowTitle(_translate("projectTab", "projectTab"))
        self.label_12.setText(_translate("projectTab", "New Project"))
        self.label_4.setText(_translate("projectTab", "Overview"))
        self.lineEdit_project_name.setPlaceholderText(_translate("projectTab", "input a name"))
        self.label_3.setText(_translate("projectTab", "Label Folder："))
        self.pushButton_label.setText(_translate("projectTab", "Browse"))
        self.lineEdit_image_folder.setPlaceholderText(_translate("projectTab", "select image folder"))
        self.pushButton_image.setText(_translate("projectTab", "Browse"))
        self.label_2.setText(_translate("projectTab", "Image Folder："))
        self.label.setText(_translate("projectTab", "Project Name："))
        self.lineEdit_label_folder.setPlaceholderText(_translate("projectTab", "select label folder"))
        self.label_20.setText(_translate("projectTab", "Test  Folder："))
        self.lineEdit_test_folder.setPlaceholderText(_translate("projectTab", "select test folder"))
        self.pushButton_test.setText(_translate("projectTab", "Browse"))
        self.textEdit.setHtml(_translate("projectTab", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'黑体\'; font-size:20pt; font-weight:600;\">Guide</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'黑体\'; font-size:20pt; font-weight:600;\">1.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'黑体\'; font-size:20pt; font-weight:600;\">2.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'黑体\'; font-size:20pt; font-weight:600;\">3.</span></p></body></html>"))
        self.pushButton_back.setText(_translate("projectTab", "Back"))
        self.pushButton_create.setText(_translate("projectTab", "Create"))
        self.pushButton_new.setText(_translate("projectTab", "New"))
        self.pushButton_confirm.setText(_translate("projectTab", "Confirm"))
        self.label_13.setText(_translate("projectTab", "Recent Project"))
