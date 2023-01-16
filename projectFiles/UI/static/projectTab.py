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
        self.verticalLayout = QtWidgets.QVBoxLayout(projectTab)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(1131, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.label_4 = QtWidgets.QLabel(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        spacerItem3 = QtWidgets.QSpacerItem(1131, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
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
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
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
        self.gridLayout.addWidget(self.lineEdit_project_name, 0, 1, 1, 1)
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
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lineEdit_train_folder = QtWidgets.QLineEdit(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit_train_folder.setFont(font)
        self.lineEdit_train_folder.setInputMask("")
        self.lineEdit_train_folder.setText("")
        self.lineEdit_train_folder.setReadOnly(True)
        self.lineEdit_train_folder.setObjectName("lineEdit_train_folder")
        self.gridLayout.addWidget(self.lineEdit_train_folder, 1, 1, 1, 1)
        self.pushButton_train = QtWidgets.QPushButton(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_train.setFont(font)
        self.pushButton_train.setObjectName("pushButton_train")
        self.gridLayout.addWidget(self.pushButton_train, 1, 2, 1, 1)
        self.label_20 = QtWidgets.QLabel(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_20.setFont(font)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.gridLayout.addWidget(self.label_20, 2, 0, 1, 1)
        self.lineEdit_test_folder = QtWidgets.QLineEdit(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit_test_folder.setFont(font)
        self.lineEdit_test_folder.setReadOnly(True)
        self.lineEdit_test_folder.setObjectName("lineEdit_test_folder")
        self.gridLayout.addWidget(self.lineEdit_test_folder, 2, 1, 1, 1)
        self.pushButton_test = QtWidgets.QPushButton(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_test.setFont(font)
        self.pushButton_test.setObjectName("pushButton_test")
        self.gridLayout.addWidget(self.pushButton_test, 2, 2, 1, 1)
        self.label_21 = QtWidgets.QLabel(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_21.setFont(font)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 3, 0, 1, 1)
        self.lineEdit_result_folder = QtWidgets.QLineEdit(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit_result_folder.setFont(font)
        self.lineEdit_result_folder.setReadOnly(True)
        self.lineEdit_result_folder.setObjectName("lineEdit_result_folder")
        self.gridLayout.addWidget(self.lineEdit_result_folder, 3, 1, 1, 1)
        self.pushButton_result = QtWidgets.QPushButton(projectTab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_result.setFont(font)
        self.pushButton_result.setObjectName("pushButton_result")
        self.gridLayout.addWidget(self.pushButton_result, 3, 2, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.horizontalLayout_2.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem6 = QtWidgets.QSpacerItem(1131, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem6)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem7)
        self.pushButton_confirm = QtWidgets.QPushButton(projectTab)
        self.pushButton_confirm.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.pushButton_confirm.setFont(font)
        self.pushButton_confirm.setObjectName("pushButton_confirm")
        self.horizontalLayout.addWidget(self.pushButton_confirm)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem8)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem9 = QtWidgets.QSpacerItem(1131, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem9)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 4)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(5, 1)
        self.verticalLayout.setStretch(6, 1)
        self.label.setBuddy(self.lineEdit_project_name)
        self.label_2.setBuddy(self.lineEdit_train_folder)
        self.label_20.setBuddy(self.lineEdit_test_folder)
        self.label_21.setBuddy(self.lineEdit_result_folder)

        self.retranslateUi(projectTab)
        QtCore.QMetaObject.connectSlotsByName(projectTab)

    def retranslateUi(self, projectTab):
        _translate = QtCore.QCoreApplication.translate
        projectTab.setWindowTitle(_translate("projectTab", "projectTab"))
        self.label_4.setText(_translate("projectTab", "Overview"))
        self.label.setText(_translate("projectTab", "Project Name："))
        self.lineEdit_project_name.setPlaceholderText(_translate("projectTab", "input a name"))
        self.label_2.setText(_translate("projectTab", "Train Folder："))
        self.lineEdit_train_folder.setPlaceholderText(_translate("projectTab", "select train folder"))
        self.pushButton_train.setText(_translate("projectTab", "Browse"))
        self.label_20.setText(_translate("projectTab", "Test Folder："))
        self.lineEdit_test_folder.setPlaceholderText(_translate("projectTab", "select test folder"))
        self.pushButton_test.setText(_translate("projectTab", "Browse"))
        self.label_21.setText(_translate("projectTab", "Result Folder："))
        self.lineEdit_result_folder.setPlaceholderText(_translate("projectTab", "select result folder"))
        self.pushButton_result.setText(_translate("projectTab", "Browse"))
        self.pushButton_confirm.setToolTip(_translate("projectTab", "confirm your modification"))
        self.pushButton_confirm.setText(_translate("projectTab", "Confirm"))
