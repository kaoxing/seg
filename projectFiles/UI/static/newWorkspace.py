# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'newWorkspace.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_newWorkspace(object):
    def setupUi(self, newWorkspace):
        newWorkspace.setObjectName("newWorkspace")
        newWorkspace.resize(385, 46)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        newWorkspace.setFont(font)
        self.horizontalLayout = QtWidgets.QHBoxLayout(newWorkspace)
        self.horizontalLayout.setContentsMargins(-1, 9, 9, 9)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit = QtWidgets.QLineEdit(newWorkspace)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.pushButton = QtWidgets.QPushButton(newWorkspace)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)

        self.retranslateUi(newWorkspace)
        QtCore.QMetaObject.connectSlotsByName(newWorkspace)

    def retranslateUi(self, newWorkspace):
        _translate = QtCore.QCoreApplication.translate
        newWorkspace.setWindowTitle(_translate("newWorkspace", "New Workspace"))
        self.pushButton.setText(_translate("newWorkspace", "new"))
