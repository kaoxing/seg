# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testTab.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_testTab(object):
    def setupUi(self, testTab):
        testTab.setObjectName("testTab")
        testTab.resize(590, 320)
        self.horizontalLayout = QtWidgets.QHBoxLayout(testTab)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(testTab)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(testTab)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(testTab)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)

        self.retranslateUi(testTab)
        QtCore.QMetaObject.connectSlotsByName(testTab)

    def retranslateUi(self, testTab):
        _translate = QtCore.QCoreApplication.translate
        testTab.setWindowTitle(_translate("testTab", "testTab"))
        self.pushButton.setText(_translate("testTab", "PushButton"))
        self.pushButton_2.setText(_translate("testTab", "PushButton"))
        self.pushButton_3.setText(_translate("testTab", "PushButton"))
