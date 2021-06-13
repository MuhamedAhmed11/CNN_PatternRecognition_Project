# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'message.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(480, 170)
        Form.setMinimumSize(QtCore.QSize(480, 170))
        Form.setMaximumSize(QtCore.QSize(480, 170))
        Form.setStyleSheet("background-color: rgb(50, 50, 50);")
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setStyleSheet("border-style: outset;\n"
"border-color: rgb(0, 106, 159);\n"
"border-width: 4px;\n"
"border-radius: 10px;\n"
"border-color: rgb(50, 50, 50);\n"
"")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(110, 30, 261, 41))
        font = QtGui.QFont()
        font.setFamily("Fira Sans ExtraBold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 4px;\n"
"border-radius: 10px;\n"
"border-color: rgb(50, 50, 50);\n"
"")
        self.label.setObjectName("label")
        self.alex_button = QtWidgets.QPushButton(self.frame)
        self.alex_button.setGeometry(QtCore.QRect(270, 90, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Fira Sans ExtraBold")
        font.setPointSize(12)
        self.alex_button.setFont(font)
        self.alex_button.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 4px;\n"
"border-radius: 10px;\n"
"border-color:rgb(0, 121, 182);\n"
"")
        self.alex_button.setObjectName("alex_button")
        self.aan_button = QtWidgets.QPushButton(self.frame)
        self.aan_button.setGeometry(QtCore.QRect(50, 90, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Fira Sans ExtraBold")
        font.setPointSize(12)
        self.aan_button.setFont(font)
        self.aan_button.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 4px;\n"
"border-radius: 10px;\n"
"border-color:rgb(0, 121, 182);\n"
"")
        self.aan_button.setObjectName("aan_button")
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Please, Choose a Model?"))
        self.alex_button.setText(_translate("Form", "AlexNet"))
        self.aan_button.setText(_translate("Form", "Our Model"))

