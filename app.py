# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'classification.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from pyqtgraph import PlotWidget
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1022, 726)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("background-color: rgb(36, 36, 36);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Mongolian Baiti")
        font.setPointSize(14)
        self.tabWidget.setFont(font)
        self.tabWidget.setStyleSheet("border-style: outset;\n"
                                     "border-width: 4px;\n"
                                     "border-radius: 10px;\n"
                                     "border-color: rgb(36,  36, 36);\n"
                                     "color: rgb(0, 121, 182);\n"
                                     "")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 0, 1, 1)
        self.graphicsView = PlotWidget(self.tab)
        self.graphicsView.setMinimumSize(QtCore.QSize(350, 350))
        self.graphicsView.setMaximumSize(QtCore.QSize(350, 350))
        self.graphicsView.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                        "border-style: outset;\n"
                                        "border-width: 4px;\n"
                                        "border-radius: 10px;\n"
                                        "border-color: rgb(0, 121, 182);\n"
                                        "color:rgb(0, 121, 182);\n"
                                        "")
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout_2.addWidget(self.graphicsView, 2, 2, 1, 3)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 3, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setMinimumSize(QtCore.QSize(350, 0))
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("border-style: outset;\n"
                                   "color: rgb(255, 255, 255);\n"
                                   "border-width: 2px;\n"
                                   "border-radius: 5px;\n"
                                   "border-color: rgb(0, 121, 182);\n"
                                   "")
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 3, 4, 1, 2)
        self.browse_button = QtWidgets.QPushButton(self.tab)
        self.browse_button.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setFamily("Forte")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.browse_button.setFont(font)
        self.browse_button.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                         "border-style: outset;\n"
                                         "border-width: 4px;\n"
                                         "border-radius: 10px;\n"
                                         "border-color: rgb(0, 121, 182);\n"
                                         "color:rgb(0, 121, 182);\n"
                                         "")
        self.browse_button.setObjectName("browse_button")
        self.gridLayout_2.addWidget(self.browse_button, 1, 5, 1, 1)
        self.classify_button = QtWidgets.QPushButton(self.tab)
        self.classify_button.setMinimumSize(QtCore.QSize(250, 40))
        font = QtGui.QFont()
        font.setFamily("Forte")
        font.setPointSize(15)
        self.classify_button.setFont(font)
        self.classify_button.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                           "border-style: outset;\n"
                                           "border-width: 4px;\n"
                                           "border-radius: 10px;\n"
                                           "border-color:rgb(0, 121, 182);\n"
                                           "")
        self.classify_button.setObjectName("classify_button")
        self.gridLayout_2.addWidget(self.classify_button, 3, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(
            340, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 2, 5, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.tab)
        self.lineEdit.setMinimumSize(QtCore.QSize(0, 35))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                    "border-style: outset;\n"
                                    "border-width: 2px;\n"
                                    "border-radius: 7px;\n"
                                    "border-color: rgb(0, 121, 182);\n"
                                    "color:rgb(0, 121, 182);")
        self.lineEdit.setText("")
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_2.addWidget(self.lineEdit, 1, 1, 1, 3)
        spacerItem3 = QtWidgets.QSpacerItem(
            249, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 2, 1, 1, 1)
        self.model_name_viewer = QtWidgets.QTextBrowser(self.tab)
        self.model_name_viewer.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setFamily("Fira Sans ExtraBold")
        font.setPointSize(12)
        self.model_name_viewer.setFont(font)
        self.model_name_viewer.setStyleSheet("color: rgb(255, 255, 255);")
        self.model_name_viewer.setObjectName("model_name_viewer")
        self.gridLayout_2.addWidget(self.model_name_viewer, 0, 3, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem4 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem4, 0, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.tab_3)
        self.comboBox.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.comboBox.setFont(font)
        self.comboBox.setStyleSheet("border-style: outset;\n"
                                    "color: rgb(255, 255, 255);\n"
                                    "border-width: 2px;\n"
                                    "border-radius: 5px;\n"
                                    "border-color: rgb(0, 121, 182);\n"
                                    "")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout_4.addWidget(self.comboBox, 0, 1, 1, 1)
        self.tableWidget = QtWidgets.QTableWidget(self.tab_3)
        font = QtGui.QFont()
        font.setFamily("Raleway")
        font.setPointSize(12)
        self.tableWidget.setFont(font)
        self.tableWidget.setStyleSheet("border-style: outset;\n"
                                       "border-width: 2px;\n"
                                       "border-radius: 10px;\n"
                                       "border-color: rgb(0, 121, 182);;\n"
                                       "background-color: rgb(48, 48, 48);\n"
                                       "color: rgb(0, 121, 182);\n"
                                       "\n"
                                       "")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Raleway")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        item.setBackground(QtGui.QColor(255, 255, 255))
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Raleway")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        item.setBackground(QtGui.QColor(253, 253, 253))
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.tableWidget.verticalHeader().setVisible(False)
        self.gridLayout_4.addWidget(self.tableWidget, 1, 0, 1, 2)
        self.score_button = QtWidgets.QPushButton(self.tab_3)
        self.score_button.setMinimumSize(QtCore.QSize(250, 40))
        self.score_button.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setFamily("Forte")
        font.setPointSize(15)
        self.score_button.setFont(font)
        self.score_button.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                        "border-style: outset;\n"
                                        "border-width: 4px;\n"
                                        "border-radius: 10px;\n"
                                        "border-color:rgb(0, 121, 182);\n"
                                        "")
        self.score_button.setObjectName("score_button")
        self.gridLayout_4.addWidget(self.score_button, 2, 1, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem5 = QtWidgets.QSpacerItem(
            347, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem5, 1, 2, 1, 1)
        self.graphs_button = QtWidgets.QPushButton(self.tab_2)
        self.graphs_button.setMaximumSize(QtCore.QSize(350, 40))
        font = QtGui.QFont()
        font.setFamily("Forte")
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.graphs_button.setFont(font)
        self.graphs_button.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                         "border-style: outset;\n"
                                         "border-width: 4px;\n"
                                         "border-radius: 10px;\n"
                                         "border-color: rgb(0, 121, 182);\n"
                                         "color: rgb(0, 121, 182);\n"
                                         "")
        self.graphs_button.setObjectName("graphs_button")
        self.gridLayout_3.addWidget(self.graphs_button, 1, 1, 1, 1)
        self.graphicsView_3 = PlotWidget(self.tab_2)
        self.graphicsView_3.setMinimumSize(QtCore.QSize(350, 300))
        self.graphicsView_3.setMaximumSize(QtCore.QSize(350, 300))
        self.graphicsView_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                          "border-style: outset;\n"
                                          "border-width: 4px;\n"
                                          "border-radius: 10px;\n"
                                          "border-color: rgb(0, 121, 182);\n"
                                          "color: rgb(73, 220, 220);\n"
                                          "")
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.gridLayout_3.addWidget(self.graphicsView_3, 0, 2, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(
            347, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem6, 1, 0, 1, 1)
        self.graphicsView_2 = PlotWidget(self.tab_2)
        self.graphicsView_2.setMinimumSize(QtCore.QSize(350, 300))
        self.graphicsView_2.setMaximumSize(QtCore.QSize(350, 300))
        self.graphicsView_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                          "border-style: outset;\n"
                                          "border-width: 4px;\n"
                                          "border-radius: 10px;\n"
                                          "border-color: rgb(0, 121, 182);\n"
                                          "color: rgb(73, 220, 220);\n"
                                          "")
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout_3.addWidget(self.graphicsView_2, 0, 0, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(
            198, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem7, 0, 1, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.graphicsView.hideAxis('bottom')
        self.graphicsView.hideAxis('left')
        self.graphicsView.setBackground('w')
        self.graphicsView_2.hideAxis('bottom')
        self.graphicsView_2.hideAxis('left')
        self.graphicsView_2.setBackground('w')
        self.graphicsView_3.hideAxis('bottom')
        self.graphicsView_3.hideAxis('left')
        self.graphicsView_3.setBackground('w')

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate(
            "MainWindow", "Predicted Class: -----"))
        self.browse_button.setText(_translate("MainWindow", "Browse"))
        self.classify_button.setText(_translate("MainWindow", "Show Result"))
        self.model_name_viewer.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                  "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                  "p, li { white-space: pre-wrap; }\n"
                                                  "</style></head><body style=\" font-family:\'Fira Sans ExtraBold\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                                  "<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">----</span></p></body></html>"))
        
        self.tabWidget.setTabText(self.tabWidget.indexOf(
            self.tab), _translate("MainWindow", "Classifiaction"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Choose"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Accuracy"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Precision"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Recall"))
        self.comboBox.setItemText(4, _translate("MainWindow", "F1"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Class Labels"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Score"))
        self.score_button.setText(_translate("MainWindow", "Show Score"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(
            self.tab_3), _translate("MainWindow", "Statistics"))
        self.graphs_button.setText(_translate("MainWindow", "Show Graphs"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(
            self.tab_2), _translate("MainWindow", "Graphs"))
