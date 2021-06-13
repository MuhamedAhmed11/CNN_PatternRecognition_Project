from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from PyQt5 import QtWidgets, QtCore
from app import Ui_MainWindow
from tensorflow import keras
from message import Ui_Form
import pyqtgraph as pg
import numpy as np
import sys
import os
import cv2


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model_to_run = ""
        self.image_array = []
        self.X_test = np.load("./data/X_test.npy")
        self.Y_test = np.load("./data/Y_test.npy")
        ##
        self.ui.browse_button.clicked.connect(self.browse_image)
        self.ui.classify_button.clicked.connect(self.show_result)
        self.ui.score_button.clicked.connect(self.stats)
        self.ui.graphs_button.clicked.connect(self.plot_graphs)
        ##
        self.labels_dic = {
            0: "Bonsai",
            1: "Brain",
            2: "Butterfly",
            3: "Chandelier",
            4: "Grand Piano",
            5: "Hawks Bill",
            6: "Helicopter",
            7: "Ketch",
            8: "Leaopards",
            9: "Starfish",
            10: "Watch",
        }

        self.message_window()

    def message_window(self):
        self.Form = QtWidgets.QWidget()
        self.ui_2 = Ui_Form()
        self.ui_2.setupUi(self.Form)
        self.Form.show()
        self.ui_2.aan_button.clicked.connect(
            lambda: self.change_model_to_run(1))
        self.ui_2.alex_button.clicked.connect(
            lambda: self.change_model_to_run(2))

    def change_model_to_run(self, value):
        _translate = QtCore.QCoreApplication.translate
        if value == 1:
            self.model_to_run = "./stats/our_model_stats/"
            self.model = keras.models.load_model('./Models/Model')
            self.ui.model_name_viewer.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                         "p, li { white-space: pre-wrap; }\n"
                                                         "</style></head><body style=\" font-family:\'Fira Sans ExtraBold\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                                         "<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Our Model</span></p></body></html>"))

            self.ui_2 = Ui_Form()
            self.ui_2.setupUi(self.Form)
            self.Form.close()

        if value == 2:
            self.model_to_run = "./stats/alex_net_stats/"

            self.model = keras.models.load_model('./Models/AlexNet')

            self.ui.model_name_viewer.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                         "p, li { white-space: pre-wrap; }\n"
                                                         "</style></head><body style=\" font-family:\'Fira Sans ExtraBold\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                                         "<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">AlexNet Model</span></p></body></html>"))

            self.ui_2 = Ui_Form()
            self.ui_2.setupUi(self.Form)
            self.Form.close()

    def save_scores(self):
        if self.model_to_run != "":
            y_pred = self.model.predict(self.X_test)
            Y_test_ = np.argmax(self.Y_test, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

            precision_of_classes = precision_score(
                Y_test_, y_pred, average=None)
            np.save(self.model_to_run + "Precision_Scores", precision_of_classes,
                    allow_pickle=False, fix_imports=False)

            recall_of_classes = recall_score(Y_test_, y_pred, average=None)
            np.save(self.model_to_run + "Recall_Scores", recall_of_classes,
                    allow_pickle=False, fix_imports=False)

            f1_of_classes = f1_score(Y_test_, y_pred, average=None)
            np.save(self.model_to_run + "F1_Scores", f1_of_classes,
                    allow_pickle=False, fix_imports=False)

            # Get the confusion matrix
            cm = confusion_matrix(Y_test_, y_pred)

            # We will store the results in a dictionary for easy access later
            per_class_accuracies = {}

            # Calculate the accuracy for each one of our classes
            for idx, cls in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
                # True negatives are all the samples that are not our current GT class (not the current row)
                # and were not predicted as the current class (not the current column)
                true_negatives = np.sum(
                    np.delete(np.delete(cm, idx, axis=0), idx, axis=1))

                # True positives are all the samples of our current GT class that were predicted as such
                true_positives = cm[idx, idx]

                # The accuracy for the current class is ratio between correct predictions to all predictions
                per_class_accuracies[cls] = (
                    true_positives + true_negatives) / np.sum(cm)

            accuracy_list = list(per_class_accuracies.values())
            np.save(self.model_to_run + "Accuracy_Scores", accuracy_list,
                    allow_pickle=False, fix_imports=False)
        else:
            self.message_window()

    def browse_image(self):
        if self.model_to_run != "":
            filepath = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open file', os.getcwd(), "Image files (*.jpg *.JPG *.png *.PNG)")

            if filepath[0] == '':
                QtWidgets.QMessageBox.setStyleSheet(
                    self, "background-color: rgb(255, 255, 255);")
                choice = QtWidgets.QMessageBox.question(
                    self, 'WARNING!', "Please Choose file", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                if choice == QtWidgets.QMessageBox.Yes:
                    self.browse_image()
                    sys.exit
                else:
                    return

            self.ui.lineEdit.setText(filepath[0])

            self.image_array = cv2.imread(filepath[0])
            self.image_array = cv2.cvtColor(
                self.image_array, cv2.COLOR_BGR2RGB)
            img = pg.ImageItem(self.image_array)
            img.rotate(270)

            self.image_array = cv2.resize(
                self.image_array, (300, 300), interpolation=cv2.INTER_AREA)

            self.ui.graphicsView.clear()
            self.ui.graphicsView.addItem(img)
        else:
            self.message_window()

    def show_result(self):
        if self.image_array != []:
            self.image_array = self.image_array.astype('float32')
            self.image_array /= 255

            X_pred = np.reshape(
                self.image_array, (1, self.image_array.shape[0], self.image_array.shape[1], 3))
            y_pred = self.model.predict(X_pred)
            y_pred = np.argmax(y_pred, axis=1)

            self.ui.label_3.setText("Predicted Class: " +
                                    (self.labels_dic[y_pred[0]]))

    def plot_graphs(self):
        graph_1 = cv2.imread("./stats/classification accuracy.jpeg")
        graph_1 = cv2.cvtColor(graph_1, cv2.COLOR_BGR2RGB)
        img = pg.ImageItem(graph_1)
        img.rotate(270)
        self.ui.graphicsView_2.clear()
        self.ui.graphicsView_2.addItem(img)

        graph_2 = cv2.imread("./stats/crossentropy loss.jpeg")
        graph_2 = cv2.cvtColor(graph_2, cv2.COLOR_BGR2RGB)
        img = pg.ImageItem(graph_2)
        img.rotate(270)
        self.ui.graphicsView_3.clear()
        self.ui.graphicsView_3.addItem(img)

    def checkClicked(self):
        self.isClicked = True

    def stats(self):
        if self.ui.comboBox.currentText() == "Accuracy":
            Accuracy_Scores = np.load(
                self.model_to_run + 'Accuracy_Scores.npy', mmap_mode='r')
            arr = []
            for accuracy in (Accuracy_Scores):
                arr.append(accuracy)
            self.show_accuracy_precision_recall_f1(arr)

        if self.ui.comboBox.currentText() == "Precision":
            Precision_Scores = np.load(
                self.model_to_run + 'Precision_Scores.npy', mmap_mode='r')
            arr = []
            for precision in (Precision_Scores):
                arr.append(precision)
            self.show_accuracy_precision_recall_f1(arr)

        if self.ui.comboBox.currentText() == "Recall":
            Recall_Scores = np.load(
                self.model_to_run + 'Recall_Scores.npy', mmap_mode='r')
            arr = []
            for recall in (Recall_Scores):
                arr.append(recall)
            self.show_accuracy_precision_recall_f1(arr)

        if self.ui.comboBox.currentText() == "F1":
            F1_Scores = np.load(self.model_to_run +
                                'F1_Scores.npy', mmap_mode='r')
            arr = []
            for f1 in (F1_Scores):
                arr.append(f1)
            self.show_accuracy_precision_recall_f1(arr)

    def show_accuracy_precision_recall_f1(self, scores):
        self.ui.tableWidget.setRowCount(len(scores))

        for index, score in enumerate(scores):
            self.ui.tableWidget.setItem(
                index, 0, QtWidgets.QTableWidgetItem(self.labels_dic[index]))
            self.ui.tableWidget.setItem(
                index, 1, QtWidgets.QTableWidgetItem(str(format(score*100, ".2f"))))


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()

    app.exec_()


if __name__ == "__main__":
    main()


# self.graphicsView.hideAxis('bottom')
# self.graphicsView.hideAxis('left')
# self.graphicsView.setBackground('w')
# self.graphicsView_2.hideAxis('bottom')
# self.graphicsView_2.hideAxis('left')
# self.graphicsView_2.setBackground('w')
# self.graphicsView_3.hideAxis('bottom')
# self.graphicsView_3.hideAxis('left')
# self.graphicsView_3.setBackground('w')
# item.setBackground(QtGui.QColor(36, 36, 36))
