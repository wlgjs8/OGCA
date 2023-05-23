import os
import sys
import cv2
import numpy as np

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QLabel, QDesktopWidget, QPushButton, QFileDialog, QMessageBox, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from gui_util import inference

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set Window
        # x, y, w, h
        self.setWindowTitle('Salient Object Detection')
        self.resize(1080, 1200)
        self.center()

        # File Button
        self.upload_button = QPushButton('Upload', self)
        self.upload_button.clicked.connect(self.upload)
        self.upload_button.setGeometry(25, 1110, 325, 50)

        self.process_button = QPushButton('Process', self)
        self.process_button.clicked.connect(self.process)
        self.process_button.setGeometry(375, 1110, 330, 50)

        self.clear_button = QPushButton('Clear', self)
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setGeometry(730, 1110, 325, 50)

        self.original_label = QLabel(self)
        self.original_label.move(25, 30)
        
        self.front_label = QLabel(self)
        self.front_label.move(555, 30)

        self.mask_label = QLabel(self)
        self.mask_label.move(25, 560)
        
        self.silhouette_label = QLabel(self)
        self.silhouette_label.move(555, 560)

        self.file_name = None
        self.original_image = None
        self.IMAGE_EXTENSIONS = ['jpg', 'png', 'jpeg', 'PNG', 'BMP', 'bmp', 'ppm', 'PPM']

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def upload(self):
        # self.file_name = getOpenFilesAndDirs()
        self.file_name = QFileDialog.getOpenFileName(self, 'Open file', './')[0]
        print('self.file_name : ', self.file_name)

        self.open_file(self.file_name)
        print('upload')
    
    def process(self):
        print('process')

        single_channel_mask = inference(self.file_name)
        mask_file_name = '../result.png'
        mask_output = self.convert_cv_qt(mask_file_name)

        self.mask_label.setPixmap(mask_output)
        self.mask_label.setContentsMargins(10, 10, 10, 10)
        self.mask_label.resize(500, 500)

        front_file_name = '../front_result.png'
        compose_file_name = '../compose_result.png'
        background_file_name = '../gross.jpg'
        self.process_front_and_compose(single_channel_mask, front_file_name, compose_file_name, background_file_name)

        front_output = self.convert_cv_qt(front_file_name)
        self.front_label.setPixmap(front_output)
        self.front_label.setContentsMargins(10, 10, 10, 10)
        self.front_label.resize(500, 500)

        compose_output = self.convert_cv_qt(compose_file_name)
        self.silhouette_label.setPixmap(compose_output)
        self.silhouette_label.setContentsMargins(10, 10, 10, 10)
        self.silhouette_label.resize(500, 500)


    def process_front_and_compose(self, single_channel_mask, front_file_name, compose_file_name, background_file_name):
        original_image = cv2.imread(self.file_name)
        '''
        print('original_image : ', original_image.shape)
        print('single_channel_mask : ', single_channel_mask.shape)

        original_image :  (480, 640, 3)
        single_channel_mask :  (480, 640, 1)
        '''

        h, w, c = original_image.shape
        front_image = np.zeros((h, w, c), dtype="uint8")
        compose_image = np.zeros((h, w, c), dtype="uint8")
        background_image = cv2.imread(background_file_name)
        background_image= cv2.resize(background_image, (w, h))

        print('background_image : ', background_image.shape)

        forH, forW = single_channel_mask.shape

        for i in range(forH):
            for j in range(forW):
                if single_channel_mask[i][j] == 255.0:
                    for cc in range(c):
                        front_image[i][j][cc] = original_image[i][j][cc]
                        compose_image[i][j][cc] = original_image[i][j][cc]
                else:
                    for cc in range(c):
                        front_image[i][j][cc] = 0
                        compose_image[i][j][cc] = background_image[i][j][cc]

        # return front_image
        cv2.imwrite(front_file_name, front_image)
        cv2.imwrite(compose_file_name, compose_image)

    def clear(self):
        self.original_label.clear()
        self.front_label.clear()
        self.mask_label.clear()
        self.silhouette_label.clear()

    def open_file(self, file_name):
        if file_name:
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaledToWidth(500)
            width = pixmap.width()
            height = pixmap.height()
            self.original_label.setPixmap(pixmap)
            self.original_label.setContentsMargins(10, 10, 10, 10)
            self.original_label.resize(500, 500)

        else:
            QMessageBox.about(self, 'Warning', 'No File Selected')

    def convert_cv_qt(self, file_name):
        """Convert from an opencv image to QPixmap"""
        # image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        # image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        image = cv2.imread(file_name)
        image = cv2.resize(image, (500, 500))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h,w,c  = image.shape
        bytesPerLine = 3 * w
        image = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)

        # image = QtGui.QImage(image.data, w, h, w*c, QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(image)
        # return image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())