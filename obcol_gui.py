# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'obcol.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1066, 802)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 1051, 641))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.movie_tab = QtGui.QWidget()
        self.movie_tab.setObjectName(_fromUtf8("movie_tab"))
        self.movie_groupbox = QtGui.QGroupBox(self.movie_tab)
        self.movie_groupbox.setGeometry(QtCore.QRect(50, 19, 512, 512))
        self.movie_groupbox.setTitle(_fromUtf8(""))
        self.movie_groupbox.setObjectName(_fromUtf8("movie_groupbox"))
        self.horizontalLayoutWidget = QtGui.QWidget(self.movie_tab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 540, 551, 51))
        self.horizontalLayoutWidget.setObjectName(_fromUtf8("horizontalLayoutWidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.frame_slider = QtGui.QSlider(self.horizontalLayoutWidget)
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_slider.setObjectName(_fromUtf8("frame_slider"))
        self.horizontalLayout.addWidget(self.frame_slider)
        self.label_7 = QtGui.QLabel(self.horizontalLayoutWidget)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout.addWidget(self.label_7)
        self.channel_spinbox = QtGui.QSpinBox(self.horizontalLayoutWidget)
        self.channel_spinbox.setObjectName(_fromUtf8("channel_spinbox"))
        self.horizontalLayout.addWidget(self.channel_spinbox)
        self.verticalLayoutWidget = QtGui.QWidget(self.movie_tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(600, 20, 431, 501))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_2 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_3.addWidget(self.label_2)
        self.fmin_slider = QtGui.QSlider(self.verticalLayoutWidget)
        self.fmin_slider.setOrientation(QtCore.Qt.Horizontal)
        self.fmin_slider.setObjectName(_fromUtf8("fmin_slider"))
        self.horizontalLayout_3.addWidget(self.fmin_slider)
        self.fmin_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.fmin_spinbox.setObjectName(_fromUtf8("fmin_spinbox"))
        self.horizontalLayout_3.addWidget(self.fmin_spinbox)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_3 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_4.addWidget(self.label_3)
        self.fmax_slider = QtGui.QSlider(self.verticalLayoutWidget)
        self.fmax_slider.setOrientation(QtCore.Qt.Horizontal)
        self.fmax_slider.setObjectName(_fromUtf8("fmax_slider"))
        self.horizontalLayout_4.addWidget(self.fmax_slider)
        self.fmax_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.fmax_spinbox.setObjectName(_fromUtf8("fmax_spinbox"))
        self.horizontalLayout_4.addWidget(self.fmax_spinbox)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.label_5 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.horizontalLayout_8.addWidget(self.label_5)
        self.threshold_slider = QtGui.QSlider(self.verticalLayoutWidget)
        self.threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.threshold_slider.setObjectName(_fromUtf8("threshold_slider"))
        self.horizontalLayout_8.addWidget(self.threshold_slider)
        self.threshold_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.threshold_spinbox.setObjectName(_fromUtf8("threshold_spinbox"))
        self.horizontalLayout_8.addWidget(self.threshold_spinbox)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_4 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_5.addWidget(self.label_4)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.red_channel_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.red_channel_spinbox.setObjectName(_fromUtf8("red_channel_spinbox"))
        self.horizontalLayout_5.addWidget(self.red_channel_spinbox)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_6 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout_6.addWidget(self.label_6)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.green_channel_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.green_channel_spinbox.setObjectName(_fromUtf8("green_channel_spinbox"))
        self.horizontalLayout_6.addWidget(self.green_channel_spinbox)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem2)
        self.run_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.run_button.setObjectName(_fromUtf8("run_button"))
        self.horizontalLayout_9.addWidget(self.run_button)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.layoutWidget = QtGui.QWidget(self.movie_tab)
        self.layoutWidget.setGeometry(QtCore.QRect(600, 540, 429, 57))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.horizontalLayout_10 = QtGui.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.radio_groupbox = QtGui.QGroupBox(self.layoutWidget)
        self.radio_groupbox.setTitle(_fromUtf8(""))
        self.radio_groupbox.setObjectName(_fromUtf8("radio_groupbox"))
        self.result_radio = QtGui.QRadioButton(self.radio_groupbox)
        self.result_radio.setGeometry(QtCore.QRect(90, 10, 75, 22))
        self.result_radio.setObjectName(_fromUtf8("result_radio"))
        self.movie_radio = QtGui.QRadioButton(self.radio_groupbox)
        self.movie_radio.setGeometry(QtCore.QRect(10, 10, 73, 22))
        self.movie_radio.setChecked(True)
        self.movie_radio.setObjectName(_fromUtf8("movie_radio"))
        self.horizontalLayout_10.addWidget(self.radio_groupbox)
        self.tabWidget.addTab(self.movie_tab, _fromUtf8(""))
        self.colocalisation_tab = QtGui.QWidget()
        self.colocalisation_tab.setObjectName(_fromUtf8("colocalisation_tab"))
        self.coloc_groupbox = QtGui.QGroupBox(self.colocalisation_tab)
        self.coloc_groupbox.setGeometry(QtCore.QRect(50, 20, 512, 512))
        self.coloc_groupbox.setTitle(_fromUtf8(""))
        self.coloc_groupbox.setObjectName(_fromUtf8("coloc_groupbox"))
        self.verticalLayoutWidget_2 = QtGui.QWidget(self.colocalisation_tab)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(600, 20, 431, 501))
        self.verticalLayoutWidget_2.setObjectName(_fromUtf8("verticalLayoutWidget_2"))
        self.results_groupbox = QtGui.QVBoxLayout(self.verticalLayoutWidget_2)
        self.results_groupbox.setObjectName(_fromUtf8("results_groupbox"))
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_8 = QtGui.QLabel(self.verticalLayoutWidget_2)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.horizontalLayout_7.addWidget(self.label_8)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem3)
        self.overlap_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget_2)
        self.overlap_spinbox.setMaximum(100)
        self.overlap_spinbox.setProperty("value", 0)
        self.overlap_spinbox.setObjectName(_fromUtf8("overlap_spinbox"))
        self.horizontalLayout_7.addWidget(self.overlap_spinbox)
        self.results_groupbox.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem4)
        self.update_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.update_button.setObjectName(_fromUtf8("update_button"))
        self.horizontalLayout_11.addWidget(self.update_button)
        self.reset_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.reset_button.setObjectName(_fromUtf8("reset_button"))
        self.horizontalLayout_11.addWidget(self.reset_button)
        self.results_groupbox.addLayout(self.horizontalLayout_11)
        self.results_table = QtGui.QTableWidget(self.verticalLayoutWidget_2)
        self.results_table.setRowCount(1)
        self.results_table.setColumnCount(3)
        self.results_table.setObjectName(_fromUtf8("results_table"))
        self.results_groupbox.addWidget(self.results_table)
        self.horizontalLayoutWidget_4 = QtGui.QWidget(self.colocalisation_tab)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(30, 540, 421, 51))
        self.horizontalLayoutWidget_4.setObjectName(_fromUtf8("horizontalLayoutWidget_4"))
        self.horizontalLayout_12 = QtGui.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.label_9 = QtGui.QLabel(self.horizontalLayoutWidget_4)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_12.addWidget(self.label_9)
        self.coloc_frame_slider = QtGui.QSlider(self.horizontalLayoutWidget_4)
        self.coloc_frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.coloc_frame_slider.setObjectName(_fromUtf8("coloc_frame_slider"))
        self.horizontalLayout_12.addWidget(self.coloc_frame_slider)
        self.tabWidget.addTab(self.colocalisation_tab, _fromUtf8(""))
        self.horizontalLayoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 710, 1051, 41))
        self.horizontalLayoutWidget_2.setObjectName(_fromUtf8("horizontalLayoutWidget_2"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.open_button = QtGui.QPushButton(self.horizontalLayoutWidget_2)
        self.open_button.setObjectName(_fromUtf8("open_button"))
        self.horizontalLayout_2.addWidget(self.open_button)
        self.quit_button = QtGui.QPushButton(self.horizontalLayoutWidget_2)
        self.quit_button.setObjectName(_fromUtf8("quit_button"))
        self.horizontalLayout_2.addWidget(self.quit_button)
        self.progress_bar = QtGui.QProgressBar(self.centralwidget)
        self.progress_bar.setGeometry(QtCore.QRect(10, 670, 1041, 23))
        self.progress_bar.setProperty("value", 0)
        self.progress_bar.setObjectName(_fromUtf8("progress_bar"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1066, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Frame:", None))
        self.label_7.setText(_translate("MainWindow", "Channel:", None))
        self.label_2.setText(_translate("MainWindow", "Min:     ", None))
        self.label_3.setText(_translate("MainWindow", "Max:    ", None))
        self.label_5.setText(_translate("MainWindow", "Threshold:", None))
        self.label_4.setText(_translate("MainWindow", "Red channel:", None))
        self.label_6.setText(_translate("MainWindow", "Green channel:", None))
        self.run_button.setText(_translate("MainWindow", "Run now", None))
        self.result_radio.setText(_translate("MainWindow", "Result", None))
        self.movie_radio.setText(_translate("MainWindow", "Movie", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.movie_tab), _translate("MainWindow", "Movie", None))
        self.label_8.setText(_translate("MainWindow", "Overlap:", None))
        self.update_button.setText(_translate("MainWindow", "Update", None))
        self.reset_button.setText(_translate("MainWindow", "Reset", None))
        self.label_9.setText(_translate("MainWindow", "Frame:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.colocalisation_tab), _translate("MainWindow", "Colocalisation", None))
        self.open_button.setText(_translate("MainWindow", "Open", None))
        self.quit_button.setText(_translate("MainWindow", "Quit", None))

