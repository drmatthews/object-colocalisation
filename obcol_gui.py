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
        MainWindow.resize(1066, 839)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 1051, 691))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.movie_tab = QtGui.QWidget()
        self.movie_tab.setObjectName(_fromUtf8("movie_tab"))
        self.movie_groupbox = QtGui.QGroupBox(self.movie_tab)
        self.movie_groupbox.setGeometry(QtCore.QRect(20, 40, 512, 512))
        self.movie_groupbox.setTitle(_fromUtf8(""))
        self.movie_groupbox.setObjectName(_fromUtf8("movie_groupbox"))
        self.horizontalLayoutWidget = QtGui.QWidget(self.movie_tab)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 600, 511, 51))
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
        self.verticalLayoutWidget = QtGui.QWidget(self.movie_tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(550, 20, 491, 571))
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
        self.horizontalLayout_15 = QtGui.QHBoxLayout()
        self.horizontalLayout_15.setObjectName(_fromUtf8("horizontalLayout_15"))
        self.label_14 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.horizontalLayout_15.addWidget(self.label_14)
        self.min_size_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.min_size_spinbox.setMaximum(100000)
        self.min_size_spinbox.setProperty("value", 1)
        self.min_size_spinbox.setObjectName(_fromUtf8("min_size_spinbox"))
        self.horizontalLayout_15.addWidget(self.min_size_spinbox)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem2)
        self.label_15 = QtGui.QLabel(self.verticalLayoutWidget)
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.horizontalLayout_15.addWidget(self.label_15)
        self.max_size_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget)
        self.max_size_spinbox.setMaximum(100000)
        self.max_size_spinbox.setProperty("value", 100000)
        self.max_size_spinbox.setObjectName(_fromUtf8("max_size_spinbox"))
        self.horizontalLayout_15.addWidget(self.max_size_spinbox)
        self.verticalLayout.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem3)
        self.run_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.run_button.setObjectName(_fromUtf8("run_button"))
        self.horizontalLayout_9.addWidget(self.run_button)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.layoutWidget = QtGui.QWidget(self.movie_tab)
        self.layoutWidget.setGeometry(QtCore.QRect(850, 590, 181, 51))
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
        self.movie_radio.raise_()
        self.result_radio.raise_()
        self.horizontalLayout_10.addWidget(self.radio_groupbox)
        self.layoutWidget1 = QtGui.QWidget(self.movie_tab)
        self.layoutWidget1.setGeometry(QtCore.QRect(540, 600, 141, 51))
        self.layoutWidget1.setObjectName(_fromUtf8("layoutWidget1"))
        self.horizontalLayout_16 = QtGui.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_16.setObjectName(_fromUtf8("horizontalLayout_16"))
        self.label_7 = QtGui.QLabel(self.layoutWidget1)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout_16.addWidget(self.label_7)
        self.channel_spinbox = QtGui.QSpinBox(self.layoutWidget1)
        self.channel_spinbox.setObjectName(_fromUtf8("channel_spinbox"))
        self.horizontalLayout_16.addWidget(self.channel_spinbox)
        self.tabWidget.addTab(self.movie_tab, _fromUtf8(""))
        self.colocalisation_tab = QtGui.QWidget()
        self.colocalisation_tab.setObjectName(_fromUtf8("colocalisation_tab"))
        self.coloc_groupbox = QtGui.QGroupBox(self.colocalisation_tab)
        self.coloc_groupbox.setGeometry(QtCore.QRect(20, 40, 512, 512))
        self.coloc_groupbox.setTitle(_fromUtf8(""))
        self.coloc_groupbox.setObjectName(_fromUtf8("coloc_groupbox"))
        self.verticalLayoutWidget_2 = QtGui.QWidget(self.colocalisation_tab)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(540, 20, 501, 631))
        self.verticalLayoutWidget_2.setObjectName(_fromUtf8("verticalLayoutWidget_2"))
        self.results_groupbox = QtGui.QVBoxLayout(self.verticalLayoutWidget_2)
        self.results_groupbox.setObjectName(_fromUtf8("results_groupbox"))
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_8 = QtGui.QLabel(self.verticalLayoutWidget_2)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.horizontalLayout_7.addWidget(self.label_8)
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem4)
        self.overlap_spinbox = QtGui.QSpinBox(self.verticalLayoutWidget_2)
        self.overlap_spinbox.setMaximum(100)
        self.overlap_spinbox.setProperty("value", 0)
        self.overlap_spinbox.setObjectName(_fromUtf8("overlap_spinbox"))
        self.horizontalLayout_7.addWidget(self.overlap_spinbox)
        self.results_groupbox.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_13 = QtGui.QHBoxLayout()
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.label_10 = QtGui.QLabel(self.verticalLayoutWidget_2)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.horizontalLayout_13.addWidget(self.label_10)
        self.min_size_spinbox_2 = QtGui.QSpinBox(self.verticalLayoutWidget_2)
        self.min_size_spinbox_2.setMaximum(100000)
        self.min_size_spinbox_2.setProperty("value", 1)
        self.min_size_spinbox_2.setObjectName(_fromUtf8("min_size_spinbox_2"))
        self.horizontalLayout_13.addWidget(self.min_size_spinbox_2)
        self.label_11 = QtGui.QLabel(self.verticalLayoutWidget_2)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.horizontalLayout_13.addWidget(self.label_11)
        self.max_size_spinbox_2 = QtGui.QSpinBox(self.verticalLayoutWidget_2)
        self.max_size_spinbox_2.setMaximum(100000)
        self.max_size_spinbox_2.setProperty("value", 100000)
        self.max_size_spinbox_2.setObjectName(_fromUtf8("max_size_spinbox_2"))
        self.horizontalLayout_13.addWidget(self.max_size_spinbox_2)
        self.results_groupbox.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem5)
        self.update_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.update_button.setObjectName(_fromUtf8("update_button"))
        self.horizontalLayout_11.addWidget(self.update_button)
        self.reset_button = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.reset_button.setObjectName(_fromUtf8("reset_button"))
        self.horizontalLayout_11.addWidget(self.reset_button)
        self.results_groupbox.addLayout(self.horizontalLayout_11)
        self.histogram_groupbox = QtGui.QGroupBox(self.verticalLayoutWidget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.histogram_groupbox.sizePolicy().hasHeightForWidth())
        self.histogram_groupbox.setSizePolicy(sizePolicy)
        self.histogram_groupbox.setTitle(_fromUtf8(""))
        self.histogram_groupbox.setObjectName(_fromUtf8("histogram_groupbox"))
        self.results_groupbox.addWidget(self.histogram_groupbox)
        self.horizontalLayout_14 = QtGui.QHBoxLayout()
        self.horizontalLayout_14.setObjectName(_fromUtf8("horizontalLayout_14"))
        self.label_12 = QtGui.QLabel(self.verticalLayoutWidget_2)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.horizontalLayout_14.addWidget(self.label_12)
        self.channel_combobox = QtGui.QComboBox(self.verticalLayoutWidget_2)
        self.channel_combobox.setObjectName(_fromUtf8("channel_combobox"))
        self.channel_combobox.addItem(_fromUtf8(""))
        self.channel_combobox.addItem(_fromUtf8(""))
        self.horizontalLayout_14.addWidget(self.channel_combobox)
        spacerItem6 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem6)
        self.label_13 = QtGui.QLabel(self.verticalLayoutWidget_2)
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.horizontalLayout_14.addWidget(self.label_13)
        self.data_combobox = QtGui.QComboBox(self.verticalLayoutWidget_2)
        self.data_combobox.setObjectName(_fromUtf8("data_combobox"))
        self.data_combobox.addItem(_fromUtf8(""))
        self.data_combobox.addItem(_fromUtf8(""))
        self.data_combobox.addItem(_fromUtf8(""))
        self.horizontalLayout_14.addWidget(self.data_combobox)
        self.results_groupbox.addLayout(self.horizontalLayout_14)
        self.horizontalLayoutWidget_4 = QtGui.QWidget(self.colocalisation_tab)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(20, 600, 511, 51))
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
        self.analysis_tab = QtGui.QWidget()
        self.analysis_tab.setObjectName(_fromUtf8("analysis_tab"))
        self.verticalLayoutWidget_3 = QtGui.QWidget(self.analysis_tab)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(540, 20, 501, 631))
        self.verticalLayoutWidget_3.setObjectName(_fromUtf8("verticalLayoutWidget_3"))
        self.analysis_groupbox = QtGui.QVBoxLayout(self.verticalLayoutWidget_3)
        self.analysis_groupbox.setObjectName(_fromUtf8("analysis_groupbox"))
        self.horizontalLayout_17 = QtGui.QHBoxLayout()
        self.horizontalLayout_17.setObjectName(_fromUtf8("horizontalLayout_17"))
        spacerItem7 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem7)
        self.import_button = QtGui.QPushButton(self.verticalLayoutWidget_3)
        self.import_button.setObjectName(_fromUtf8("import_button"))
        self.horizontalLayout_17.addWidget(self.import_button)
        self.analysis_groupbox.addLayout(self.horizontalLayout_17)
        self.import_list_widget = QtGui.QListWidget(self.verticalLayoutWidget_3)
        self.import_list_widget.setObjectName(_fromUtf8("import_list_widget"))
        self.analysis_groupbox.addWidget(self.import_list_widget)
        self.horizontalLayout_18 = QtGui.QHBoxLayout()
        self.horizontalLayout_18.setObjectName(_fromUtf8("horizontalLayout_18"))
        self.add_button = QtGui.QPushButton(self.verticalLayoutWidget_3)
        self.add_button.setObjectName(_fromUtf8("add_button"))
        self.horizontalLayout_18.addWidget(self.add_button)
        self.remove_button = QtGui.QPushButton(self.verticalLayoutWidget_3)
        self.remove_button.setObjectName(_fromUtf8("remove_button"))
        self.horizontalLayout_18.addWidget(self.remove_button)
        spacerItem8 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem8)
        self.analysis_groupbox.addLayout(self.horizontalLayout_18)
        spacerItem9 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.analysis_groupbox.addItem(spacerItem9)
        self.analysis_plot_groupbox = QtGui.QGroupBox(self.analysis_tab)
        self.analysis_plot_groupbox.setGeometry(QtCore.QRect(20, 20, 501, 461))
        self.analysis_plot_groupbox.setTitle(_fromUtf8(""))
        self.analysis_plot_groupbox.setObjectName(_fromUtf8("analysis_plot_groupbox"))
        self.horizontalLayoutWidget_3 = QtGui.QWidget(self.analysis_tab)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(0, 570, 531, 80))
        self.horizontalLayoutWidget_3.setObjectName(_fromUtf8("horizontalLayoutWidget_3"))
        self.horizontalLayout_19 = QtGui.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_19.setObjectName(_fromUtf8("horizontalLayout_19"))
        self.label_17 = QtGui.QLabel(self.horizontalLayoutWidget_3)
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.horizontalLayout_19.addWidget(self.label_17)
        self.analysis_data_combobox = QtGui.QComboBox(self.horizontalLayoutWidget_3)
        self.analysis_data_combobox.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContents)
        self.analysis_data_combobox.setObjectName(_fromUtf8("analysis_data_combobox"))
        self.analysis_data_combobox.addItem(_fromUtf8(""))
        self.analysis_data_combobox.addItem(_fromUtf8(""))
        self.analysis_data_combobox.addItem(_fromUtf8(""))
        self.horizontalLayout_19.addWidget(self.analysis_data_combobox)
        spacerItem10 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_19.addItem(spacerItem10)
        self.tabWidget.addTab(self.analysis_tab, _fromUtf8(""))
        self.horizontalLayoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 740, 1051, 41))
        self.horizontalLayoutWidget_2.setObjectName(_fromUtf8("horizontalLayoutWidget_2"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem11 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem11)
        self.open_button = QtGui.QPushButton(self.horizontalLayoutWidget_2)
        self.open_button.setObjectName(_fromUtf8("open_button"))
        self.horizontalLayout_2.addWidget(self.open_button)
        self.save_button = QtGui.QPushButton(self.horizontalLayoutWidget_2)
        self.save_button.setObjectName(_fromUtf8("save_button"))
        self.horizontalLayout_2.addWidget(self.save_button)
        self.quit_button = QtGui.QPushButton(self.horizontalLayoutWidget_2)
        self.quit_button.setObjectName(_fromUtf8("quit_button"))
        self.horizontalLayout_2.addWidget(self.quit_button)
        self.progress_bar = QtGui.QProgressBar(self.centralwidget)
        self.progress_bar.setGeometry(QtCore.QRect(10, 710, 1041, 23))
        self.progress_bar.setProperty("value", 0)
        self.progress_bar.setObjectName(_fromUtf8("progress_bar"))
        self.horizontalLayoutWidget_2.raise_()
        self.progress_bar.raise_()
        self.tabWidget.raise_()
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
        self.label_2.setText(_translate("MainWindow", "Min:     ", None))
        self.label_3.setText(_translate("MainWindow", "Max:    ", None))
        self.label_5.setText(_translate("MainWindow", "Threshold:", None))
        self.label_4.setText(_translate("MainWindow", "Red channel:", None))
        self.label_6.setText(_translate("MainWindow", "Green channel:", None))
        self.label_14.setText(_translate("MainWindow", "Min size (pixels):", None))
        self.label_15.setText(_translate("MainWindow", "Max size (pixels):", None))
        self.run_button.setText(_translate("MainWindow", "Run now", None))
        self.result_radio.setText(_translate("MainWindow", "Result", None))
        self.movie_radio.setText(_translate("MainWindow", "Movie", None))
        self.label_7.setText(_translate("MainWindow", "Channel:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.movie_tab), _translate("MainWindow", "Movie", None))
        self.label_8.setText(_translate("MainWindow", "Overlap:", None))
        self.label_10.setText(_translate("MainWindow", "Min size:", None))
        self.label_11.setText(_translate("MainWindow", "Max size:", None))
        self.update_button.setText(_translate("MainWindow", "Update", None))
        self.reset_button.setText(_translate("MainWindow", "Reset", None))
        self.label_12.setText(_translate("MainWindow", "Channel:", None))
        self.channel_combobox.setItemText(0, _translate("MainWindow", "Red", None))
        self.channel_combobox.setItemText(1, _translate("MainWindow", "Green", None))
        self.label_13.setText(_translate("MainWindow", "Parameter:", None))
        self.data_combobox.setItemText(0, _translate("MainWindow", "Size of overlap (pixels)", None))
        self.data_combobox.setItemText(1, _translate("MainWindow", "Size of overlap (%)", None))
        self.data_combobox.setItemText(2, _translate("MainWindow", "Signal in overlap", None))
        self.label_9.setText(_translate("MainWindow", "Frame:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.colocalisation_tab), _translate("MainWindow", "Colocalisation", None))
        self.import_button.setText(_translate("MainWindow", "Import", None))
        self.add_button.setText(_translate("MainWindow", "Add", None))
        self.remove_button.setText(_translate("MainWindow", "Remove", None))
        self.label_17.setText(_translate("MainWindow", "Parameter:", None))
        self.analysis_data_combobox.setItemText(0, _translate("MainWindow", "Fraction of objects overlapped", None))
        self.analysis_data_combobox.setItemText(1, _translate("MainWindow", "Ave. Size of overlapped regions", None))
        self.analysis_data_combobox.setItemText(2, _translate("MainWindow", "Ave. Signal in overlapped regions", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.analysis_tab), _translate("MainWindow", "Analysis", None))
        self.open_button.setText(_translate("MainWindow", "Open", None))
        self.save_button.setText(_translate("MainWindow", "Save", None))
        self.quit_button.setText(_translate("MainWindow", "Quit", None))

