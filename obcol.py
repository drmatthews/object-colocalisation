#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from PyQt4 import QtCore, QtGui

import obcol_gui as obcol_gui
import sys
import os
import numpy as np

from matplotlib import pyplot as plt
# matplotlib.use("Qt4Agg") # must be called before .backends or .pylab
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from tifffile import imread, imsave
import utils
from threads import ObcolWorker, SaveWorker

os.environ["QT_API"] = "pyqt4"


class HistogramView(QtGui.QWidget):
    def __init__(self, parent):
        super(HistogramView, self).__init__(parent.ui.histogram_groupbox)
        self.parent = parent
        dpi = 100.0
        self.fig = Figure((512.0 / dpi, 512.0 / dpi), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(parent.ui.histogram_groupbox)

        self.ax = self.fig.add_subplot(111)
        self.mpl_toolbar = NavigationToolbar(
            self.canvas, parent.ui.histogram_groupbox, coordinates=False)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)

        parent.ui.histogram_groupbox.setLayout(vbox)
        self.mpl_toolbar.hide()
        self.canvas.hide()

    def draw(self, data, data_type, channel):
        """ Draws the figure
        """
        self.canvas.show()
        self.mpl_toolbar.show()
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_ylabel('Count')
        if data_type == 0:  # sizes
            plot_data = data.sizes_of_patches(channel)
            self.ax.set_xlabel('Size of overlap [pixels]')
        if data_type == 1:  # fraction
            plot_data = data.fraction_of_patch_overlapped(channel)
            self.ax.set_xlabel('Size of overlap [%]')
        elif data_type == 2:  # signals
            plot_data = data.signals_of_patches(channel)
            self.ax.set_xlabel('Total grey levels in overlap region')
        self.ax.hist(plot_data, range=(1, max(plot_data)))
        self.fig.tight_layout()
        self.canvas.draw()

# class ResultsTable(QtGui.QWidget):
#     def __init__(self, parent, table):
#         super(ResultsTable, self).__init__(parent)
#         self.parent = parent
#         self.table = table
#         self.header = ["frame id",
#                        "red_coloc_fraction",
#                        "green_coloc_fraction"]

#     def set_data(self, data, channels):
#         for i, row in enumerate(data):
#             frame_item = QtGui.QTableWidgetItem(str(i))
#             red_item = QtGui.QTableWidgetItem(
#                 str(row.patches[channels[0]].fraction_with_overlap))
#             print(str(row.patches[channels[0]].fraction_with_overlap))
#             green_item = QtGui.QTableWidgetItem(
#                 str(row.patches[channels[1]].fraction_with_overlap))
#             self.table.insertRow(i + 1)
#             self.table.setItem(i, 0, frame_item)
#             self.table.setItem(i, 1, red_item)
#             self.table.setItem(i, 2, green_item)

#         self.table.resizeColumnsToContents()
#         self.table.resizeRowsToContents()
#         self.table.setHorizontalHeaderLabels(self.header)
#         self.table.show()


class MovieView(QtGui.QGraphicsView):

    # key_press = QtCore.pyqtSignal(object)
    mouse_press = QtCore.Signal(float, float, name='mouse_press')

    def __init__(self, parent, container):
        super(MovieView, self).__init__(container)
        self.parent = parent
        self.statusbar = parent.ui.statusbar
        # Class variables.
        self.data = False
        im = np.ones((512, 512, 4), dtype=np.uint8)
        self.image = QtGui.QImage(
            im.data,
            im.shape[1],
            im.shape[0],
            im.shape[1],
            QtGui.QImage.Format_Indexed8)
        self.margin = 128.0
        self.zoom_in = 1.2
        self.zoom_out = 1.0 / self.zoom_in
        self.origin = QtCore.QPoint()

        # UI initializiation.
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.MinimumExpanding,
            QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(512, 512))

        # Scene initialization.
        self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.scene = QtGui.QGraphicsScene()
        self.setScene(self.scene)
        self.setMouseTracking(True)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.scale(1.0, 1.0)

    def show_movie_frame(self, frame, fmin, fmax, threshold=0.0):
        self.scene.clear()
        # process image
        # save image
        self.data = frame.copy()
        # scale image.
        frame = 255.0 * ((frame - fmin) / (fmax - fmin))
        frame[(frame > 255.0)] = 255.0
        frame[(frame < 0.0)] = 0.0

        # and set type to uint8
        frame = frame.astype(np.uint8)
        # convert to RGB
        h, w = frame.shape
        frame_RGB = np.zeros((
            frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

        if threshold == 0.0:
            r = frame
            g = frame
            b = frame

        elif threshold > 0.0:

            idx = np.where(self.data > threshold)

            r = frame
            r[idx] = 255
            g = frame
            b = frame

        frame_RGB[:, :, 0] = r
        frame_RGB[:, :, 1] = g
        frame_RGB[:, :, 2] = b
        frame_RGB[:, :, 3] = 255

        # convert to QImage
        self.image = QtGui.QImage(
            frame_RGB.data, w, h, QtGui.QImage.Format_RGB32)
        self.image.ndarray1 = frame
        self.image.ndarray2 = frame_RGB

        # add to scene
        self.scene.addPixmap(QtGui.QPixmap.fromImage(self.image))

    def show_segmentation_frame(self, frame):
        self.scene.clear()
        # process image
        # save image
        self.data = frame.copy()

        red = frame[0, :, :].astype(np.uint8)
        green = frame[1, :, :].astype(np.uint8)
        h, w = red.shape
        frame_RGB = np.zeros((
            red.shape[0], red.shape[1], 4), dtype=np.uint8)
        frame_RGB[:, :, 0] = 0
        frame_RGB[:, :, 1] = green
        frame_RGB[:, :, 2] = red
        frame_RGB[:, :, 3] = 255

        self.image = QtGui.QImage(
            frame_RGB.data, w, h, QtGui.QImage.Format_RGB32)
        self.image.ndarray1 = frame
        self.image.ndarray2 = frame_RGB

        # add to scene
        self.scene.addPixmap(QtGui.QPixmap.fromImage(self.image))

    def wheelEvent(self, event):
        if event.delta() > 0:
            self.zoomIn()
        else:
            self.zoomOut()

    def zoomIn(self):
        self.scale(self.zoom_in, self.zoom_in)

    def zoomOut(self):
        self.scale(self.zoom_out, self.zoom_out)


class ColocView(MovieView):
    def __init__(self, parent, container):
        super(ColocView, self).__init__(parent, container)
        self.parent = parent


class MainGUIWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.ui = obcol_gui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.movie_view = MovieView(self, self.ui.movie_groupbox)
        self.coloc_view = ColocView(self, self.ui.coloc_groupbox)
        # self.table_widget = ResultsTable(self, self.ui.results_table)
        self.histogram_view = HistogramView(self)
        # self.table_widget.hide()

        # parameters
        self.directory = ""
        self.curr_frame = 0
        self.old_f = 1
        self.old_coloc_f = 1
        self.curr_channel = 0
        self.is_movie = False
        self.is_segmentation = False
        self.is_colocalisation = False
        self.thresholds = None

        # ui conditions
        self.ui.channel_spinbox.setKeyboardTracking(False)
        self.ui.fmin_spinbox.setKeyboardTracking(False)
        self.ui.fmax_spinbox.setKeyboardTracking(False)

        # signals
        self.ui.tabWidget.currentChanged.connect(self.handle_tab_change)
        self.ui.open_button.clicked.connect(self.load_movie)
        self.ui.save_button.clicked.connect(self.save)
        self.ui.quit_button.clicked.connect(self.quit)
        self.ui.frame_slider.valueChanged.connect(
            self.handle_frame_slider)
        self.ui.coloc_frame_slider.valueChanged.connect(
            self.handle_coloc_frame_slider)
        self.ui.channel_spinbox.valueChanged.connect(
            self.handle_channel_spinbox)
        self.ui.fmin_slider.valueChanged.connect(self.handle_fmin_slider)
        self.ui.fmax_slider.valueChanged.connect(self.handle_fmax_slider)
        self.ui.fmin_slider.sliderReleased.connect(self.update_display)
        self.ui.fmax_slider.sliderReleased.connect(self.update_display)
        self.ui.fmin_spinbox.valueChanged.connect(self.handle_fmin_spinbox)
        self.ui.fmax_spinbox.valueChanged.connect(self.handle_fmax_spinbox)
        self.ui.threshold_slider.sliderReleased.connect(
            self.handle_threshold_slider_released)
        self.ui.threshold_slider.valueChanged.connect(
            self.handle_threshold_slider)
        self.ui.threshold_spinbox.valueChanged.connect(
            self.handle_threshold_spinbox)
        self.ui.run_button.clicked.connect(self.start_obcol_thread)
        self.ui.movie_radio.toggled.connect(self.handle_radio_state)
        self.ui.result_radio.toggled.connect(self.handle_radio_state)
        self.ui.min_size_spinbox.valueChanged.connect(
            self.handle_min_size_spinbox)
        self.ui.max_size_spinbox.valueChanged.connect(
            self.handle_max_size_spinbox)
        self.ui.min_size_spinbox_2.valueChanged.connect(
            self.handle_min_size_spinbox)
        self.ui.max_size_spinbox_2.valueChanged.connect(
            self.handle_max_size_spinbox)
        self.ui.update_button.clicked.connect(
            self.start_filter_thread)
        self.ui.reset_button.clicked.connect(self.handle_colocalisation_reset)
        self.ui.channel_combobox.currentIndexChanged.connect(
            self.handle_channel_combo)
        self.ui.data_combobox.currentIndexChanged.connect(
            self.handle_data_combo)
        self.ui.radio_groupbox.hide()

        # threads
        self.obcol_worker = ObcolWorker(self)
        self.obcol_worker.progress_signal.connect(self.update_progress)
        self.save_worker = SaveWorker(self)

    # slots
    def handle_tab_change(self):
        self.curr_frame = 0
        self.ui.frame_slider.setValue(0)
        self.ui.coloc_frame_slider.setValue(0)

    def handle_frame_slider(self, value):
        # print("is_movie", self.is_movie)
        # print("is_segmentation", self.is_segmentation)
        if self.is_movie or self.is_segmentation:
            curr_f = value
            if curr_f > self.old_f:
                self.get_frame(1)
            else:
                self.get_frame(-1)
            self.old_f = curr_f

    def handle_coloc_frame_slider(self, value):
        if self.is_segmentation:
            curr_f = value
            if curr_f > self.old_coloc_f:
                self.get_frame(1)
            else:
                self.get_frame(-1)
            self.old_coloc_f = curr_f

    def handle_channel_spinbox(self, value):
        self.curr_channel = value
        frame = self.movie[self.curr_frame].get_image()
        self.fmin = float(
            np.min(frame[self.curr_channel, :, :]))
        self.fmax = float(
            np.max(frame[self.curr_channel, :, :]))
        self.ui.fmax_slider.setValue(self.fmax)
        self.ui.threshold_spinbox.setValue(self.thresholds[self.curr_channel])
        self.ui.threshold_slider.setValue(self.thresholds[self.curr_channel])
        self.update_display()

    def handle_fmin_slider(self, value):
        self.ui.fmin_spinbox.setValue(value)

    def handle_fmax_slider(self, value):
        self.ui.fmax_spinbox.setValue(value)

    def handle_fmin_spinbox(self, value):
        self.ui.fmin_slider.setValue(value)
        self.update_display()

    def handle_fmax_spinbox(self, value):
        self.ui.fmax_slider.setValue(value)
        self.update_display()

    def handle_threshold_slider(self, value):
        self.ui.threshold_spinbox.setValue(value)

    def handle_threshold_slider_released(self):
        self.thresholds[self.curr_channel] = float(
            self.ui.threshold_slider.value())
        self.update_display()

    def handle_threshold_spinbox(self, value):
        self.ui.threshold_slider.setValue(value)
        self.update_display()

    def handle_radio_state(self):
        radio = self.sender()
        if radio.text() == "Movie":
            if radio.isChecked():
                self.is_movie = True
                self.is_segmentation = False
        elif radio.text() == "Result":
            if radio.isChecked():
                self.is_movie = False
                self.is_segmentation = True

        self.display_frame()

    def handle_result_radio(self, value):
        if value:
            self.ui.movie_radio.setValue(False)
            self.is_movie = False
            self.is_segmentation = True
            self.display_frame()

    def handle_channel_combo(self, value):
        channel = self.params['channels'][value]
        data_type = self.ui.data_combobox.currentIndex()
        data = self.segmentation_result[self.curr_frame]
        self.histogram_view.draw(data, data_type, channel)

    def handle_min_size_spinbox(self, value):
        sender = self.sender().objectName()
        if "2" in sender:
            self.ui.min_size_spinbox.setValue(value)
        else:
            self.ui.min_size_spinbox_2.setValue(value)

    def handle_max_size_spinbox(self, value):
        sender = self.sender().objectName()
        if "2" in sender:
            self.ui.max_size_spinbox.setValue(value)
        else:
            self.ui.max_size_spinbox_2.setValue(value)

    def handle_data_combo(self, value):
        coloc_channel = self.ui.channel_combobox.currentIndex()
        channel = self.params['channels'][coloc_channel]
        data = self.segmentation_result[self.curr_frame]
        self.histogram_view.draw(data, value, channel)

    def update_display(self):
        self.fmin = float(self.ui.fmin_slider.value())
        self.fmax = float(self.ui.fmax_slider.value())
        self.display_frame()

    def load_movie(self):
        self.movie_path = (
            str(QtGui.QFileDialog.getOpenFileName(self,
                "Load Movie", self.directory, "*.tif")))
        if self.movie_path:
            self.directory = os.path.dirname(self.movie_path)
            movie_array = imread(self.movie_path)
            self.movie = utils.generate_frames(movie_array)
            if movie_array.ndim == 4:
                self.is_movie = True
                self.is_segmentation = False
                self.is_colocalisation = False

                self.num_frames = movie_array.shape[0]
                self.num_channels = movie_array.shape[1]
                self.height = movie_array.shape[2]
                self.width = movie_array.shape[3]

                self.thresholds = [0.0 for c in range(self.num_channels)]
                self.ui.frame_slider.setMaximum(self.num_frames)
                self.ui.coloc_frame_slider.setMaximum(self.num_frames)
                self.ui.channel_spinbox.setMaximum(self.num_channels - 1)
                first = movie_array[0, :, :, :]
                self.fmin = float(
                    np.min(first[self.curr_channel, :, :]))
                self.fmax = float(
                    np.max(first[self.curr_channel, :, :]))

                self.movie_dtype = movie_array.dtype
                if self.movie_dtype == np.uint16:
                    bit_depth = 2**16
                elif self.movie_dtype == np.uint8:
                    bit_depth = 2**8

                self.bit_depth = bit_depth

                self.ui.fmin_spinbox.setMaximum(bit_depth)
                self.ui.fmax_spinbox.setMaximum(bit_depth)
                self.ui.fmin_slider.setMaximum(bit_depth)
                self.ui.fmax_slider.setMaximum(bit_depth)
                self.ui.fmax_slider.setValue(self.fmax)
                self.ui.threshold_slider.setMaximum(self.fmax)
                self.ui.threshold_spinbox.setMaximum(bit_depth)
                self.ui.red_channel_spinbox.setMaximum(self.num_channels - 1)
                self.ui.green_channel_spinbox.setMaximum(self.num_channels - 1)
                self.ui.movie_radio.setChecked(True)
                self.ui.result_radio.setChecked(False)

                # reset everything
                self.curr_frame = 0
                self.old_f = 1
                self.old_coloc_f = 1
                self.curr_channel = 0
                self.segmentation_result = []
                self.colocalisation_result = []
                self.ui.channel_spinbox.setValue(0)
                self.ui.frame_slider.setValue(0)
                self.ui.coloc_frame_slider.setValue(0)
                self.ui.red_channel_spinbox.setValue(0)
                self.ui.green_channel_spinbox.setValue(0)

                self.get_frame(0)
                self.ui.progress_bar.setMaximum(self.num_frames)
            else:
                QtGui.QMessageBox.information(self, "Error",
                                              "Timelapse data only please")

    def save(self):
        if self.is_segmentation:
            self.save_worker.progress_signal.connect(self.update_progress)
            self.save_worker.finished_signal.connect(self.update_progress)
            self.save_worker.start_thread(
                self.segmentation_result,
                self.params['channels'],
                self.movie_path)

    def prepare_parameters(self, segment=True):
        red_chan = int(self.ui.red_channel_spinbox.value())
        green_chan = int(self.ui.green_channel_spinbox.value())
        channels = [red_chan, green_chan]
        thresholds = [
            self.thresholds[red_chan], self.thresholds[green_chan]]
        overlap = float(self.ui.overlap_spinbox.value())
        if overlap > 0.0:
            overlap = overlap / 100.0
        min_size = self.ui.min_size_spinbox.value()
        max_size = self.ui.max_size_spinbox.value()
        params = {}
        params['movie_path'] = self.movie_path
        params['thresholds'] = thresholds
        params['channels'] = channels
        params['overlap'] = overlap
        params['size_range'] = (min_size, max_size)
        params['segment'] = segment
        self.params = params
        return params

    def start_obcol_thread(self):
        self.is_filtered = False
        self.is_result = False
        self.is_movie = True
        self.segmentation_result = []
        self.colocalisation_result = []
        params = self.prepare_parameters()
        self.obcol_worker.results_signal.connect(
            self.obcol_plotter)
        self.obcol_worker.start_thread(self.movie, params)

    def obcol_plotter(self, result):
        self.segmentation_result = result
        self.is_segmentation = True
        self.is_movie = False
        self.ui.radio_groupbox.show()
        self.ui.result_radio.setChecked(True)
        self.ui.movie_radio.setChecked(False)
        self.ui.progress_bar.setValue(0)
        channel = self.params['channels'][0]
        self.histogram_view.draw(result[0], 0, channel)
        # self.table_widget.set_data(result, params['channels'])
        # self.table_widget.show()
        self.get_frame(0)

    def start_filter_thread(self):
        params = self.prepare_parameters(segment=False)
        self.ui.result_radio.setChecked(True)
        self.ui.movie_radio.setChecked(False)
        if params['overlap'] == 0.0:
            self.handle_colocalisation_reset()
        else:
            self.obcol_worker.results_signal.connect(
                self.filter_plotter)
            self.obcol_worker.start_thread(self.segmentation_result, params)

    def filter_plotter(self, result):
        self.is_colocalisation = True
        self.colocalisation_result = result
        self.ui.progress_bar.setValue(0)
        self.get_frame(0)

    def handle_colocalisation_reset(self):
        self.is_colocalisation = False
        self.is_segmentation = True
        self.ui.overlap_spinbox.setValue(0)
        self.get_frame(0)

    def update_progress(self, frame_num):
        self.ui.progress_bar.setValue(frame_num)

    # controllers
    def display_frame(self):
        if self.is_movie:
            current_frame = self.movie[self.curr_frame].get_image()
            frame = (
                np.ascontiguousarray(
                    current_frame[self.curr_channel, :, :]))
            self.movie_view.show_movie_frame(
                frame,
                self.fmin,
                self.fmax,
                self.thresholds[self.curr_channel])
        if self.is_segmentation:
            sr = self.segmentation_result[self.curr_frame]
            current_labels = (
                sr.get_mono_labels())
            frame = (
                np.ascontiguousarray(current_labels))
            self.movie_view.show_segmentation_frame(frame)
            self.coloc_view.show_segmentation_frame(frame)
        if self.is_colocalisation:
            cr = self.colocalisation_result[self.curr_frame]
            current_labels = (
                cr.get_mono_labels(filtered=True))
            filtered = (
                np.ascontiguousarray(current_labels))
            self.coloc_view.show_segmentation_frame(filtered)

    def get_frame(self, step):
        self.curr_frame += step
        if (self.curr_frame < 0):
            self.curr_frame = 0
        if (self.curr_frame >= self.num_frames):
            self.curr_frame = self.num_frames - 1
        if self.is_movie or self.is_segmentation:
            self.display_frame()

    def quit(self):
        self.close()


if __name__ == "__main__":

    app = QtGui.QApplication.instance()
    standalone = app is None
    if standalone:
        app = QtGui.QApplication(sys.argv)
    window = MainGUIWindow()
    window.show()
    if standalone:
        sys.exit(app.exec_())
    else:
        print "We're back with the Qt window still active"
