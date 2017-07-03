#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from PyQt4 import QtCore, QtGui

import obcol_gui as obcol_gui
import sys
import os
import numpy as np

# import matplotlib
# matplotlib.use("Qt4Agg") # must be called before .backends or .pylab
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from tifffile import imread, imsave
from utils import Movie
from threads import ColocWorker

os.environ["QT_API"] = "pyqt4"


class ColocView(QtGui.QWidget):
    def __init__(self, parent):
        super(ColocView, self).__init__(parent.ui.coloc_groupbox)
        self.parent = parent
        self.dpi = 100
        self.fig = Figure((512 / self.dpi, 512 / self.dpi), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(parent.ui.coloc_groupbox)
        self.ax = self.fig.add_subplot(111)
        self.mpl_toolbar = NavigationToolbar(
            self.canvas, parent.ui.coloc_groupbox, coordinates=False)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        parent.ui.coloc_groupbox.setLayout(vbox)

    def draw_frame(self, data):
        img = np.zeros((512, 512, 4), dtype=np.uint8)
        img[:, :, 0] = data[0]
        img[:, :, 1] = data[1]
        img[:, :, 2] = 255
        img[:, :, 3] = 255
        self.ax.imshow(img, interpolation='nearest')
        self.canvas.draw()


class MovieView(QtGui.QGraphicsView):

    # key_press = QtCore.pyqtSignal(object)
    mouse_press = QtCore.Signal(float, float, name='mouse_press')

    def __init__(self, parent):
        super(MovieView, self).__init__(parent.ui.movie_groupbox)
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

    def show_new_frame(self, frame, fmin, fmax, threshold=0.0):
        self.scene.clear()
        # process image
        # save image
        self.data = frame.copy()

        if frame.ndim == 2 and threshold == 0.0:
            # scale image.
            frame = 255.0 * ((frame - fmin) / (fmax - fmin))
            frame[(frame > 255.0)] = 255.0
            frame[(frame < 0.0)] = 0.0

            # convert to QImage.
            frame = frame.astype(np.uint8)
            h, w = frame.shape
            frame_RGB = np.zeros((
                frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            frame_RGB[:, :, 0] = frame
            frame_RGB[:, :, 1] = frame
            frame_RGB[:, :, 2] = frame
            frame_RGB[:, :, 3] = 255
        elif frame.ndim == 2 and threshold > 0.0:

            idx = np.where(frame > threshold)

            # scale image.
            frame = 255.0 * ((frame - fmin) / (fmax - fmin))
            frame[(frame > 255.0)] = 255.0
            frame[(frame < 0.0)] = 0.0

            r = frame
            r[idx] = 255
            g = frame
            b = frame
            # convert to QImage.
            frame = frame.astype(np.uint8)
            h, w = frame.shape
            frame_RGB = np.zeros((
                frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            frame_RGB[:, :, 0] = r
            frame_RGB[:, :, 1] = g
            frame_RGB[:, :, 2] = b
            frame_RGB[:, :, 3] = 255
        elif frame.ndim == 3:
            # convert to QImage.
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


class MainGUIWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.ui = obcol_gui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.movie_view = MovieView(self)
        self.coloc_view = ColocView(self)

        # parameters
        self.directory = ""
        self.curr_frame = 0
        self.old_f = 1
        self.curr_channel = 0
        self.is_movie = False
        self.is_result = False
        self.thresholds = None

        # ui conditions
        self.ui.channel_spinbox.setKeyboardTracking(False)
        self.ui.fmin_spinbox.setKeyboardTracking(False)
        self.ui.fmax_spinbox.setKeyboardTracking(False)

        # signals
        self.ui.open_button.clicked.connect(self.load_movie)
        self.ui.quit_button.clicked.connect(self.quit)
        self.ui.frame_slider.valueChanged.connect(self.handle_frame_slider)
        self.ui.channel_spinbox.valueChanged.connect(
            self.handle_channel_spinbox)
        self.ui.fmin_slider.valueChanged.connect(self.handle_fmin_slider)
        self.ui.fmax_slider.valueChanged.connect(self.handle_fmax_slider)
        self.ui.fmin_slider.sliderReleased.connect(self.update_display)
        self.ui.fmax_slider.sliderReleased.connect(self.update_display)
        self.ui.fmin_spinbox.valueChanged.connect(self.handle_fmin_spinbox)
        self.ui.fmax_spinbox.valueChanged.connect(self.handle_fmax_spinbox)
        self.ui.threshold_slider.sliderReleased.connect(self.handle_threshold_slider_released)
        self.ui.threshold_slider.valueChanged.connect(
            self.handle_threshold_slider)
        self.ui.threshold_spinbox.valueChanged.connect(
            self.handle_threshold_spinbox)
        self.ui.run_button.clicked.connect(self.run_colocalisation)
        self.ui.movie_radio.toggled.connect(self.handle_radio_state)
        self.ui.result_radio.toggled.connect(self.handle_radio_state)

        # threads
        self.coloc_worker = ColocWorker(self)
        self.coloc_worker.processing_frame.connect(self.update_progress)
#        self.coloc_worker.colocalisation_signal.connect(self.colocalisation_analyser)
        self.coloc_worker.label_signal.connect(self.colocalisation_plotter)

        self.ui.radio_groupbox.hide()

    # slots
    def handle_frame_slider(self, value):
        if self.is_movie or self.is_result:
            curr_f = value
            if curr_f > self.old_f:
                self.get_frame(1)
            else:
                self.get_frame(-1)
            self.old_f = curr_f

    def handle_channel_spinbox(self, value):
        self.curr_channel = value
        self.fmin = float(
            np.min(self.movie[self.curr_frame].img[self.curr_channel, :, :]))
        self.fmax = float(
            np.max(self.movie[self.curr_frame].img[self.curr_channel, :, :]))
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
                self.is_result = False
        elif radio.text() == "Result":
            if radio.isChecked():
                self.is_movie = False
                self.is_result = True

        self.display_frame()

    def handle_result_radio(self, value):
        if value:
            self.ui.movie_radio.setValue(False)
            self.is_movie = False
            self.is_result = True
            self.display_frame()

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
            self.movie = Movie(self.movie_path)
            if self.movie.ndim == 4:
                self.is_movie = True
                self.num_frames = self.movie.num_frames
                self.num_channels = self.movie.num_channels
                self.height = self.movie.height
                self.width = self.movie.width

                self.thresholds = [0.0 for c in range(self.num_channels)]
                self.ui.frame_slider.setMaximum(self.num_frames)
                self.ui.channel_spinbox.setMaximum(self.num_channels - 1)
                self.fmin = float(
                    np.min(self.movie[0].img[self.curr_channel, :, :]))
                self.fmax = float(
                    np.max(self.movie[0].img[self.curr_channel, :, :]))

                self.movie_dtype = self.movie.dtype
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
                self.get_frame(0)
                self.ui.progress_bar.setMaximum(self.num_frames)
            else:
                QtGui.QMessageBox.information(self, "Error",
                                              "Timelapse data only please")

    def run_colocalisation(self):
        red_chan = int(self.ui.red_channel_spinbox.value())
        green_chan = int(self.ui.green_channel_spinbox.value())
        channels = [red_chan, green_chan]
        thresholds = [
            self.thresholds[red_chan], self.thresholds[green_chan]]
        overlap = float(self.ui.overlap_spinbox.value()) / 100.0
        params = {}
        params['thresholds'] = thresholds
        params['channels'] = channels
        params['overlap'] = overlap
        self.coloc_worker.start_thread(self.movie, params)

    def colocalisation_plotter(self, result):
        self.result = result
        self.is_result = True
        self.is_movie = False
        self.ui.radio_groupbox.show()
        self.ui.result_radio.setChecked(True)
        self.ui.movie_radio.setChecked(False)
        self.get_frame(0)
        self.ui.progress_bar.setValue(0)

    def update_progress(self, frame_num):
        self.ui.progress_bar.setValue(frame_num)

    # controllers
    def display_frame(self):
        if self.is_movie:
            frame = (
                np.ascontiguousarray(
                    self.movie[self.curr_frame].img[self.curr_channel, :, :]))
            self.movie_view.show_new_frame(
                frame,
                self.fmin,
                self.fmax,
                self.thresholds[self.curr_channel])
        if self.is_result:
            frame = (
                np.ascontiguousarray(self.result[self.curr_frame, :, :, :]))
            self.movie_view.show_new_frame(frame, 0, 255)

    def get_frame(self, step):
        self.curr_frame += step
        if (self.curr_frame < 0):
            self.curr_frame = 0
        if (self.curr_frame >= self.num_frames):
            self.curr_frame = self.num_frames - 1
        if self.is_movie or self.is_result:
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
