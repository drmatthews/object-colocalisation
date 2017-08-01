#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from PyQt4 import QtCore, QtGui

import obcol_gui as obcol_gui
import sys
import os
import numpy as np

from matplotlib import pyplot as plt
from itertools import cycle
# matplotlib.use("Qt4Agg") # must be called before .backends or .pylab
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from skimage.filters import gaussian

from tifffile import imread
import trackpy as tp
import utils
from threads import (ObcolWorker,
                     SaveWorker,
                     ImportWorker)

os.environ["QT_API"] = "pyqt4"


class Trajectory(object):
    def __init__(self, particle):
        self.pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        self.pen.setWidthF(1.0)
        self.trajectory = self._create_trajectory(particle)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.trajectory[key]
        elif isinstance(key, slice):
            return self.trajectory[key.start:key.stop:key.step]

    def _create_trajectory(self, particle):
        lines = []
        first = particle.iloc[0]
        x1 = first['x'].item()
        y1 = first['y'].item()
        for pid, p in particle.iterrows():
            x2 = p['x'].item()
            y2 = p['y'].item()
            line = QtGui.QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(self.pen)
            lines.append(line)
            x1 = p['x'].item()
            y1 = p['y'].item()
        return lines


class TrajectoryList(object):
    def __init__(self, tracks):
        self.trajectories = self._create_trajectory_items(tracks)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.trajectories[key]
        elif isinstance(key, slice):
            return self.trajectories[key.start:key.stop:key.step]

    def _create_trajectory_items(self, tracks):
        trajectories = []
        for tid, track in tracks.groupby('particle'):
            trajectories.append(Trajectory(track))
        return trajectories


class ImportList(QtGui.QWidget):

    def __init__(self, parent=None, list_widget=None):
        super(ImportList, self).__init__(parent)
        self.parent = parent
        self.list_widget = list_widget
        self.list_widget.setSelectionMode(
            QtGui.QAbstractItemView.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.connect(self.list_widget, QtCore.SIGNAL(
            "customContextMenuRequested(QPoint)"),
            self.item_right_clicked)
        self.list_widget.itemSelectionChanged.connect(self.items_selected)

    def update(self, import_list):
        if import_list:
            self.list_widget.data = import_list
            self.list_widget.clear()
            for val in import_list:
                self.list_widget.addItem(os.path.basename(str(val)))

    def items_selected(self):
        items = []
        for index in xrange(self.list_widget.count()):
            items.append(self.list_widget.item(index))
        labels = [i.text() for i in items]

        selected_labels = [si.text()
                           for si in self.list_widget.selectedItems()]

        indices = [labels.index(sl) for sl in selected_labels]

        selected = []
        channels = []
        for index in indices:
            selected.append(self.parent.analysis_data[index])
            channels.append(self.parent.analysis_channels[index])

        data_type = self.parent.ui.analysis_data_combobox.currentIndex()
        self.parent.analysis_view.draw(selected, data_type, channels)

    def add(self, new):
        self.list_widget.addItem(os.path.basename(str(new)))

    def remove(self, row):
        item = self.list_widget.takeItem(row)
        item = None

    def set_selected(self, row):
        self.list_widget.setCurrentRow(row)
        item = self.list_widget.item(row)
        item.setSelected(True)

    def item_right_clicked(self, qpos):
        self.list_menu = QtGui.QMenu()
        menu_item = self.list_menu.addAction("Copy")
        self.connect(menu_item,
                     QtCore.SIGNAL("triggered()"), self.menu_item_clicked)
        parent_position = self.list_widget.mapToGlobal(QtCore.QPoint(0, 0))
        self.list_menu.move(parent_position + qpos)
        self.list_menu.show()

    def menu_item_clicked(self):
        name = str(self.list_widget.currentItem().text())
        cb = QtGui.QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(name, mode=cb.Clipboard)


class AnalysisView(QtGui.QWidget):
    def __init__(self, parent):
        super(AnalysisView, self).__init__(parent.ui.analysis_plot_groupbox)
        self.parent = parent
        dpi = 100.0
        self.fig = Figure((512.0 / dpi, 512.0 / dpi), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(parent.ui.analysis_plot_groupbox)

        self.ax = self.fig.add_subplot(111)
        self.mpl_toolbar = NavigationToolbar(
            self.canvas, parent.ui.analysis_plot_groupbox, coordinates=False)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)

        parent.ui.analysis_plot_groupbox.setLayout(vbox)
        self.mpl_toolbar.hide()
        self.canvas.hide()

    def draw(self, data_list, data_type, channels_list):
        # note that channel can be int or list
        self.canvas.show()
        self.mpl_toolbar.show()
        self.ax.clear()
        self.ax.grid(True)
        lines = ["-", "--", "-.", ":"]
        linecycler = cycle(lines)
        colors = ['r', 'g']
        for d, data in enumerate(data_list):
            coloc_channels = channels_list[d]
            linestyle = next(linecycler)
            for c, chan in enumerate(colors):
                x_data = [i for i in range(len(data[coloc_channels[c]]))]
                y_data = []
                for frame in data[coloc_channels[c]]:
                    if data_type == 0:
                        y_data.append(frame.overlap_fraction)
                        self.ax.set_ylabel('Number')
                    elif data_type == 1:
                        y_data.append(frame.overlap_size)
                        self.ax.set_ylabel('Size')
                    elif data_type == 2:
                        y_data.append(frame.overlap_signal)
                        self.ax.set_ylabel('Signal')

                self.ax.set_xlabel('Frame number')
                self.ax.plot(
                    x_data, y_data, chan, linestyle=linestyle)

        self.fig.tight_layout()
        self.canvas.draw()


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
            self.ax.set_xlabel('Size of overlap [pixels]')
            plot_data = data.sizes_of_patches(channel)
            if all(v == 0.0 for v in plot_data):
                range_min = 0
            else:
                range_min = 1
        if data_type == 1:  # fraction
            plot_data = data.fraction_of_patch_overlapped(channel)
            self.ax.set_xlabel('Fraction of object with overlap')
            range_min = 0
        elif data_type == 2:  # signals
            plot_data = data.signals_of_patches(channel)
            self.ax.set_xlabel('Total grey levels in overlap region')
            if all(v == 0.0 for v in plot_data):
                range_min = 0
            else:
                range_min = 1
        self.ax.hist(plot_data, range=(range_min, max(plot_data)))
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

    def show_movie_frame(self, frame, fmin, fmax, sigma, threshold=0.0):
        self.scene.clear()
        # process image
        # save image
        self.data = frame.copy()
        if sigma > 0.0:
            self.data = gaussian(self.data.astype(float), sigma=sigma)
            frame = gaussian(frame.astype(float), sigma=sigma)
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

    def show_segmentation_frame(self, frame, trajectories=None):
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

        self.scene.addPixmap(QtGui.QPixmap.fromImage(self.image))

        if trajectories:
            self.add_trajectories(trajectories)

    def add_trajectories(self, trajectories):
        for traj in trajectories:
            for line in traj:
                self.scene.addItem(line)

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
        self.list_view = ImportList(self, self.ui.import_list_widget)
        self.histogram_view = HistogramView(self)
        self.analysis_view = AnalysisView(self)
        # self.table_widget.hide()

        # parameters
        self.basepath = ""
        self.import_directory = ""
        self.curr_frame = 0
        self.old_f = 1
        self.old_coloc_f = 1
        self.curr_channel = 0
        self.curr_seg_channel = 0
        self.is_movie = False
        self.is_segmentation = False
        self.is_colocalisation = False
        self.thresholds = None
        self.blur = None
        self.import_paths = []
        self.analysis_data = []
        self.analysis_channels = []
        self.tracks = None  # needs to be list or dict

        # ui conditions
        self.ui.channel_spinbox.setKeyboardTracking(False)
        self.ui.fmin_spinbox.setKeyboardTracking(False)
        self.ui.fmax_spinbox.setKeyboardTracking(False)

        # signals
        # self.ui.tabWidget.currentChanged.connect(self.handle_tab_change)
        self.ui.actionOpen.triggered.connect(self.load_movie)
        self.ui.open_button.clicked.connect(self.load_movie)
        self.ui.save_button.clicked.connect(self.save)
        self.ui.quit_button.clicked.connect(self.quit)
        self.ui.import_button.clicked.connect(self.import_patches)
        self.ui.frame_slider.installEventFilter(self)
        self.ui.frame_slider.valueChanged.connect(
            self.handle_frame_slider)
        self.ui.coloc_frame_slider.installEventFilter(self)
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
        self.ui.blur_slider.sliderReleased.connect(
            self.handle_blur_slider_released)
        self.ui.blur_slider.valueChanged.connect(
            self.handle_blur_slider)
        self.ui.blur_spinbox.valueChanged.connect(
            self.handle_blur_spinbox)
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
        # self.ui.analysis_channel_combobox.currentIndexChanged.connect(
        #     self.handle_analysis_channel_combo)
        self.ui.analysis_data_combobox.currentIndexChanged.connect(
            self.handle_analysis_data_combo)
        self.ui.add_button.clicked.connect(self.handle_add_button)
        self.ui.remove_button.clicked.connect(self.handle_remove_button)

        # threads
        self.obcol_worker = ObcolWorker(self)
        self.obcol_worker.results_signal.connect(self.obcol_plotter)
        self.obcol_worker.progress_signal.connect(self.update_progress)
        self.obcol_worker.progress_message.connect(self.update_progress_text)
        self.save_worker = SaveWorker(self)
        self.save_worker.progress_signal.connect(self.update_progress)
        self.save_worker.progress_message.connect(self.update_progress_text)
        self.import_worker = ImportWorker(self)
        self.import_worker.finished_signal.connect(self.complete_import)
        self.import_worker.progress_signal.connect(self.update_progress)
        self.import_worker.progress_message.connect(self.update_progress_text)
        self.import_worker.progress_range.connect(self.update_progress_range)

    # slots
    # def handle_tab_change(self):
    #     self.curr_frame = 0
    #     self.ui.frame_slider.setValue(0)
    #     self.ui.coloc_frame_slider.setValue(0)

    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.Wheel):
            return True
        else:
            return False

    def handle_frame_slider(self, value):
        # print("is_movie", self.is_movie)
        # print("is_segmentation", self.is_segmentation)
        if self.is_movie or self.is_segmentation:
            # curr_f = value
            # if curr_f > self.old_f:
            #     self.get_frame(1)
            # else:
            #     self.get_frame(-1)
            # self.old_f = curr_f
            self.get_frame(int(value))

    def handle_coloc_frame_slider(self, value):
        print(value)
        print("self.old_coloc_f", self.old_coloc_f)
        if self.is_segmentation:
            # curr_f = value
            # if curr_f > self.old_coloc_f:
            #     self.get_frame(1, self.sender())
            # else:
            #     self.get_frame(-1, self.sender())
            # self.old_coloc_f = curr_f
            self.get_frame(int(value))
            data_type = self.ui.data_combobox.currentIndex()
            self.update_histogram(data_type,
                                  self.ui.channel_combobox.currentIndex())

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
        self.thresholds[self.curr_channel] = float(
            self.ui.threshold_slider.value())
        self.update_display()

    def handle_blur_slider(self, value):
        val = float(value) / 8
        self.ui.blur_spinbox.setValue(val)

    def handle_blur_slider_released(self):
        self.blur = float(
            self.ui.blur_slider.value()) / 8
        self.update_display()

    def handle_blur_spinbox(self, value):
        val = int(value * 8)
        self.ui.blur_slider.setValue(val)
        self.blur = float(
            self.ui.blur_slider.value()) / 8
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
        self.curr_seg_channel = value
        data_type = self.ui.data_combobox.currentIndex()
        self.update_histogram(data_type, value)
        self.update_display()

    # def handle_analysis_channel_combo(self, value):
    #     data_type = self.ui.analysis_data_combobox.currentIndex()
    #     self.update_analysis_plot(data_type, value)

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
        self.curr_seg_channel = self.ui.channel_combobox.currentIndex()
        self.update_histogram(value, self.curr_seg_channel)

    def handle_analysis_data_combo(self, value):
        self.update_analysis_plot(value)

    def handle_add_button(self):
        path = (
            QtGui.QFileDialog.getOpenFileName(
                self, "Load Data", self.import_directory, "*.xlsx"))
        self.import_paths.append(path)
        if path and len(self.import_paths) < 5:
            self.import_worker.start_thread([path])

    def handle_remove_button(self):
        item = int(self.ui.import_list_widget.currentRow())
        self.list_view.remove(item)
        del self.import_paths[item]
        del self.analysis_data[item]
        del self.analysis_channels[item]

        data_type = self.ui.analysis_data_combobox.currentIndex()
        self.analysis_view.draw(
            self.analysis_data, data_type, self.analysis_channels)

    def update_display(self):
        self.fmin = float(self.ui.fmin_slider.value())
        self.fmax = float(self.ui.fmax_slider.value())
        self.display_frame()

    def update_histogram(self, data_type, coloc_channel):
        channel = self.params['channels'][coloc_channel]
        data = self.segmentation_result[self.curr_frame]
        self.histogram_view.draw(data, data_type, channel)

    def update_analysis_plot(self, data_type):
        if self.analysis_data:
            self.analysis_view.draw(
                self.analysis_data, data_type, self.analysis_channels)

    def load_movie(self):
        self.movie_path = (
            str(QtGui.QFileDialog.getOpenFileName(self,
                "Load Movie", self.basepath, "*.tif")))
        if self.movie_path:
            self.basepath = os.path.dirname(self.movie_path)
            self.basename = os.path.splitext(self.movie_path)[0]
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
            self.ui.progress_bar.setMaximum(2 * self.num_frames)
            self.save_worker.finished_signal.connect(self.complete_save)
            self.save_worker.start_thread(
                self.segmentation_result,
                self.params['channels'],
                self.movie_path)

    def complete_save(self):
        self.ui.progress_bar.setMaximum(self.num_frames)
        self.update_progress(0)
        self.update_progress_text("0%")

    def import_patches(self):
        import_paths = (
            QtGui.QFileDialog.getOpenFileNames(
                self, "Load Data", self.import_directory, "*.xlsx"))
        self.import_paths.extend(import_paths)
        if import_paths and len(self.import_paths) < 5:
            self.import_worker.start_thread(self.import_paths)

    def complete_import(self, results):
        self.update_progress(0)

        if (len(self.analysis_data) > 0 and
                len(self.analysis_channels) > 0):

            self.analysis_data.extend(results[0])
            self.analysis_channels.extend(results[1])
        else:
            self.analysis_data = results[0]
            self.analysis_channels = results[1]

        self.list_view.update(self.import_paths)
        self.list_view.set_selected(0)
        data_type = self.ui.analysis_data_combobox.currentIndex()
        self.analysis_view.draw(
            self.analysis_data, data_type, self.analysis_channels)

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
        params['sigma'] = self.blur
        self.params = params
        return params

    def start_obcol_thread(self):
        self.is_filtered = False
        self.is_result = False
        self.is_movie = True
        self.segmentation_result = []
        self.colocalisation_result = []
        params = self.prepare_parameters()
        self.obcol_worker.start_thread(self.movie, params)

    def obcol_plotter(self, result):
        self.segmentation_result = result
        self.tracks = utils.track(result, self.params['channels'])
        # update GUI parameters
        self.is_segmentation = True
        self.is_movie = False
        # save the result
        self.update_progress(0)
        self.update_progress_text("0%")
        self.save()
        # update the GUI itself
        self.ui.radio_groupbox.show()
        self.ui.result_radio.setChecked(True)
        self.ui.movie_radio.setChecked(False)
        # plot the histogram for each frame
        channel = self.params['channels'][0]
        self.histogram_view.draw(result[0], 0, channel)

        # now to plot overall stats effectively do an import
        # update the paths
        channel_str = "_channels_"
        for channel in self.params['channels']:
            channel_str += str(channel)

        xlpath = os.path.join(self.basename + channel_str + '_obcol.xlsx')
        if xlpath not in self.import_paths:
            self.import_paths.append(xlpath)

        # and channels that will be analysed
        self.analysis_channels.append(self.params['channels'])
        # prepare the data for the stats plot
        self.analysis_data.append(utils.convert_frames_to_patches(
            result, self.params['channels']))

        # and update the overall stats plot
        self.analysis_view.draw(self.analysis_data, 0, self.analysis_channels)

        # finally update the list widget
        self.list_view.update(self.import_paths)

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

    def update_progress(self, value):
        self.ui.progress_bar.setValue(value)

    def update_progress_range(self, value):
        self.ui.progress_bar.setMaximum(value)

    def update_progress_text(self, message):
        # self.ui.progress_bar.setRange(0, 0)
        self.ui.progress_bar.setFormat(message)

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
                self.blur,
                self.thresholds[self.curr_channel])
        if self.is_segmentation:
            sr = self.segmentation_result[self.curr_frame]
            current_labels = sr.get_mono_labels()
            frame = np.ascontiguousarray(current_labels)

            trajectories = TrajectoryList(self.tracks[self.curr_seg_channel])
            self.movie_view.show_segmentation_frame(
                frame, trajectories[0: self.curr_frame])

            self.coloc_view.show_segmentation_frame(
                frame)

        if self.is_colocalisation:
            cr = self.colocalisation_result[self.curr_frame]
            current_labels = (
                cr.get_mono_labels(filtered=True))
            filtered = (
                np.ascontiguousarray(current_labels))
            self.coloc_view.show_segmentation_frame(filtered)

    def get_frame(self, value):
        self.curr_frame = value
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
