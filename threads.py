#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from PyQt4.QtCore import QThread
from PyQt4.QtCore import Signal
import numpy as np
from multiprocessing import Pool, Manager, cpu_count

import utils

# this is pulled from SMLM as example - modify for colocalisation worker thread
# note that you should also look at deploying this thread across multiple cores


class SegmentationWorker(QThread):
    progress_signal = Signal(int)
    results_signal = Signal(list)

    def __init__(self, parent=None):
        super(SegmentationWorker, self).__init__(parent)
        self.stopped = False
        num_cpus = cpu_count() - 1  # don't kill the computer
        self.pool = Pool(num_cpus)
        manager = Manager()
        self.queue = manager.Queue()

    def stop(self):
        self.pool.terminate()
        self.stopped = True

    def is_stopped(self):
        return self.stopped

    def start_thread(self, movie, parameters):
        """
        Method called from the GUI
        """
        self.movie = movie
        self.parameters = parameters
        self.start()

    def run(self):
        """
        This gets called when self.start() is called
        """
        if self.is_stopped():
            return
        results = self.segment_movie()
        self.results_signal.emit(results)
        # results = self.colocalise_movie_sequential()
        # self.post_process(results)

    def segment_movie(self):
        """
        Distribute the work of colocalisation across multiple CPUs.
        Note this uses the utils.parallel_obcol helper function
        """
        self.channels = self.parameters['channels']
        self.thresholds = self.parameters['thresholds']
        self.overlap = 0.0
        q = self.queue
        parameter_list = [(q, utils.Parameters(
            frame, self.channels, self.thresholds, self.overlap))
            for frame in self.movie]
        results = []
        rs = self.pool.map_async(
            utils.parallel_segment, parameter_list, callback=results.append)
        while (True):
            if (rs.ready()):
                break
            self.progress_signal.emit(q.qsize())
        rs.wait()
        return results[0]

    def segment_movie_sequential(self):
        """
        Colocalise frames in a movie instance using parameters
        defined in the parameter list - run on a single CPU
        and uses the obcol method on the frame object
        """
        self.channels = self.parameters['channels']
        self.thresholds = self.parameters['thresholds']
        self.overlap = self.parameters['overlap']
        results = []
        for i, frame in enumerate(self.movie):
            self.processing_frame.emit(i)
            frame.segment(self.channels, self.thresholds, self.overlap)
            results.append(frame)

        return results

    # def post_process(self, results):
    #     """
    #     Prepare the result of coloclisation for display in the GUI
    #     """
    #     num_frames = self.movie.num_frames
    #     all_labels = np.zeros((
    #         num_frames, 2, self.movie.height, self.movie.width))
    #     colocalised = []
    #     print(len(results))
    #     for i, frame in enumerate(results):
    #         for channel in range(2):
    #             all_labels[i, channel, :, :] = (
    #                 frame.mono_labels(self.channels[channel]))
    #         c = (frame.red_overlaps, frame.green_overlaps)
    #         colocalised.append(c)
    #     self.label_signal.emit(all_labels)
    #     self.colocalisation_signal.emit(colocalised)


class ColocWorker(QThread):
    progress_signal = Signal(int)
    results_signal = Signal(list)

    def __init__(self, parent=None):
        super(ColocWorker, self).__init__(parent)
        self.stopped = False

    def stop(self):
        self.stopped = True

    def isStopped(self):
        return self.stopped

    def start_thread(self, movie, parameters):
        """
        Method called from the GUI
        """
        self.movie = movie
        self.parameters = parameters
        self.start()

    def run(self):
        """
        This gets called when self.start() is called
        """
        results = self.coloclisation()
        self.results_signal.emit(results)

    def coloclisation(self):
        results = []
        channels = self.parameters['channels']
        overlap = self.parameters['overlap']

        for i, frame in enumerate(self.movie):
            self.progress_signal.emit(i)
            frame.object_colocalisation(channels, overlap)
            results.append(frame)

        return results
