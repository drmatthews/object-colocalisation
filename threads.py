#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
from PyQt4.QtCore import QThread
from PyQt4.QtCore import Signal
import pandas as pd
import numpy as np
from tifffile import imsave
from multiprocessing import Pool, Manager, cpu_count

import utils

# this is pulled from SMLM as example - modify for colocalisation worker thread
# note that you should also look at deploying this thread across multiple cores


class ObcolWorker(QThread):
    progress_signal = Signal(int)
    results_signal = Signal(list)

    def __init__(self, parent=None):
        super(ObcolWorker, self).__init__(parent)
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
        results = self.object_colocalisation()
        self.results_signal.emit(results)
        self.clear_queue()
        # self.save(results)
        # results = self.colocalise_movie_sequential()
        # self.post_process(results)

    def object_colocalisation(self):
        """
        Distribute the work of colocalisation across multiple CPUs.
        Note this uses the utils.parallel_obcol helper function
        """
        self.channels = self.parameters['channels']
        self.thresholds = self.parameters['thresholds']
        self.overlap = self.parameters['overlap']
        print("overlap", self.overlap)
        self.size_range = self.parameters['size_range']
        self.segment = self.parameters['segment']
        print("segment", self.segment)
        q = self.queue
        parameter_list = [(q, utils.Parameters(
            frame,
            self.channels,
            self.thresholds,
            self.overlap,
            self.size_range,
            self.segment))
            for frame in self.movie]
        results = []
        rs = self.pool.map_async(
            utils.parallel_process,
            parameter_list,
            callback=results.append)
        while (True):
            if (rs.ready()):
                break
            self.progress_signal.emit(q.qsize())
        rs.wait()
        return results[0]

    def object_colocalisation_sequential(self):
        """
        Colocalise frames in a movie instance using parameters
        defined in the parameter list - run on a single CPU
        and uses the obcol method on the frame object
        """
        self.channels = self.parameters['channels']
        self.thresholds = self.parameters['thresholds']
        self.overlap = self.parameters['overlap']
        self.size_range = self.parameters['size_range']
        results = []
        for i, frame in enumerate(self.movie):
            self.processing_frame.emit(i)
            frame.segment(
                self.channels, self.thresholds, self.overlap, self.size_range)
            results.append(frame)

        return results

    def save(self, results):
        movie_path = self.parameters['movie_path']
        channels = self.channels
        utils.save_patches(
            os.path.basename(movie_path), results, channels)

    def clear_queue(self):
        while not self.queue.empty():
            self.queue.get_nowait()


class SaveWorker(QThread):
    progress_signal = Signal(int)
    finished_signal = Signal(int)

    def __init__(self, parent=None):
        super(SaveWorker, self).__init__(parent)
        self.stopped = False

    def stop(self):
        self.stopped = True

    def is_stopped(self):
        return self.stopped

    def start_thread(self, data, channels, path):
        """
        Method called from the GUI
        """
        self.data = data
        self.channels = channels
        self.path = path
        self.start()

    def run(self):
        """
        This gets called when self.start() is called
        """
        if self.is_stopped():
            return
        self.save()
        self.finished_signal.emit(0)

    def save(self):
        frames = self.data
        movie_array = np.zeros(
            (len(frames), 2, frames[0].height, frames[0].width))
        movie_path = self.path
        channels = self.channels
        basepath = os.path.dirname(movie_path)
        basename = os.path.splitext(movie_path)[0]

        channel_str = "_channels_"
        for channel in channels:
            channel_str += str(channel)

        path = os.path.join(basepath, basename + channel_str + '_obcol.xlsx')
        writer = pd.ExcelWriter(path)
        channel_names = ["red", "green"]
        counter = 0
        for c, channel in enumerate(channels):
            output = []
            for frame_id, frame in enumerate(frames):
                movie_array[frame_id, c, :, :] = (
                    frame.get_mono_labels_in_channel(channel))

                self.progress_signal.emit(counter)
                for patch in frame.patches[channel]:
                    output.append(
                        [frame_id,
                         patch.id,
                         patch.centroid[0],
                         patch.centroid[1],
                         patch.intensity,
                         patch.channel,
                         patch.size,
                         patch.size_overlapped,
                         float(patch.size_overlapped) / float(patch.size)])

                df = pd.DataFrame(output)
                df.columns = [
                    "frame id",
                    "patch id",
                    "centroid x",
                    "centroid y",
                    "intensity",
                    "channel",
                    "size",
                    "size overlapped",
                    "fraction overlapped"]
                df.to_excel(
                    writer, sheet_name=channel_names[c], index=False)
                counter += 1
        imsave(
            os.path.join(basepath, basename + channel_str + '_segmented.tif'),
            movie_array.astype(np.uint8),
            compress=6,
            metadata={'axes': 'TCYX'})
        writer.save()


class ImportWorker(QThread):
    finished_signal = Signal(tuple)
    progress_signal = Signal(int)
    progress_message = Signal(str)
    progress_range = Signal(int)

    def __init__(self, parent=None):
        super(ImportWorker, self).__init__(parent)
        self.stopped = False

    def stop(self):
        self.stopped = True

    def is_stopped(self):
        return self.stopped

    def start_thread(self, path_list):
        """
        Method called from the GUI
        """
        self.path_list = path_list
        self.start()

    def run(self):
        if self.is_stopped():
            return
        imported_data = []
        analysis_channels = []
        for p, path in enumerate(self.path_list):
            imported_data.append(
                self.read_patches_from_file(str(path)))
            analysis_channels.append(
                self.get_channels_from_path(str(path)))
        self.progress_message.emit("0%")
        self.finished_signal.emit((imported_data, analysis_channels))

    def generate_patches(self, df):
        frames = []
        self.progress_range.emit(df['frame id'].max())
        for gid, group in df.groupby('frame id'):
            patches = utils.Patches()
            for rid, row in group.iterrows():
                patches.add(utils.Patch(
                    [int(row['patch id'].item()),
                     None,
                     int(row['size'].item()),
                     (row['centroid x'].item(), row['centroid y'].item()),
                     int(row['intensity']),
                     int(row['channel'].item()),
                     int(row['size overlapped'].item()),
                     row['fraction overlapped'].item()]))
            patches.calculate_overlap_fraction()
            patches.calculate_overlap_size()
            patches.calculate_overlap_signal()
            frames.append(patches)
            self.progress_signal.emit(gid)
        return frames

    def get_channels_from_path(self, path):
        cidx = path.find('channels') + 9
        return [int(i) for i in list(path[cidx: cidx + 2])]

    def read_sheet(self, path, sheetname):
        return pd.read_excel(path, sheetname=sheetname)

    def read_patches_from_file(self, path):
        if path.endswith('xlsx'):
            sheets = ['red', 'green']
            channels = self.get_channels_from_path(path)
            patches = {}
            for c, channel in enumerate(channels):
                self.progress_message.emit(
                    "{}: Importing {}".format(
                        os.path.basename(path), sheets[c]))

                patches[channel] = self.generate_patches(
                    self.read_sheet(path, sheets[c]))
            return patches
        else:
            raise ValueError("Input data must be in Excel format")
