#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
from PyQt4.QtCore import QThread
from PyQt4.QtCore import Signal
import pandas as pd
import numpy as np
from collections import OrderedDict
from tifffile import imsave
from multiprocessing import Pool, Manager, cpu_count

import utils

# this is pulled from SMLM as example - modify for colocalisation worker thread
# note that you should also look at deploying this thread across multiple cores


class ObcolWorker(QThread):
    progress_signal = Signal(int)
    progress_message = Signal(str)
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
        self.num_frames = len(movie)
        self.movie_path = parameters['movie_path']
        self.channels = parameters['channels']
        self.thresholds = parameters['thresholds']
        self.overlap = parameters['overlap']
        self.size_range = parameters['size_range']
        self.sigma = parameters['sigma']
        self.segment = parameters['segment']
        self.save_parameters()
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
        q = self.queue
        parameter_list = [(q, utils.Parameters(
            frame,
            self.channels,
            self.thresholds,
            self.overlap,
            self.size_range,
            self.segment,
            self.sigma))
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
            self.progress_message.emit(
                "Object colocalisation: {}%".format(
                    int((float(q.qsize()) /
                        float(self.num_frames)) * 100)))
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
        self.sigma = self.parameters['sigma']
        results = []
        for i, frame in enumerate(self.movie):
            self.processing_frame.emit(i)
            frame.segment(self.channels,
                          self.thresholds,
                          self.overlap,
                          self.size_range,
                          self.sigma)
            results.append(frame)

        return results

    def save_parameters(self):
        basepath = os.path.dirname(self.movie_path)
        basename = os.path.splitext(self.movie_path)[0]

        channel_str = "_channels_"
        for channel in self.channels:
            channel_str += str(channel)

        path = os.path.join(
            basepath, basename + channel_str + '_obcol_parameters.txt')
        with open(path, "w") as file:
            file.write("channels:" +
                       ",".join([str(c) for c in self.channels]) + "\n")
            file.write("thresholds:" +
                       ",".join([str(c) for c in self.thresholds]) + "\n")
            file.write("size range:" +
                       ",".join([str(c) for c in self.size_range]) + "\n")
            file.write("gaussian filter sigma:" + str(self.sigma) + "\n")

    def clear_queue(self):
        while not self.queue.empty():
            self.queue.get_nowait()


class SaveWorker(QThread):
    progress_signal = Signal(int)
    progress_message = Signal(str)
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

        labels = []

        movie_path = self.path
        channels = self.channels
        basepath = os.path.dirname(movie_path)
        basename = os.path.splitext(movie_path)[0]

        channel_str = "_channels_"
        for channel in channels:
            channel_str += str(channel)

        path = os.path.join(basepath, basename + channel_str + '_obcol.xlsx')
        self.writer = pd.ExcelWriter(path)
        channel_names = ["red", "green"]
        counter = 0
        for c, channel in enumerate(channels):
            labels = []
            obcol = []

            vesicle_number = OrderedDict()
            vesicle_sizes = OrderedDict()
            vesicle_signals = OrderedDict()
            vesicle_overlaps = OrderedDict()

            for frame_id, frame in enumerate(frames):

                movie_array[frame_id, c, :, :] = (
                    frame.get_mono_labels_in_channel(channel))

                labels.append(frame.labels[channel, :, :])

                fkey = "frame_{}".format(str(frame_id).zfill(3))
                vesicle_number[fkey] = len(frame.patches[channel])
                vesicle_sizes[fkey] = []
                vesicle_signals[fkey] = []
                vesicle_overlaps[fkey] = []

                self.progress_signal.emit(counter)
                self.progress_message.emit(
                    "Saving object colocalisation results: {}%".format(
                        int((float(counter) / float(2 * len(frames))) * 100)))

                if len(frame.patches[channel]) > 0:
                    for patch in frame.patches[channel]:
                        obcol.append(
                            [frame_id,
                             patch.id,
                             patch.centroid[0],
                             patch.centroid[1],
                             patch.intensity,
                             patch.channel,
                             patch.size,
                             patch.size_overlapped,
                             patch.signal,
                             float(patch.size_overlapped) / float(patch.size)])

                        vesicle_sizes[fkey].append(patch.size)
                        vesicle_signals[fkey].append(patch.signal)
                        vesicle_overlaps[fkey].append(
                            float(patch.size_overlapped) / float(patch.size))
                else:
                    obcol.append([frame_id] + [0.0 for i in range(9)])

                counter += 1

            obcol_df = pd.DataFrame(obcol)
            obcol_df.columns = [
                "frame id",
                "patch id",
                "centroid x",
                "centroid y",
                "intensity",
                "channel",
                "size",
                "size overlapped",
                "signal",
                "fraction overlapped"]

            obcol_df.to_excel(
                self.writer, sheet_name=channel_names[c], index=False)

            vnumber_df = self._df_from_dict(vesicle_number,
                                            columns=['num vesicles per frame'])
            vnumber_df.to_excel(
                self.writer,
                sheet_name="vesicle number {}".format(channel_names[c]),
                index=True)

            params = dict([("vesicle size", vesicle_sizes),
                           ("vesicle signal", vesicle_signals),
                           ("vesicle overlap", vesicle_overlaps)])

            for k, v in params.iteritems():
                df = self._df_from_dict(v)
                df.to_excel(self.writer,
                            sheet_name="{} {}".format(k, channel_names[c]),
                            index=True)

            label_array = np.rollaxis(np.dstack(labels), -1)
            label_filename = (
                basename + channel_str +
                '_{}labels.tif'.format(channel_names[c]))
            imsave(
                os.path.join(basepath, label_filename),
                label_array.astype(np.uint8),
                compress=6,
                metadata={'axes': 'TYX'})

        imsave(
            os.path.join(basepath, basename + channel_str + '_segmented.tif'),
            movie_array.astype(np.uint8),
            compress=6,
            metadata={'axes': 'TCYX'})

        self.writer.save()

    def _df_from_dict(self, data_dict, columns=None):
        df = pd.DataFrame().from_dict(
            data_dict, orient='index')
        if columns:
            df.columns = columns
        df['mean'] = df.mean(axis=1)
        df['std'] = df.std(axis=1)
        return df


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
                     int(row['signal'].item()),
                     row['fraction overlapped'].item()]))
            patches.average_overlap_fraction()
            patches.average_overlap_size()
            patches.average_overlap_signal()
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
