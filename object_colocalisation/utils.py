from __future__ import print_function

import os
import time
from scipy import ndimage as ndi
from skimage.feature import peak_local_max as plm
from skimage.morphology import watershed, label
from skimage.measure import regionprops
from skimage.filters import gaussian
import numpy as np
import pandas as pd
from xlrd import XLRDError
from tifffile import imread
import trackpy as tp
from matplotlib import pyplot as plt
from wavelet_denoise import wavelet_filter as wf

from multiprocessing import Pool, Manager, cpu_count


class Frame(object):
    """A single frame of a movie. Used as an abstraction of a numpy array
    extracted from a TIF file (specifically a timelapse microscopy data
    sequence) read using the tifffile module. The Frame object stores
    information about the size and shape of the input array and information
    relating to the segmentation of image (the labels attribute).
    """
    frame_num = 0

    def __init__(self, img):
        self.img = img
        self.frame_id = Frame.frame_num
        self.num_channels, self.height, self.width = img.shape
        self.dtype = img.dtype
        self.thresholds_for_segmentation = []
        self.labels = np.zeros(
            (self.num_channels, self.height, self.width), dtype=np.uint8)
        self.filtered_labels = np.zeros(
            (self.num_channels, self.height, self.width), dtype=np.uint8)
        self.channels_to_overlap = []
        self.red_overlaps = []
        self.green_overlaps = []
        self.patches = {}
        self.patch_ids = {}
        self.size_range = ()
        self.sigma = 1
        Frame.frame_num += 1

    def get_image(self):
        return self.img

    def get_labels(self):
        return self.labels

    def get_labels_in_channel(self, channel):
        return self.labels[channel, :, :]

    def get_mono_labels_in_channel(self, channel, filtered=False):
        """Used when display overlapping labels. Sets all labels to 255 for the
        selected channel.
        Input: channel  - the index of the channel we want to display
               filtered - a boolean indicating whether to display all the
                          segmented labels or just those with an overlap
                          greater than the mimimum value
        """
        if filtered:
            labels = self.filtered_labels[channel, :, :].copy()
        else:
            labels = self.labels[channel, :, :].copy()
        lidx = labels > 0
        labels[lidx] = 255
        return labels

    def get_mono_labels(self, filtered=False):
        """Used when display overlapping labels. Sets all labels to 255 for the
        selected channel.
        Input: channel  - the index of the channel we want to display
               filtered - a boolean indicating whether to display all the
                          segmented labels or just those with an overlap
                          greater than the mimimum value
        """
        channels = self.channels_to_overlap
        labels = np.zeros((2, self.height, self.width))
        for channel in range(2):
            labels[channel, :, :] = (
                self.get_mono_labels_in_channel(channels[channel], filtered))
        return labels

    def segment(self, channels, thresholds, size_range, sigma):
        """Segments objects in the numpy representation of the Frame object.

        Input: channels -   a list of channel indices in the numpy
                            representation)
               thresholds - a list of grey level thresholds for masking
               size_range - a tuple holding the min and max size in pixels for
                            objects retained by the segmentation
        """
        self.channels_to_overlap = channels
        self.thresholds_for_segmentation = thresholds
        self.size_range = size_range
        self.sigma = sigma
        if len(channels) == 2 and len(thresholds) == 2:

            for channel, threshold in zip(channels, thresholds):
                self.locate(channel, threshold, size_range, sigma)

            # self.calculate_overlap(channels, overlap)
        else:
            return None

    def locate(self, channel, threshold, size_range, sigma):
        """Wavelet noise filtering followed by watershedding.
        Input: channel    - a index in the numpy array representation
               threshold  - the grey level threshold for that channel
               size_range - a tuple of min and max size in pixels for objects
                            to be retained by the segmentation
        """
        gauss_filtered = self._gaussian_filter(
            self.img[channel, :, :], sigma=sigma)

        (wave_filtered, thresh) = self._wavelet_filter(
            gauss_filtered, threshold=threshold)

        self.labels[channel, :, :] = self._sk_watershed(wave_filtered)

        self._label_properties(self.labels, self.img, channel, size_range)

    def object_colocalisation(self, channels, overlap):
        """Determines how much overlap there is in segmented objects represented
        by the labels attribute.
        Input: channels - a list of indices in the numpy representation of the
                          of the channels where we want to determine the
                          overlap
               overlap  - the minimum amount of overlap to look for
        """
        rlabels = self.labels[channels[0], :, :]
        glabels = self.labels[channels[1], :, :]
        self.red_overlaps = self._find_indices_of_overlaps(
            rlabels, glabels, channels[0], overlap)
        self.green_overlaps = self._find_indices_of_overlaps(
            glabels, rlabels, channels[1], overlap)

        self.filtered_labels = self.labels.copy()
        self._keep_overlapped_labels(
            self.filtered_labels[channels[0], :, :], self.red_overlaps)
        self._keep_overlapped_labels(
            self.filtered_labels[channels[1], :, :], self.green_overlaps)

    def _sk_watershed(self, image, radius=1.0):
        """Does the watershed segmentation of an image using scikit-image.
        Input: image  - a numpy representation of the data being segmented
               radius - the radius in pixels to use when creating a binary
                        mask
        """
        footprint = self._binary_mask(radius)
        distance = ndi.distance_transform_edt(image)
        coords = plm(
            distance,
            threshold_abs=0.0,
            min_distance=1,
            footprint=footprint,
            indices=False)
        markers = label(coords)
        return label(watershed(-distance, markers, mask=image), connectivity=2)

    def _label_properties(self, labels, image, channel, size_range):
        """Uses scikit-image regionprops to make measurements of segmented objects
        and creates a collection of Patch objects.
        Input: labels     - the numpy labelled array returned by watershedding
               image      - the numpy representation of the data
                            being segmented
               channel    - index in input image that has been segmented
               size_range - the min and max pixel size of objects to retain
        """
        props = regionprops(labels[channel, :, :], image[channel, :, :])
        patches = Patches()
        self.patch_ids[channel] = []
        for p in props:
            if p.area > size_range[0] and p.area < size_range[1]:
                patches.add(Patch(
                    [p.label,
                     p.coords,
                     p.area,
                     p.centroid,
                     p.intensity_image,
                     channel]))
                self.patch_ids[channel].append(p.label)
            else:
                idx = labels[channel, :, :] == p.label
                labels[channel, idx] = 0

        self.patches[channel] = patches

    def _threshold_image(self, image, thresh):
        """Returns a boolean ndarray with True where the pixel
        grey level in the input image is above the threshold value
        """
        return image > thresh

    def _binary_mask(self, radius, ndim=2, separation=None):
        """circular mask in a square array for use in scipy peak_local_max
        """
        points = np.arange(-radius, radius + 1)
        if ndim > 1:
            coords = np.array(np.meshgrid(*([points] * ndim)))
        else:
            coords = points.reshape(1, -1)
        r = np.sqrt(np.sum(coords ** 2, 0))
        return r <= radius

    def _gaussian_filter(self, image, sigma=1):
        return gaussian(image.astype(float), sigma=sigma)

    def _wavelet_filter(self, image, threshold=None):
        """Wavelet denoising of the input image
        """
        cwf = wf.CompoundWaveletFilter(3, 2.0)
        filtered = cwf.filter_image(image.astype(float))

        if threshold is None:
            threshold = np.std(cwf.result_f1)

        mask = self._threshold_image(image, threshold)
        masked = filtered * mask

        return (masked, threshold)

    def _find_indices_of_overlaps(
            self, first_labels, second_labels, channel, overlap):
        """Does the actual work of finding the overlaps between
        segmented objects in the two chosen channels.
        """
        overlapping = []
        for flabel in np.unique(first_labels):
            if flabel > 0:
                # find indices of pixels with label == flabel
                idx = np.where(first_labels == flabel)

                # get the pixels in the other channel at these indices
                pix = second_labels[idx]

                # determine the size of that region
                object_size = float(pix.shape[0])

                # determine the size of the region where the pixels in
                # the other channel have a non-zero label
                overlap_size = float(len(np.where(pix > 0)[0]))

                # identify the Patch object that belongs to this region
                patch = self.identify_patch(flabel, channel)

                # calculate the colocalisation parameters for that patch
                patch.fraction_overlapped = overlap_size / object_size
                patch.size_overlapped = object_size * patch.fraction_overlapped
                patch.signal = patch.intensity * patch.fraction_overlapped
                # keep the indices of the overlapped region for filtering
                # objects in the GUI
                if (np.any(pix) > 0 and overlap > 0.0):
                    if (overlap_size > object_size * overlap):
                        overlapping.append(flabel)

        self.patches[channel].average_overlap_fraction()
        self.patches[channel].average_overlap_size()
        self.patches[channel].average_overlap_signal()
        return overlapping

    def _keep_overlapped_labels(self, channel_labels, label_ids):
        """Determine which labels to keep for display
        """
        uids = list(np.unique(channel_labels))
        to_remove = [uid for uid in uids if uid not in label_ids]
        for label_id in to_remove:
            lidx = np.where(channel_labels == label_id)
            channel_labels[lidx] = 0

    def identify_patch(self, label_id, channel):
        """Return a Patch object for the specified label_id
        in the specified channel"""
        pout = [patch for patch in self.patches[channel]
                if patch.id == label_id]
        return pout[0]

    def sizes_of_patches(self, channel):
        """For a given channel return the sizes of all the Patches"""
        return self.patches[channel].get_sizes()

    def signals_of_patches(self, channel):
        """For a given channel return the intensity of the Patches"""
        return self.patches[channel].get_signals()

    def fraction_of_patch_overlapped(self, channel):
        """For a given channel return the amount of overlap for each
        Patch"""
        return self.patches[channel].get_overlap_fraction()

    def get_fraction_overlapped(self, channel):
        """For a given channel return the fraction of objects in the Frame
        that have an overlap above the mimimum specified"""
        return self.patches[channel].overlap_fraction


class Patch(object):
    """A patch is an abstraction of a label generate by watershed segmentation
    of a wavelet denoised image. It is used as a way of collecting measurements
    of the segmented objects.
    """
    def __init__(self, parameters):
        self.dimensions = [0, 1]  # representing x,y
        self.id = parameters[0]
        self.pixels = parameters[1]  # x,y coords of pixels in image
        self.size = parameters[2]
        self.centroid = parameters[3]  # (row, col)
        intensity_image = parameters[4]
        self.channel = parameters[5]
        if len(parameters) > 6:
            self.size_overlapped = parameters[6]
            self.fraction_overlapped = parameters[7]
        self.intensity = np.sum(intensity_image)
        self.size_overlapped = 0.0
        self.fraction_overlapped = 0.0
        self.signal = 0.0

    def __str__(self):
        return "Patch {}\nchannel: {}\nsize:{}\n"\
               "intensity: {}\nfraction overlapped: {}"\
               .format(self.id,
                       self.channel,
                       self.size,
                       self.intensity,
                       self.fraction_overlapped)

    def squared_distance_to(self, other):
        sum_squared = 0
        for c in self.dimensions:
            this_val = self.centroid[c]
            other_val = other.centroid[c]
            sum_squared += (other_val - this_val) * (other_val - this_val)

        return sum_squared


class Patches(object):
    """
    A collection of Patch objects
    """
    def __init__(self):
        self.patches = []
        self.num_patches = 0
        self.total_signal = 0.0
        self.total_size = 0.0
        self.overlap_fraction = None

    def __getitem__(self, key):
        return self.patches[key]

    def __len__(self):
        return len(self.patches)

    def __str__(self):
        return "collection of Patch objects of length {}"\
            .format(self.num_patches)

    def __repr__(self):
        return "collection of Patch objects of length {}"\
            .format(self.num_patches)

    def add(self, patch):
        self.patches.append(patch)
        self.num_patches = len(self.patches)
        self.total_size += float(patch.size)
        self.total_signal += float(patch.intensity * patch.size)

    def average_overlap_fraction(self):
        has_overlap = []
        for patch in self.patches:
            if patch.fraction_overlapped > 0.0:
                has_overlap.append(patch)

        try:
            self.overlap_fraction = (
                float(len(has_overlap)) / float(len(self.patches)))
        except ZeroDivisionError:
            self.overlap_fraction = 0.0

    def average_overlap_size(self):
        sizes = 0.0
        for patch in self.patches:
            if patch.fraction_overlapped > 0.0:
                sizes += float(patch.size) * patch.fraction_overlapped
        try:
            self.overlap_size = sizes / self.total_size
        except ZeroDivisionError:
            self.overlap_size = 0.0

    def average_overlap_signal(self):
        signal = 0.0
        for patch in self.patches:
            if patch.fraction_overlapped > 0.0:
                signal += float(patch.intensity) * patch.fraction_overlapped
        try:
            self.overlap_signal = signal / self.total_signal
        except ZeroDivisionError:
            self.overlap_signal = 0.0

    def labelled_image(self, shape, dtype):
        img = np.zeros(shape, dtype=dtype)
        for patch in self.patches:
            img[patch.pixels[:, 0], patch.pixels[:, 1]] = patch.id
        return img

    def mono_labels(self, shape, dtype):
        labels = self.labelled_image(shape, dtype)
        lidx = np.where(labels > 0)
        labels[lidx] = 255
        return labels

    def get_sizes(self):
        sizes = []
        for patch in self.patches:
            sizes.append(patch.size_overlapped)
        return sizes

    def get_signals(self):
        signals = []
        for patch in self.patches:
            signals.append(patch.signal)
        return signals

    def get_overlap_fraction(self):
        fraction = []
        for patch in self.patches:
            fraction.append(patch.fraction_overlapped)
        return fraction


class Trajectory(object):
    def __init__(self, particle):
        self.particle = particle
        self.patches = self._patches_from_particle(particle)
        self.trajectory = self._create_trajectory(particle)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.trajectory[key]
        elif isinstance(key, slice):
            return self.trajectory[key.start:key.stop:key.step]

    def _patches_from_particle(self, particle):
        patches = {}
        frame_ids = particle['frame'].values
        patch_ids = particle['patch id'].values
        for fid, pid in zip(frame_ids, patch_ids):
            patches[str(fid)] = pid
        return patches

    def _create_trajectory(self, particle):
        lines = []
        first = particle.iloc[0]
        x1 = first['x'].item()
        y1 = first['y'].item()
        for pid, p in particle.iterrows():
            x2 = p['x'].item()
            y2 = p['y'].item()
            lines.append((x1, y1, x2, y2))
            x1 = p['x'].item()
            y1 = p['y'].item()
        return lines


class TrajectoryList(object):
    def __init__(self, tracks):
        self.trajectories = self._create_trajectories(tracks)
        self.max_particle_id = tracks['particle'].max()
        print("max_particle_id", self.max_particle_id)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.trajectories[key]
        elif isinstance(key, slice):
            return self.trajectories[key.start:key.stop:key.step]

    def _create_trajectories(self, tracks):
        trajectories = []
        for tid, track in tracks.groupby('particle'):
            trajectories.append(Trajectory(track))
        return trajectories

#
# helpers for GUI
#


def generate_frames(movie_array):
    """Creates a list of Frame objects from the input numpy
    array created by tifffile's imread function
    """
    ndim = movie_array.ndim
    if ndim == 4:
        num_frames = movie_array.shape[0]
        movie_list = []
        for frame in range(num_frames):
            movie_list.append(Frame(movie_array[frame, :, :, :]))
        return movie_list

#
# for running colocalisation sequentially
#


def object_colocalisation(movie, params):
    """Colocalise frames in a movie using parameters
    defined in the parameter list
    """
    channels, thresholds, overlap, size_range, sigma = params
    results = []
    for i, frame in enumerate(movie):
        frame.segment(channels, thresholds, size_range, sigma)
        frame.object_colocalisation(channels, overlap)
        results.append(frame)
    return results
#
# helpers for running on a pool of processes
#


class Parameters(object):
    def __init__(self,
                 frame,
                 channels,
                 thresholds,
                 overlap,
                 size_range,
                 segment,
                 sigma):

        self.frame = frame
        self.channels = channels
        self.thresholds = thresholds
        self.overlap = overlap
        self.size_range = size_range
        self.segment = segment
        self.sigma = sigma


def parallel_process(parameters):
    """Colocalise objects (patches) after watershed segmentation
    in a single frame of a movie instance. Parameters is a tuple
    of (Queue, Parameters). Note that Frame objects returned
    here are copies of those created initially and are referred
    to in the GUI as results
    """
    queue = parameters[0]
    params = parameters[1]

    channels = params.channels
    thresholds = params.thresholds
    overlap = params.overlap
    size_range = params.size_range
    sigma = params.sigma
    frame = params.frame
    queue.put(frame.frame_id)
    if len(channels) == 2 and len(thresholds) == 2:
        if params.segment:
            frame.segment(channels, thresholds, size_range, sigma)
        frame.object_colocalisation(channels, overlap)
        return frame
    else:
        return None


def convert_frames_to_patches(frames, coloc_channels):
    patch_dict = {}
    for chan in coloc_channels:
        patch_list = []
        for frame in frames:
            patch_list.append(frame.patches[chan])
        patch_dict[chan] = patch_list
    return patch_dict


#
# helpers for tracking
#
def convert_to_features(frames, channel):
    output = []
    for frame_id, frame in enumerate(frames):
        for patch in frame.patches[channel]:
            output.append(
                [patch.centroid[1],
                 patch.centroid[0],
                 frame_id,
                 patch.id])

    df = pd.DataFrame(output)
    df.columns = [
        "x",
        "y",
        "frame",
        "patch id"]
    return df


def run_tracking(features):
    t = tp.link_df(
        features, 5, adaptive_stop=1, adaptive_step=0.99, memory=1)
    t1 = tp.filter_stubs(t, 10)
    particles = t1.particle.unique()
    particle_reset = 0
    for p in particles:
        t1.loc[t1['particle'] == p, 'particle'] = particle_reset
        particle_reset += 1
    return t1


def track(frames, channels):
    tracks = []
    for channel in channels:
        features = convert_to_features(frames, channel)
        t = run_tracking(features)
        tracks.append(TrajectoryList(t))
    return tracks


def redo_tracking(data):
    try:
        features = data.filter(['frame id',
                                'patch id',
                                'centroid x',
                                'centroid y'], axis=1)
        features.columns = ['frame', 'patch id', 'x', 'y']
        t = run_tracking(features)
        return t
    except ValueError:
        print('no tracks')
        return None


#
# for saving to excel using pandas
#
def save_patches(movie_path, frames, channels):
    basepath = os.path.dirname(movie_path)
    basename = os.path.splitext(movie_path)[0]
    path = os.path.join(basepath, basename + '_obcol.xlsx')
    writer = pd.ExcelWriter(path)
    channel_names = ["red", "green"]
    for c, channel in enumerate(channels):
        output = []
        for frame_id, frame in enumerate(frames):
            for patch in frame.patches[channel]:
                output.append(
                    [frame_id,
                     patch.id,
                     patch.centroid[0],
                     patch.centroid[1],
                     patch.intensity,
                     patch.channel,
                     patch.size,
                     patch.size_overlapped])

            df = pd.DataFrame(output)
            df.columns = [
                "frame id",
                "patch id",
                "centroid x",
                "centroid y",
                "intensity",
                "channel",
                "size",
                "size overlapped"]
            df.to_excel(
                writer, sheet_name=channel_names[c], index=False)
    writer.save()


#
# helper function for testing purposes only
#
def run(movie, parameters, segment=True):
    num_cpus = cpu_count()
    pool = Pool(processes=num_cpus - 1)
    m = Manager()
    q = m.Queue()
    channels, thresholds, overlap, size_range, sigma = parameters
    parameter_list = [(q, Parameters(
        frame, channels, thresholds, overlap, size_range, segment, sigma))
        for frame in movie]

    tic3 = time.time()
    results = []
    rs = pool.map_async(
        parallel_process,
        parameter_list,
        callback=results.append)
    pool.close()  # No more work
    while (True):
        if (rs.ready()):
            break
        print("Object colocalisation: {}%".format(
            int((float(q.qsize()) / float(len(movie))) * 100)))
    toc3 = time.time()
    print(toc3 - tic3)
    print(len(movie))
    return results[0]

#
# helpers for importing data from xls
#


def generate_patches(df):
    frames = []
    for gid, group in df.groupby('frame id'):
        patches = Patches()
        for rid, row in group.iterrows():
            patches.add(Patch(
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
    return frames


def get_channels_from_path(path):
    cidx = path.find('channels') + 9
    return [int(i) for i in list(path[cidx: cidx + 2])]


def read_sheet(path, sheetname):
    return pd.read_excel(path, sheetname=sheetname)


def read_patches_from_file(path):
    if path.endswith('xlsx'):
        sheets = ['red', 'green']
        channels = get_channels_from_path(path)
        patches = {}
        for c, channel in enumerate(channels):
            patches[channel] = generate_patches(read_sheet(path, sheets[c]))
        return patches
    else:
        raise ValueError("Input data must be in Excel format")


def import_tracks(path, sheetname=None):
    if path.endswith('xlsx'):
        sheets = ['red', 'green']
        tracks = {}
        for sheet in sheets:
            sn = ''
            if sheetname:
                sn = '{} {}'.format(sheet, sheetname)

            try:
                df = pd.read_excel(path, sheet_name=sn)
                tracks[sheet] = df[['x', 'y', 'frame', 'particle']].copy()
                print('read sheet {}'.format(sn))
            except ValueError:
                sn = '{} tracks'.format(sheet)
                df = pd.read_excel(path, sheet_name=sn)
                tracks[sheet] = df[['x', 'y', 'frame', 'particle']].copy()
                print('read sheet {}'.format(sn))
            except XLRDError:
                sn = '{} motion'.format(sheet)
                df = pd.read_excel(path, sheet_name=sn)
                tracks[sheet] = df[['x', 'y', 'frame', 'particle']].copy()
                print('read sheet {}'.format(sn))

        return tracks
    else:
        raise ValueError("Input data must be in Excel format")


def batch_import_tracks(input_dir):
    data = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".xlsx"):
            print(filename)
            tracks_path = os.path.join(input_dir, filename)
            data[filename] = import_tracks(tracks_path)

    return data


if __name__ == '__main__':

    path = "/home/daniel/Documents/Image Processing/Mag/data/WT 0-30min 2.tif"
    movie_array = imread(path)
    movie = generate_frames(movie_array)
    channels = [2, 0]
    thresholds = [2061, 2735]
    overlap = 0.0
    size_range = (1, 100000)
    params = [channels, thresholds, overlap, size_range, 1]

    results = run(movie, params)

    features = convert_to_features(results, 2)
    t = tp.link_df(features, 5, adaptive_stop=1, adaptive_step=0.99, memory=1)
    t1 = tp.filter_stubs(t, 10)
    plt.figure()
    tp.plot_traj(t1)
    for pid, particle in t1.groupby('particle'):
        if pid == 0.0:
            print(particle)

    trajs = TrajectoryList(t1)
    for trajid, traj in enumerate(trajs):
        if trajid == 0:
            print(traj.trajectory)

    unstacked = t1.set_index(['particle', 'frame'])[['x', 'y']].unstack()
    print(unstacked.head())
    # results = object_colocalisation(movie, params)
    # overlap = 0.2
    # params = [channels, thresholds, overlap, size_range]
    # filtered = run(results, params, segment=False)

    # plt.figure()
    # plt.imshow(results[0].get_mono_labels_in_channel(0), interpolation='nearest')
    # plt.show()

    # plt.figure()
    # plt.imshow(filtered[0].get_mono_labels_in_channel(0, filtered=True), interpolation='nearest')
    # plt.show()

    # results = object_colocalisation(movie, params)
    # patches = read_patches_from_file("WT_channels_20_obcol.xlsx")
    # red = patches[0]
    # print(red[0][0])
    # img = imread(path)[20, :, :, :]
    # plt.figure()
    # plt.imshow(img[2, :, :], interpolation='nearest')
    # plt.show()

    # # gimg = gaussian(img[2, :, :].astype(float), 1)
    # # plt.figure()
    # # plt.imshow(gimg, interpolation='nearest')
    # # plt.show()

    # frame = Frame(img)
    # frame.segment([2, 0], [2390, 2140], [1, 1000000])

    # plt.figure()
    # plt.imshow(frame.labels[2, :, :], interpolation='nearest')
    # plt.show()

    # plt.figure()
    # plt.imshow(frame.labels[0, :, :], interpolation='nearest')
    # plt.show()