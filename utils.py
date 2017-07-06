import os
import time
from scipy import ndimage as ndi
from skimage.feature import peak_local_max as plm
from skimage.morphology import watershed, label
from skimage.measure import regionprops
import numpy as np
import pandas as pd
from tifffile import imread
from matplotlib import pyplot as plt
from wavelet_denoise import wavelet_filter as wf

from multiprocessing import Pool, Manager, cpu_count


class Frame:
    def __init__(self, frame_id, im):
        self.img = im
        self.frame_id = frame_id
        self.num_channels, self.height, self.width = im.shape
        self.dtype = im.dtype
        self.thresholds_for_segmentation = []
        self.labels = np.zeros(
            (self.num_channels, self.height, self.width), dtype=np.uint8)
        self.filtered_labels = np.zeros(
            (self.num_channels, self.height, self.width), dtype=np.uint8)
        self.channels_to_overlap = []
        self.red_overlaps = []
        self.green_overlaps = []
        self.patches = {}

    def segment(self, channels, thresholds):
        """Colocalise objects (patches) after watershed segmentation
        """
        self.channels_to_overlap = channels
        self.thresholds_for_segmentation = thresholds
        if len(channels) == 2 and len(thresholds) == 2:

            for channel, threshold in zip(channels, thresholds):
                self.locate(channel, threshold)

            # self.calculate_overlap(channels, overlap)
        else:
            return None

    def locate(self, channel, threshold):
        """Watershed segmentation of the image
        """
        (filtered, thresh) = self._wavelet_filter(
            self.img[channel, :, :], threshold=threshold)
        self.labels[channel, :, :] = self._sk_watershed(filtered)

        self._label_properties(self.labels, self.img, channel)

    def object_colocalisation(self, channels, overlap):
        """Identifies the overlapping objects in two segmented channels.
        The amount of overlap is specified by the user.
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

    def mono_labels(self, channel, filtered):
        """Used when display overlapping labels. Sets all labels to 255 for the
        selected channel.
        """
        if filtered:
            labels = self.filtered_labels[channel, :, :].copy()
        else:
            labels = self.labels[channel, :, :].copy()
        lidx = np.where(labels > 0)
        labels[lidx] = 255
        return labels

    def _sk_watershed(self, image, radius=1.0):
        """Does the watershed segmentation of an image using scikit-image.
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

    def _label_properties(self, labels, image, channel):
        """Uses scikit-image regionprops to make measurements of segmented objects
        and creates a collection of Patch objects.
        """
        props = regionprops(labels[channel, :, :], image[channel, :, :])
        patches = Patches()
        for p in props:
            patches.add(Patch(
                [p.label,
                 p.coords,
                 p.area,
                 p.centroid,
                 p.intensity_image,
                 channel]))
        self.patches[channel] = patches

    def _threshold_image(self, image, thresh):
        """Returns a boolean ndarray
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
                idx = np.where(first_labels == flabel)
                pix = second_labels[idx]
                object_size = float(pix.shape[0])
                overlap_size = float(len(np.where(pix > 0)[0]))
                patch = self._identify_patch(flabel, channel)
                patch.fraction_overlapped = overlap_size / patch.area
                if (np.any(pix) > 0 and overlap > 0.0):
                    if (overlap_size > object_size * overlap):
                        overlapping.append(flabel)

        self.patches[channel].calculate_fraction_overlapped()
        return overlapping

    def _keep_overlapped_labels(self, channel_labels, label_ids):
        uids = list(np.unique(channel_labels))
        # print("image labels", uids)
        to_remove = [uid for uid in uids if uid not in label_ids]
        # print("to remove", to_remove)
        for label_id in to_remove:
            lidx = np.where(channel_labels == label_id)
            channel_labels[lidx] = 0

    def _identify_patch(self, label_id, channel):
        pout = [patch for patch in self.patches[channel]
                if patch.id == label_id]
        return pout[0]


class Patch:
    """A patch is an abstraction of a label generate by watershed segmentation
    of a wavelet denoised image. It is used as a way of collecting measurements
    of the segmented objects.
    """
    def __init__(self, parameters):
        self.id = parameters[0]
        self.pixels = parameters[1]  # x,y coords of pixels in image
        self.area = parameters[2]
        self.centroid = parameters[3]
        intensity_image = parameters[4]
        self.intensity = np.sum(intensity_image)
        self.channel = parameters[5]
        self.fraction_overlapped = None


class Patches:
    """This is a collection of Patch objects. This is used to recontruct
    a label image for display.
    """
    def __init__(self):
        self.patches = []
        self.num_patches = len(self.patches)
        self.total_intensity = self._calculate_total_intensity()
        self.fraction_with_overlap = None

    def __getitem__(self, key):
        return self.patches[key]

    def add(self, patch):
        self.patches.append(patch)

    def _calculate_total_intensity(self):
        total = 0
        for patch in self.patches:
            total += patch.intensity
        return total

    def calculate_fraction_overlapped(self):
        has_overlap = []
        for patch in self.patches:
            if patch.fraction_overlapped > 0.0:
                has_overlap.append(patch)

        self.fraction_with_overlap = (
            float(len(has_overlap)) / float(len(self.patches)))

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
            movie_list.append(Frame(frame, movie_array[frame, :, :, :]))
        return movie_list


def load_frame(movie, frame_id):
    """Return the numpy array representation of the Frame
    object for GUI image display
    """
    return movie[frame_id].img


def load_frame_labels(results, frame_id, filtered=False):
    frame = results[frame_id]
    channels = frame.channels_to_overlap
    labels = np.zeros((2, frame.height, frame.width))
    for channel in range(2):
        labels[channel, :, :] = (
            frame.mono_labels(channels[channel], filtered))
    return labels
#
# for running colocalisation sequentially
#


def segment_movie(movie, params):
    """Colocalise frames in a movie instance using parameters
    defined in the parameter list
    """
    channels, thresholds, overlap = params

    for i, frame in enumerate(movie):
        frame.segment(channels, thresholds)

#
# helpers for running on a pool of processes
#


class Parameters:
    def __init__(self, frame, channels, thresholds, overlap):
        self.frame = frame
        self.channels = channels
        self.thresholds = thresholds
        self.overlap = overlap


def parallel_obcol(parameters):
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
    frame = params.frame
    queue.put(frame.frame_id)
    if len(channels) == 2 and len(thresholds) == 2:
        frame.segment(channels, thresholds)
        frame.object_colocalisation(channels, overlap)
        return frame
    else:
        return None

#
# helper function for testing purposes only
#


def run(movie, parameters):
    num_cpus = cpu_count()
    pool = Pool(processes=num_cpus - 1)
    m = Manager()
    q = m.Queue()

    parameter_list = [(q, Parameters(frame, channels, thresholds, overlap))
                      for frame in movie]

    tic3 = time.time()
    results = []
    rs = pool.map_async(
        parallel_segment, parameter_list, callback=results.append)
    pool.close()  # No more work
    while (True):
        if (rs.ready()):
            break
        # print(q.qsize())
    toc3 = time.time()
    print(toc3 - tic3)
    return results[0]


if __name__ == '__main__':

    path = "WT.tif"
    movie_array = imread(path)[0:2, :, :, :]
    movie = generate_frames(movie_array)
    channels = [2, 0]
    thresholds = [250, 1600]
    overlap = 0.0
    params = [channels, thresholds, overlap]

    results = run(movie, params)
    # results = segment_movie(movie, params)

    # print(len(results))
    # result = results[0]
    # print(result == movie[0])
    # print(result)
    # print("red overlaps", result.red_overlaps)
    # print("green overlaps", result.green_overlaps)
    # plt.figure()
    # plt.imshow(result.mono_labels(2), interpolation='nearest')
    # plt.show()
    frame = results[0]
    frame.object_colocalisation(channels, 0.5)
    print(frame.patches[0].fraction_with_overlap)

    # # # plt.figure()
    # # # plt.imshow(result.mono_labels(2), interpolation='nearest')
    # # # plt.show()

    # labels = load_frame_labels(results, 0)
    # filtered = load_frame_labels(results, 0, filtered=True)
    # plt.figure()
    # plt.imshow(labels[0, :, :], interpolation='nearest')
    # plt.show()

    # plt.figure()
    # plt.imshow(filtered[0, :, :], interpolation='nearest')
    # plt.show()    