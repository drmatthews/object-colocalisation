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


class Movie:
    def __init__(self, path):
        ext = os.path.splitext(path)[1]
        if 'tif' in ext:
            self.movie = self._get_frames(imread(path))
        else:
            raise IOError

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self.movie[ii]
                    for ii in xrange(*key.indices(len(self.movie)))]
        elif isinstance(key, int):
            if key >= 0 and key <= len(self.movie):
                return self.movie[key]

    def __len__(self):
        return len(self.movie)

    def _get_frames(self, movie_array):
        self.ndim = movie_array.ndim
        if self.ndim == 4:
            self.dtype = movie_array.dtype
            self.num_frames = movie_array.shape[0]
            self.num_channels = movie_array.shape[1]
            self.height = movie_array.shape[2]
            self.width = movie_array.shape[3]
            movie_list = []
            for frame in range(self.num_frames):
                movie_list.append(Frame(frame, movie_array[frame, :, :, :]))
            return movie_list
        else:
            print("movie should have 4 dimensions")


class Frame:
    def __init__(self, frame_id, im):
        self.img = im
        self.frame_id = frame_id
        self.num_channels, self.height, self.width = im.shape
        self.dtype = im.dtype
        self.threshold = None
        self.labels = np.zeros(
            (self.num_channels, self.height, self.width), dtype=np.uint8)

    def locate(self, channel, threshold):
        """
        Watershed segmentation of the image
        """
        (filtered, thresh) = self._filter(
            self.img[channel, :, :], threshold=threshold)
        self.labels[channel, :, :] = self._sk_watershed(filtered)

        self._label_properties(
            self.labels[channel, :, :], self.img[channel, :, :])

    def obcol(self, channels, thresholds, overlap):
        """
        Colocalise objects (patches) after watershed segmentation
        """
        if len(channels) == 2 and len(thresholds) == 2:

            for channel, threshold in zip(channels, thresholds):
                self.locate(channel, threshold)

            self.overlap(channels, overlap)
        else:
            return None

    def _sk_watershed(self, image, radius=1.0):
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

    def _label_properties(self, labels, image):

        props = regionprops(labels, image)
        self.patches = Patches()
        for p in props:
            if p.area > 1:
                self.patches.add(Patch(p.label, p.coords, p.area, p.centroid))

    def mono_labels(self, channel):
        labels = self.labels[channel, :, :]
        lidx = np.where(labels > 0)
        labels[lidx] = 255
        return labels

    def _threshold_image(self, image, thresh):
        return image > thresh

    def _binary_mask(self, radius, ndim=2, separation=None):
        "circular mask in a square array"
        points = np.arange(-radius, radius + 1)
        if ndim > 1:
            coords = np.array(np.meshgrid(*([points] * ndim)))
        else:
            coords = points.reshape(1, -1)
        r = np.sqrt(np.sum(coords ** 2, 0))
        return r <= radius

    def _filter(self, image, threshold=None):

        cwf = wf.CompoundWaveletFilter(3, 2.0)
        filtered = cwf.filter_image(image.astype(float))

        if threshold is None:
            threshold = np.std(cwf.result_f1)

        mask = self._threshold_image(image, threshold)
        masked = filtered * mask

        return (masked, threshold)

    def overlap(self, channels, overlap):
        """
        Identifies the overlapping objects in two segmented channels.
        The amount of overlap is specified by the user.
        """
        rlabels = self.labels[channels[0], :, :]
        glabels = self.labels[channels[1], :, :]

        self.red_overlaps = self._find_overlaps(rlabels, glabels, overlap)
        self.green_overlaps = self._find_overlaps(glabels, rlabels, overlap)

    def _find_overlaps(self, first_labels, second_labels, overlap):
        """
        Does the actual work of finding the overlaps between
        segmented objects in the two chosen channels.
        """
        overlapping = []
        for flabel in np.unique(first_labels):
            if flabel > 0:
                idx = np.where(first_labels == flabel)
                pix = second_labels[idx]
                if (np.any(pix) > 0):
                    overlapping.append(flabel)
                    if (float(len(np.where(pix > 0)[0])) >
                            float(pix.shape[0] * overlap)):
                        overlapping.append(flabel)
        return overlapping


class Patch:
    def __init__(self, label_id, pixels, area, centroid):
        self.id = label_id
        self.pixels = pixels  # x,y coords of pixels in image
        self.area = area
        self.centroid = centroid


class Patches:
    def __init__(self):
        self.patches = []

    def add(self, patch):
        self.patches.append(patch)

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
# for running colocalisation sequentially
#


def colocalise_movie(movie, params):
    """
    Colocalise frames in a movie instance using parameters
    defined in the parameter list
    """
    channels, thresholds, overlap = params

    for i, frame in enumerate(movie):
        frame.obcol(channels, thresholds, overlap)

#
# helpers for running in parallel from gui
#


class Parameters:
    def __init__(self, frame, channels, thresholds, overlap):
        self.frame = frame
        self.channels = channels
        self.thresholds = thresholds
        self.overlap = overlap


def parallel_obcol(parameters):
    """
    Colocalise objects (patches) after watershed segmentation
    in a single frame of a movie instance. Parameters is a tuple
    of (Queue, Parameters)
    """
    queue = parameters[0]
    params = parameters[1]

    channels = params.channels
    thresholds = params.thresholds
    overlap = params.overlap
    frame = params.frame
    queue.put(frame.frame_id)
    if len(channels) == 2 and len(thresholds) == 2:

        for channel, threshold in zip(channels, thresholds):
            frame.locate(channel, threshold)

        frame.overlap(channels, overlap)
        return frame
    else:
        return None


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
        parallel_obcol, parameter_list, callback=results.append)
    pool.close()  # No more work
    while (True):
        if (rs.ready()):
            break
        print(q.qsize())
    toc3 = time.time()
    print(toc3 - tic3)
    return results[0]


if __name__ == '__main__':

    path = "WT.tif"
    movie = Movie(path)
    channels = [2, 0]
    thresholds = [250, 1600]
    overlap = 0.5
    params = [channels, thresholds, overlap]

    # tic = time.time()
    # colocalise_movie(movie, params)
    # toc = time.time()
    # print(toc - tic)

    results = run(movie, params)
    # print(parameter_list[len(parameter_list) - 1].overlap)
    # tic2 = time.time()
    # pool.map(parallel_obcol, parameter_list)
    # toc2 = time.time()
    # print(toc2 - tic2)

    # movie = imread(path)
    # im = movie[0, 0, :, :]
    # print(im.shape)
    # thresholds = [100, 250]
    # channels = [2, 0]
    # l, p = locate(im, 1000)
    # # print(p.patches[0].pixels)
    # # print(p.patches[0].pixels.T)
    # movie = Movie(path)
    # first = movie[0]
    # print(type(first))
    print(len(results))
    result = results[0]
    plt.figure()
    plt.imshow(result.mono_labels(2), interpolation='nearest')
    plt.show()

