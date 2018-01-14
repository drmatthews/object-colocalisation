# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
from openpyxl import load_workbook
import ijroi
import trackpy as tp

import utils


def get_writer(path):
    book = load_workbook(path)
    writer = pd.ExcelWriter(path, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    return writer


def write_slopes_to_file(path, slope_df, sheetname):
    # slope_df = pd.DataFrame(slopes)
    # slope_df.columns = ['particle id', 'slope', 'good']
    writer = get_writer(path)
    slope_df.to_excel(writer, sheet_name=sheetname, startcol=8, index=False)
    writer.save()


def write_tracks_to_file(path, tracks, sheetname):
    writer = get_writer(path)
    tracks.to_excel(writer, sheet_name=sheetname, index=False)
    writer.save()


def import_tracks(path):
    if path.endswith('xlsx'):
        sheets = ['red', 'green']
        tracks = {}
        for sheet in sheets:
            sheetname = '{} tracks'.format(sheet)
            print('reading sheet {}'.format(sheetname))
            try:
                tracks[sheet] = pd.read_excel(path, sheetname=sheetname)
            except: # noqa
                print('no tracks in file - now running tracking')
                data = pd.read_excel(path, sheetname=sheet)
                tracks[sheet] = utils.redo_tracking(data)
        return tracks
    else:
        raise ValueError("Input data must be in Excel format")


def linear_regress(data, log=True, clip=None, r2=0.8, **kwargs):
    """Fit a 1st order polynomial by doing first order polynomial fit."""
    ys = pd.DataFrame(data)
    values = pd.DataFrame(index=['slope', 'intercept', 'r2', 'good'])
    good = False
    fits = {}
    for col in ys:
        if clip:
            y = ys[col].dropna()
            limit = np.arange(1, np.min(((1 + clip), len(y.index))))
            y = ys.loc[limit, [col]][col]
            x = pd.Series(y.index.values, index=y.index, dtype=np.float64)
        else:
            y = ys[col].dropna()
            x = pd.Series(y.index.values, index=y.index, dtype=np.float64)
        if log:
            slope, intercept, r, p, stderr = \
                stats.linregress(np.log(x), np.log(y))
            if r**2 > r2:
                good = True
            values[col] = [slope, np.exp(intercept), r**2, good]
            fits[col] = x.apply(lambda x: np.exp(intercept) * x**slope)
        else:
            slope, intercept, r, p, stderr = \
                stats.linregress(x, y)
            if r**2 > r2:
                good = True
            values[col] = [slope, intercept, r**2, good]
            fits[col] = x.apply(lambda x: intercept * x**slope)
    values = values.T
    fits = pd.concat(fits, axis=1)
    return (values, fits)


def velocity_field(tracks, plot_quiver=True):

    for channel in tracks.iterkeys():
        traj = tracks[channel]
        data = pd.DataFrame()
        for item in set(traj.particle):
            sub = traj[traj.particle == item]
            dvx = np.diff(sub.x)
            dvy = np.diff(sub.y)
            sub_loop = zip(sub.x[:-1], sub.y[:-1], dvx, dvy, sub.frame[:-1],)
            for x, y, dx, dy, frame in sub_loop:
                data = data.append([{
                    'dx': dx,
                    'dy': dy,
                    'x': x,
                    'y': y,
                    'frame': frame,
                    'particle': item,
                }])

        if plot_quiver:
            i = 20
            d = data[data.frame == i]
            plt.figure()
            plt.quiver(
                d.x, d.y, d.dx, -d.dy,
                pivot='middle',
                headwidth=4,
                headlength=6,
                color='red')
            plt.axis('off')
            plt.show()


def rate_of_change_distance(traj, r2=0.8):

    slope = []
    for tid, t in traj.groupby('particle'):
        t_indexed = t.set_index(['frame'], drop=False)
        nan_array = np.empty(
            (t_indexed['frame'].max(), len(t_indexed.columns)))
        nan_array[:] = np.nan
        tnan = pd.DataFrame(nan_array,
                            index=range(0, t_indexed['frame'].max()))
        tnan.columns = t.columns
        t_filled = t_indexed.combine_first(tnan).fillna(method='bfill')

        dfits = linear_regress(t_filled['distance'].values, log=False, r2=r2)
        slope.append((dfits[0]['slope'].values[0], dfits[0]['good'].values[0]))

    f = range(0, len(slope))
    slopes = [[f[idx], s[0], s[1]] for idx, s in enumerate(slope)]
    slopes_df = pd.DataFrame(slopes)
    slopes_df.columns = ['particle id', 'slope', 'good']
    return slopes_df


def calculate_distance(tracks, tracks_path, reference):

    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distances = {}
    slopes = {}
    for traj_key in tracks.iterkeys():
        traj = tracks[traj_key]
        #num_nucleus_frames = len(reference) - 1
        #num_vesicle_frames = traj['frame'].max()
        #if num_nucleus_frames != num_vesicle_frames:
        #    print('The number of nucleus frames does'
        #          ' not match the number of vesicle frames')
        #    raise ValueError

        print('calculating distance and fitting {} tracks'.format(traj_key))
        dist_vals = []
        slopes[traj_key] = []
        if traj is not None:
            for tid, track in traj.groupby('particle'):
                for nid, node in track.iterrows():
                    frame = node['frame']
                    ref_x = reference['x'].iloc[int(frame) - 1].item()
                    ref_y = reference['y'].iloc[int(frame) - 1].item()
                    node_x = node['x'].item()
                    node_y = node['y'].item()
                    d = distance(node_x, node_y, ref_x, ref_y)
                    dist_vals.append(d)

            try:
                traj.insert(len(traj.columns), 'distance', dist_vals)
            except: # noqa
                pass

            slopes[traj_key] = rate_of_change_distance(traj, r2=0.5)

            sheetname = '{} tracks'.format(traj_key)
            write_tracks_to_file(tracks_path, traj, sheetname)
            write_slopes_to_file(tracks_path, slopes[traj_key], sheetname)

            d_df = pd.concat([traj['frame'], traj['particle']], axis=1)
            d_df.insert(len(d_df.columns), 'distance', dist_vals)
            distances[traj_key] = d_df
    return (distances, slopes)


def distance_to_reference(tracks_path, reference_path, is_manual=True):
    if is_manual:
        # import the nucleus centroids from TrackMate CSV
        data = pd.read_csv(reference_path, header=0)
        ref = data[['X', 'Y']]
        ref.columns = ['x', 'y']
        ref.index += 1
        ref.index.name = 'frame'
    else:
        # import the reference centroids from ImageJ rois
        fdir = 'rois'
        rois = []
        for filename in sorted(os.listdir(fdir)):
            if 'roi'in filename:
                fpath = os.path.join(fdir, filename)
                with open(fpath, "rb") as f:
                    roi = ijroi.read_roi(f)
                    rois.append(roi[0])

        ref = pd.DataFrame(
            np.cumsum(np.diff(np.array(rois), axis=0), axis=0))
        ref.columns = ['y', 'x']
        ref.index += 1
        ref.index.name = 'frame'
    try:
        distances, slopes = calculate_distance(
            import_tracks(tracks_path), tracks_path, ref)
        return (distances, slopes, ref)
    except ValueError:
        return (None, None, None)


def batch_distance_to_reference(input_dir, is_manual=True):
    results = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".xlsx"):
            print('processing {}'.format(filename))
            basename = filename[0: filename.index('_')]
            tracks_path = os.path.join(input_dir, filename)
            ref_filename = (u'Results from {} in '
                            '{} per sec.csv'.format(
                                basename, '\xc2\xb5m'.decode('utf8')))
            ref_path = os.path.join(input_dir, ref_filename)
            d, s, n = distance_to_reference(
                tracks_path, ref_path, is_manual=is_manual)
            if d is not None:
                results[filename] = (d, s, n)
            sys.stdout.flush()
    return results


def distance_and_speed(tracks, mpp=1, fps=1):

    for channel in tracks.iterkeys():
        traj = tracks[channel]
        diagonal_distance = []
        diagonal_speed = []
        track_distance = []
        track_speed = []
        for pid, particle in traj.groupby('particle'):
            diagonal = tp.diagonal_size(particle) * mpp
            diagonal_distance.extend(
                [diagonal * mpp for i in range(len(particle.index))])
            diagonal_speed.extend(
                [diagonal * fps for i in range(len(particle.index))])

            sq_distx = np.square(np.diff(particle.x) * mpp)
            sq_disty = np.square(np.diff(particle.y) * mpp)
            dist = np.sum(np.sqrt(np.add(sq_distx, sq_disty)), dtype=np.float)
            track_distance.extend(
                [dist for i in range(len(particle.index))])
            track_speed.extend(
                [dist * fps for i in range(len(particle.index))])

        if 'diagonal distance' not in traj.columns:
            traj.insert(
                len(traj.columns), 'diagonal distance', diagonal_distance)

        if 'diagonal speed' not in traj.columns:
            traj.insert(
                len(traj.columns), 'diagonal speed', diagonal_speed)

        if 'track distance' not in traj.columns:
            traj.insert(
                len(traj.columns), 'track distance', track_distance)

        if 'track speed' not in traj.columns:
            traj.insert(
                len(traj.columns), 'track speed', track_speed)

        tracks[channel] = traj

    return tracks


def batch_distance_and_speed(input_dir, mpp=1, fps=1, sheetname=None):

    for filename in os.listdir(input_dir):
        if filename.endswith(".xlsx"):
            print('processing {}'.format(filename))
            tracks_path = os.path.join(input_dir, filename)

            tracks = distance_and_speed(
                import_tracks(tracks_path), mpp, fps)

            tracks_sheetname = 'tracks'
            if sheetname:
                tracks_sheetname = sheetname

            for channel in tracks.iterkeys():
                ts = '{0} {1}'.format(channel, tracks_sheetname)
                traj = tracks[channel]
                write_tracks_to_file(tracks_path, traj, ts)

    return tracks


if __name__ == '__main__':
    tracks_dir = ('/home/daniel/Documents/programming/'
                  'Image Processing/object_colocalisation/test/')
    tracks_path = tracks_dir + 'KS 1_channels_10_obcol.xlsx'
    nucleus_path = tracks_dir + 'Results from KS 1 in µm per sec.csv'

    batch_distance_and_speed(tracks_dir)
