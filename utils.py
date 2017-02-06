#-*- coding: utf-8 -*-

"""
utility methods

Most of the methods here are compiled after working through the Vehicle
detection lesson
"""

import cv2
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.feature import hog
from scipy.ndimage.measurements import label


to_int_pair = lambda x: (int(x[0]), int(x[1]))


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, in_place=False):
    """draw bounding boxes on the image"""

    if in_place:
        draw_img = img
    else:
        draw_img = np.copy(img)

    for bbox in bboxes:
        cv2.rectangle(draw_img, to_int_pair(bbox[0]), to_int_pair(bbox[1]),
                      color, thick)
    return draw_img


def read_image(fname):
    """reads in image from file system"""
    return cv2.imread(fname)


def convert_color(img, color_space='RGB'):
    if color_space == 'RGB':
        color = cv2.COLOR_BGR2RGB
    elif color_space == 'LUV':
        color = cv2.COLOR_BGR2LUV
    elif color_space == 'HLS':
        color = cv2.COLOR_BGR2HLS
    elif color_space == 'YUV':
        color = cv2.COLOR_BGR2YUV
    elif color_space == 'YCrCb':
        color = cv2.COLOR_BGR2YCrCb
    else:
        raise ValueError
    return cv2.cvtColor(img, color)/255


def bin_spatial(img, size=(32, 32)):
    """compute binned color features"""

    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """compute color histogram features"""

    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0])
    )
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """extracts HOG features for the provided image"""

    if vis:
        features, hog_image = hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=True,visualise=vis,
            feature_vector=feature_vec
        )
        return features, hog_image
    else:
        features = hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=True,
            visualise=vis,
            feature_vector=feature_vec
        )
        return features


# Define a function to return HOG features and visualization
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    extracts spatial, color histogram and HOG features for the provided image
    """

    # 1) Define an empty list to receive features
    img_features = []

    # 2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space)
    # if color_space != 'RGB':
    #     if color_space == 'HSV':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     elif color_space == 'LUV':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    #     elif color_space == 'HLS':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #     elif color_space == 'YUV':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #     elif color_space == 'YCrCb':
    #         feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # else:
    #     feature_image = np.copy(img)

    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)

    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)

    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(
                    feature_image[:, :, channel],
                    orient,
                    pix_per_cell,
                    cell_per_block,
                    vis=False,
                    feature_vec=True
                ))
        else:
            hog_features = get_hog_features(
                feature_image[:, :, hog_channel],
                orient,
                pix_per_cell,
                cell_per_block,
                vis=False,
                feature_vec=True
            )

        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    """extracts features for a given list of images"""

    features = []

    for img in imgs:

        if isinstance(img, str):  # this is a file
            img = read_image(img)
        elif not isinstance(img, np.ndarray):
            raise ValueError('bad image: {}'.format(img))

        features.append(single_img_features(
            img, color_space, spatial_size, hist_bins, orient,
            pix_per_cell, cell_per_block, hog_channel, spatial_feat,
            hist_feat, hog_feat
        ))

    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 min_size=64, max_size=64, step_size=100,
                 xy_overlap=(0.5, 0.5)):

    if img is not None:
        h, w = img.shape[:2]

        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = w

        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = h

    window_list = []
    window_size = min_size

    while window_size <= max_size:

        xy_window = (window_size, window_size)

        x_incr = int(xy_window[0] * xy_overlap[0])
        y_incr = int(xy_window[1] * xy_overlap[1])

        y_iter = y_start_stop[0]
        while y_iter + xy_window[1] <= y_start_stop[1]:

            x_iter = x_start_stop[0]
            while x_iter + xy_window[0] <= x_start_stop[1]:
                top_left = (x_iter, y_iter)
                bottom_right = (x_iter + xy_window[0], y_iter + xy_window[1])
                window_list.append((top_left, bottom_right))
                x_iter += x_incr

            y_iter += y_incr

        window_size += step_size

    return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []

    # 2) Iterate over all windows in the list
    for window in windows:

        window = (to_int_pair(window[0]), to_int_pair(window[1]))

        # 3) Extract the test window from original image
        test_img = cv2.resize(
            img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
            (64, 64)
        )

        # 4) Extract features for that window using single_img_features()
        features = single_img_features(
            test_img,
            color_space=color_space,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            hog_channel=hog_channel,
            spatial_feat=spatial_feat,
            hist_feat=hist_feat,
            hog_feat=hog_feat
        )

        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # 6) Predict using your classifier
        prediction = clf.predict(test_features)

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # 8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    """build heatmap from list of detected bounding boxes"""

    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        box = (to_int_pair(box[0]), to_int_pair(box[1]))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold, in_place=False):
    """thresholds the heatmap"""

    if in_place:
        heatmap_img = heatmap
    else:
        heatmap_img = np.copy(heatmap)

    heatmap_img[heatmap_img <= threshold] = 0
    return heatmap_img


def create_labels(heatmap):
    """labels detected cars and their pixels"""

    return label(heatmap)


def draw_labeled_bboxes(img, labels, in_place=False):
    """draws clean boxes around final detections"""

    if in_place:
        draw_img = img
    else:
        draw_img = np.copy(img)

    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)

    return draw_img
