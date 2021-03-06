#-*- coding: utf-8 -*-

"""
abstraction for image processor
"""

import matplotlib.pyplot as plt
import numpy as np

import config
import utils


class Image(object):
    """abstraction for a frame/image"""

    def __init__(self, original, hot_windows):
        self.original = original
        self.hot_windows = hot_windows
        self.drawn_all_detections = None
        self.heatmap = None
        self.thresholded_heatmap = None
        self.drawn_vehicle_detections = None

    def get_drawn_detections(self, diagnostic_mode=False, vid_mode=False):
        """returns image with bbox drawn around detected vehicles"""

        if diagnostic_mode and not vid_mode and \
                self.drawn_all_detections is None:
            self.drawn_all_detections = utils.draw_boxes(
                self.original, self.hot_windows, color=(0, 0, 255), thick=6)

        if self.heatmap is None:
            heatmap_image =\
                np.zeros_like(self.original[:, :, 0]).astype(np.float)
            self.heatmap = utils.add_heat(heatmap_image, self.hot_windows)

        if self.thresholded_heatmap is None:
            self.thresholded_heatmap = utils.apply_threshold(
                self.heatmap, config.heatmap_threshold)

        if self.drawn_vehicle_detections is None:
            labels = utils.create_labels(self.thresholded_heatmap)
            self.drawn_vehicle_detections = utils.draw_labeled_bboxes(
                self.original, labels)


def save_diagnostic_plot(image, fname, vid_mode=False):
    """draws diagnostic plot"""

    image.get_drawn_detections(diagnostic_mode=True, vid_mode=vid_mode)

    nrows = 5
    if vid_mode:
        nrows = 4

    f, axarr = plt.subplots(nrows, 1, figsize=(24, 24))

    i = 0
    axarr[i].imshow(utils.convert_color(image.original))
    axarr[i].set_title('Original image', fontsize=20)

    if not vid_mode:
        i += 1
        axarr[i].imshow(utils.convert_color(image.drawn_all_detections))
        axarr[i].set_title('All detected windows', fontsize=20)

    i += 1
    axarr[i].imshow(image.heatmap, cmap='gray')
    axarr[i].set_title('Heatmap', fontsize=20)

    i += 1
    axarr[i].imshow(image.thresholded_heatmap, cmap='gray')
    axarr[i].set_title('Thresholded heatmap', fontsize=20)

    i += 1
    axarr[i].imshow(utils.convert_color(image.drawn_vehicle_detections))
    axarr[i].set_title('Detected cars', fontsize=20)

    f.subplots_adjust(left=0., right=1, top=0.9, bottom=0.2)
    f.savefig(fname)


def detect_vehicles(img, classifier):
    """
    detect vehicles on the given image

    :param img: given image
    :param feature_extractor: extractor
    :param classifier: classifier

    :return: list of bounding boxes
    """

    if isinstance(img, str):  # this is a file
        img = utils.read_image(img)
    elif not isinstance(img, np.ndarray):
        raise ValueError('bad image: {}'.format(img))

    h, w = img.shape[:2]
    windows = utils.slide_window(
        img,
        y_start_stop=[h/2, h],
        min_size=config.min_window_size,
        max_size=config.max_window_size,
        step_size=config.step_size_for_window
    )
    print("searching for cars in {} windows in image".format(len(windows)))

    hot_windows = utils.search_windows(
        img,
        windows,
        classifier.model,
        classifier.scaler,
        color_space=config.color_space,
        spatial_size=config.spatial_size,
        hist_bins=config.hist_bins,
        hist_range=config.hist_range,
        orient=config.orient,
        pix_per_cell=config.pix_per_cell,
        cell_per_block=config.cell_per_block,
        hog_channel=config.hog_channel,
        spatial_feat=config.spatial_feat,
        hist_feat=config.hist_feat,
        hog_feat=config.hog_feat
    )
    print("found car in {} windows".format(len(hot_windows)))

    image = Image(img, hot_windows)

    return image


def detect_cars_in_image(img, classifier):
    """provided end to end detection of cars in an image/frame"""

    image = detect_vehicles(img, classifier)
    image.get_drawn_detections()

    return image.drawn_vehicle_detections
