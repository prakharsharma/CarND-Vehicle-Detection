#-*- coding: utf-8 -*-

"""
abstraction for image processor
"""

import time

import numpy as np

import matplotlib.pyplot as plt

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

    def get_drawn_detections(self, diagnostic_mode=False):
        """returns image with bbox drawn around detected vehicles"""

        if diagnostic_mode and self.drawn_all_detections is None:
            self.drawn_all_detections = utils.draw_boxes(
                self.original, self.hot_windows, color=(0, 0, 255), thick=6)

        if self.heatmap is None:
            heatmap_image =\
                np.zeros_like(self.original[:, :, 0]).astype(np.float)
            self.heatmap = utils.add_heat(heatmap_image, self.hot_windows)

        if self.thresholded_heatmap is None:
            self.thresholded_heatmap = utils.apply_threshold(
                self.heatmap, config.heatmap_threshold)

        labels = utils.create_labels(self.thresholded_heatmap)

        self.drawn_vehicle_detections = utils.draw_labeled_bboxes(
            self.original, labels)


def save_diagnostic_plot(image, fname):
    """draws diagnostic plot"""

    image.get_drawn_detections(diagnostic_mode=True)

    nrows = 5
    f, axarr = plt.subplots(nrows, 1, figsize=(24, 24))

    axarr[0].imshow(utils.convert_color(image.original))
    axarr[0].set_title('Original image', fontsize=20)

    axarr[1].imshow(utils.convert_color(image.drawn_all_detections))
    axarr[1].set_title('All detected windows', fontsize=20)

    axarr[2].imshow(image.heatmap, cmap='gray')
    axarr[2].set_title('Heatmap', fontsize=20)

    axarr[3].imshow(image.thresholded_heatmap, cmap='gray')
    axarr[3].set_title('Thresholded heatmap', fontsize=20)

    axarr[4].imshow(utils.convert_color(image.drawn_vehicle_detections))
    axarr[4].set_title('Detected cars', fontsize=20)

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
        # fname = fname or img
        # parts = list(filter(None, fname.split('/')))
        # fname = parts[-1]
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

    # window_img = utils.draw_boxes(img, hot_windows,
    #                               color=(0, 0, 255), thick=6)
    #
    # heatmap_image = np.zeros_like(img[:, :, 0]).astype(np.float)
    # heatmap = utils.add_heat(heatmap_image, hot_windows)
    # thresholded_heatmap = utils.apply_threshold(heatmap,
    #                                             config.heatmap_threshold)
    # labels = utils.create_labels(thresholded_heatmap)
    # final_detection = utils.draw_labeled_bboxes(img, labels)
    #
    # if not vid_mode and config.debug:
    #     fname = fname or '{}.jpg'.format(str(int(time.time() * 1000)))
    #     fname = '{}/{}'.format(config.output_dir, fname)
    #     d = {
    #         'original': img,
    #         'all_windows': window_img,
    #         'heatmap': heatmap,
    #         'thresholded_heatmap': thresholded_heatmap,
    #         'final_detection': final_detection
    #     }
    #     draw_diagnostic_plot(d, fname)
    #
    # return final_detection

    return image
