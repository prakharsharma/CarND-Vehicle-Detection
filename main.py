#-*- coding: utf-8 -*-

"""
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a
labeled training set of images and train a classifier Linear SVM classifier.

* Optionally, you can also apply a color transform and append binned color
features, as well as histograms of color, to your HOG feature vector.

* Note: for those first two steps don't forget to normalize your features and
randomize a selection for training and testing.

* Implement a sliding-window technique and use your trained classifier to
search for vehicles in images.

* Run your pipeline on a video stream (start with the test_video.mp4 and later
implement on full project_video.mp4) and create a heat map of recurring
detections frame by frame to reject outliers and follow detected vehicles.

* Estimate a bounding box for vehicles detected.
"""

import cv2
import glob

import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import time

import config

import utils

from vehicle_classifier import VehicleClassifier


def draw_diagnostic_plot(data, fname):
    """draws diagnostic plot"""

    nrows = len(data.keys())
    f, axarr = plt.subplots(nrows, 1, figsize=(24, 24))
    # f.tight_layout()

    axarr[0].imshow(utils.convert_color(data['original']))
    axarr[0].set_title('Original image', fontsize=20)

    axarr[1].imshow(utils.convert_color(data['all_windows']))
    axarr[1].set_title('All detected windows', fontsize=20)

    axarr[2].imshow(data['heatmap'], cmap='gray')
    axarr[2].set_title('Heatmap', fontsize=20)

    axarr[3].imshow(data['thresholded_heatmap'], cmap='gray')
    axarr[3].set_title('Thresholded heatmap', fontsize=20)

    axarr[4].imshow(utils.convert_color(data['final_detection']))
    axarr[4].set_title('Detected cars', fontsize=20)

    f.subplots_adjust(left=0., right=1, top=0.9, bottom=0.2)
    f.savefig(fname)


def detect_vehicles(img, classifier, vid_mode=False, fname=None):
    """
    detect vehicles on the given image

    :param img: given image
    :param feature_extractor: extractor
    :param classifier: classifier

    :return: list of bounding boxes
    """

    if isinstance(img, str):  # this is a file
        fname = fname or img
        parts = list(filter(None, fname.split('/')))
        fname = parts[-1]
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

    window_img = utils.draw_boxes(img, hot_windows,
                                  color=(0, 0, 255), thick=6)

    heatmap_image = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatmap = utils.add_heat(heatmap_image, hot_windows)
    thresholded_heatmap = utils.apply_threshold(heatmap,
                                                config.heatmap_threshold)
    labels = utils.create_labels(thresholded_heatmap)
    final_detection = utils.draw_labeled_bboxes(img, labels)

    if not vid_mode and config.debug:
        fname = fname or '{}.jpg'.format(str(int(time.time() * 1000)))
        fname = '{}/{}'.format(config.output_dir, fname)
        d = {
            'original': img,
            'all_windows': window_img,
            'heatmap': heatmap,
            'thresholded_heatmap': thresholded_heatmap,
            'final_detection': final_detection
        }
        draw_diagnostic_plot(d, fname)

    return final_detection


def main():
    classifier = VehicleClassifier()
    classifier.load()

    # process images
    images = glob.glob('./test_images/*.jpg')
    for img in images:
        detect_vehicles(img, classifier, vid_mode=False, fname=None)

    # process video


if __name__ == "__main__":
    main()
