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

import numpy as np
import cv2

import config

import utils

from feature_extractor import FeatureExtractor
from vehicle_classifier import VehicleClassifier


def detect_vehicles(img, classifier, draw_bbox=False):
    """
    detect vehicles on the given image

    :param img: given image
    :param feature_extractor: extractor
    :param classifier: classifier

    :return: list of bounding boxes
    """

    h, w = img.shape[:2]
    windows = utils.slide_window(
        img,
        y_start_stop=[h/2, h]
    )
    print("searching for cars in {} windows in image".format(len(windows)))

    car_windows = utils.search_windows(
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
    print("found car in {} windows".format(len(car_windows)))

    if draw_bbox:
        # TODO: draw detected bboxes on the image and save the image
        pass


def main():
    classifier = VehicleClassifier()
    classifier.load_model()

    images = []
    for img in images:
        detect_vehicles(img, classifier)


if __name__ == "__main__":
    main()
