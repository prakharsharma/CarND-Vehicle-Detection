#-*- coding: utf-8 -*-

"""
abstraction for video processor
"""

import numpy as np

import config
import utils


class VideoProcessor(object):

    def __init__(self, classifier):
        self.classifier = classifier
        self.n_frames = 0
        self.time_per_frame = []
        self.past_frames = []

    def process_frame(self, image):
        """process current frame and returns image with detections drawn"""

        self.past_frames.append(image)
        if len(self.past_frames) > config.video_lookback_frame_count:
            self.past_frames.pop(0)

        heatmap = np.zeros_like(image.original[:, :, 0]).astype(np.float)
        for img in self.past_frames:
            heatmap = utils.add_heat(heatmap, img.hot_windows)
        thresholded_heatmap = utils.apply_threshold(
            heatmap,
            config.video_heatmap_threshold
        )
        labels = utils.create_labels(thresholded_heatmap)
        final_detection = utils.draw_labeled_bboxes(image.original, labels)

        image.heatmap = heatmap
        image.thresholded_heatmap = thresholded_heatmap
        image.drawn_vehicle_detections = final_detection
