#-*- coding: utf-8 -*-

"""
abstraction for video processor
"""


class VideoProcessor(object):

    def __init__(self, classifier):
        self.classifier = classifier
        self.n_frames = 0
        self.time_per_frame = []
        self.past_frames = []

    def process_frame(self, image):
        """process current frame and returns image with detections drawn"""

        # TODO:
        pass
