#-*- coding: utf-8 -*-

"""
abstraction for video processor
"""


class VideoProcessor(object):

    def __init__(self, classifier):
        self.classifier = classifier
        self.n_frames = 0
        self.time_per_frame = []
