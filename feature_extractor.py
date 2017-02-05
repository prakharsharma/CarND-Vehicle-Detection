#-*- coding: utf-8 -*-

"""
abstraction for feature extraction
"""

import collections

import config

import utils


class FeatureExtractor(object):

    def __init__(self):
        pass

    def extract_features(self, image_list):
        """extract features for the input images"""

        if not isinstance(image_list, collections.Iterable):
            image_list = [image_list]

        features = utils.extract_features(
            image_list,
            color_space=config.color_space,
            spatial_size=config.spatial_size,
            hist_bins=config.hist_bins,
            orient=config.orient,
            pix_per_cell=config.pix_per_cell,
            cell_per_block=config.cell_per_block,
            hog_channel=config.hog_channel,
            spatial_feat=config.spatial_feat,
            hist_feat=config.hist_feat,
            hog_feat=config.hog_feat
        )

        return features

    @classmethod
    def get_feature_extractor(cls):
        return cls()
