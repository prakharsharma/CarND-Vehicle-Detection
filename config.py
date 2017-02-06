#-*- coding: utf-8 -*-

"""
configuration
"""

# dump debug info
debug = True

# output dir
output_dir = './output_images'

# data params

# dir that holds labeled data for vehicles
# vehicle_data = './labeled-data/small/vehicles_smallset'
vehicles_data = './labeled-data/full/vehicles'

# dir that hold labeled data for non-vehicles
# non_vehicles_data = './labeled-data/small/non-vehicles_smallset'
non_vehicles_data = './labeled-data/full/non-vehicles'

# SVM hyper params

# save model to the file
model_path = './linearSVC.p'

# train/test split
test_train_ratio = 0.2

# cross-validation/test split
validation_test_ratio = None

# feature extractor params

# color space to use for feature extraction
color_space = 'YCrCb'

# use spatial features?
spatial_feat = True

# size for spatial bins
spatial_size = (32, 32)

# use Histogram features?
hist_feat = True

# number of histogram bins
hist_bins = 32

# histogram range
hist_range = (0, 256)

# number of orientations for HOG featurs
orient = 9

# pixels per cell for HOG features
pix_per_cell = 8

# cells per block for HOG features
cell_per_block = 2

# HOG channel
hog_channel = 0

# use HOG features
hog_feat = True

# heatmap threshold
heatmap_threshold = 2

# sliding window params

# min window size
min_window_size = 32

# max window size
max_window_size = 300

# window step size
step_size_for_window = 30

# video lookback
video_lookback_frame_count = 5

# video heatmap threshold
video_heatmap_threshold = 4
