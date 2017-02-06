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

import glob
import time

import numpy as np

from moviepy.editor import VideoFileClip

import config

from image_processor import detect_vehicles
from vehicle_classifier import VehicleClassifier
from video_processor import VideoProcessor


def process_video_frame(processor):

    def _process_frame(frame):
        t0 = time.time()
        retVal = detect_vehicles(frame, processor.classifier, vid_mode=True)
        t1 = time.time()
        processor.time_per_frame.append(t1-t0)
        processor.n_frames += 1
        return retVal

    return _process_frame


def detect_in_video(vid_file, classifier):
    """detect vehicles in a video stream"""

    parts = list(filter(None, vid_file.split('/')))
    name = parts[-1]
    out_fspath = '{}/{}'.format(config.output_dir, name)

    processor = VideoProcessor(classifier)

    clip = VideoFileClip(vid_file, audio=False)
    t0 = time.time()
    out_clip = clip.fl_image(process_video_frame(processor))
    out_clip.write_videofile(
        out_fspath,
        audio=False
    )
    t1 = time.time()
    timimg_info = np.asarray(processor.time_per_frame, dtype=np.float32)
    print(
        "Processed {} frames in total {:.2f} seconds, with {:.2f} fps and "
        "{:.2f} seconds in mean to process a frame".format(
            processor.n_frames,
            t1 - t0,
            processor.n_frames/(t1 - t0),
            np.mean(timimg_info)
        )
    )


def main():
    classifier = VehicleClassifier()
    classifier.load()

    # process images
    images = glob.glob('./test_images/*.jpg')
    for img in images:
        detect_vehicles(img, classifier, vid_mode=False, fname=None)

    # process video
    detect_in_video('./test_video.mp4', classifier)


if __name__ == "__main__":
    main()
