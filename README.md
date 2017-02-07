# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/rykeenan/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.

[//]: # (References)
[carNotCarExample]: ./output_images/car_not_car.jpg
[hogExample]: ./output_images/HOG_features.jpg
[allDetectedWindows]: ./output_images/all_detected_windows.jpg
[test1PipelineOut]: ./output_images/test1.jpg
[video1]: ./project_video.mp4
[projectVideoOut]: https://youtu.be/NQ5feKnc66E "video result"
[hogFeaturesExtractionFunc]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/utils.py#L77 "hog feature extraction"
[linearSVMTrainFunc]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/vehicle_classifier.py#L84 "SVM train"
[slidingWindowFunc]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/utils.py#L197 "sliding window"
[detectVehiclesFunc]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/image_processor.py#L84 "detect vehicles func"
[vidDuplicateDetectionAndFalsePositives]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/video_processor.py#L21 "false positives in video"
[hyperparamsTuningSheet]: https://docs.google.com/spreadsheets/d/1Dr-YBUIlQrTcv4zUpHbFt4fwoTGrAjH8Wvq2K_hcIbY/edit?usp=sharing "hyper params tuning sheet"
[featureExtractorClass]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/feature_extractor.py#L14 "feature extractor"
[detectVehiclesInImageFunc]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/image_processor.py#L134 "detect vehicles"
[videoProcessorClass]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/video_processor.py#L13 "video processor"
[config]: https://github.com/prakharsharma/CarND-Vehicle-Detection/blob/master/config.py "config"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images (in the provided labeled data set). I use a combination of spatial binning,
 histogram of colors and HOG features. Please refer [`FeatureExtractor`][featureExtractorClass]. HOG features in particular are extracted using
 `skimage.feature.hog` function at method [`get_hog_features`][hogFeaturesExtractionFunc].

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][carNotCarExample]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
 I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hogExample]

####2. Explain how you settled on your final choice of HOG parameters.

Different combinations of `orientation` (8, 9, 12), `pixels_per_cell` (8, 16) and `cell_per_block` (2, 4, 8) were tried along with different
 values for `spatial_bins`, `hist_bins`. Finally following values were used (to optimize for accuracy and training time): -

```
# size for spatial bins
spatial_size = (32, 32)

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
hog_channel = 'ALL'
```

[Doc][hyperparamsTuningSheet] captures various experiments done to arrive at the above values.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a combination of spatial, color histogram and HOG features. Training is done by [`train`][linearSVMTrainFunc].

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

Sliding window is implemented by [`slide_window`][slidingWindowFunc]. The method takes minimum window size, maximum window size,
step size (how much to increase window size between iterations) and fraction of overlap between windows and returns all possible windows
to search for cars. Following values are used: -
 
```
min_window = 32
max_window = 200  # other values tried were 300, 400. But, got better results with 200
xy_overlap = (0.5, 0.5)  # tried (0.2, 0.2) and (0.8, 0.8) and got best results using (0.5, 0.5)
step_size = 30  # tried 10, and 20.
```

I analyzed the test images to get an idea of possible sizes of bounding box that can accommodate cars. That gave guidance for min and max window
 size. After that I played with different values of min, max and step size to arrive at the one which provided the best trade-off between
 accuracy (snug fit around cars) and number of windows to search.

The strategy resulted in the following detections on one of the test images (`./test_images/test1.jpg`).

![alt text][allDetectedWindows]

####2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to try to minimize false positives and reliably detect cars?

End to end detection of vehicles in an image is provided by [`detect_cars_in_image`][detectVehiclesInImageFunc]. The function does the
 following: -

1. Extracts features (spatial binning + histogram of colors + HOG),
1. Uses multi scale windows to find list of windows that can possibly have cars in them, and
1. Creates heatmap and thresholds it to minimize false positives and remove duplicate detections.

Following image shows output of different stages of the pipeline on one of the test images (`./test_images/test1.jpg`). 

![alt text][test1PipelineOut]
---

### Video Implementation

####1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/NQ5feKnc66E)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

[`VideoProcessor`][videoProcessorClass] keeps track of the 'hot windows' (windows with detections) found in the past few frames. For a new
 frame, it does the following: -
 
1. Detect hot windows for the current frame,
1. Creates a heatmap using the hot windows in the current frame and past few frames,
1. Thesholds the heatmap (to deal with duplicate detections and false positives),
1. Identifies individual blobs in the heatmap using `scipy.ndimage.measurements.label()`, and
1. Assumes each of the blob to correspond to a vehicle and then draws bounding boxes around the blobs.

Behavior of the function can be tuned using the following [configuration][config] parameters

```
# video lookback
video_lookback_frame_count = 5

# video heatmap threshold
video_heatmap_threshold = 4
```

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

I really liked working on this project. Pretty challenging, but, lot of fun! Especially the fact that I got to apply both ML and computer vision
techniques. The course material for this project was pretty good! I had to watch most of the lectures twice, but, they were really helpful.
 P4 had built some familiarity with computer vision techniques. That was definitely helpful. I am quite proud of what I have put together, but,
 there are quite a few areas of improvement: - 

1. Detections in video are sometimes choppy.
1. Bounding box should fit more tightly around the cars.
1. Faster performance on video.
1. Better detections on challenge videos.
1. Hook up lane detection from P4 into a combined pipeline!

Few things to try out for making the above improvements: - 

1. Limit the number of candidate windows to look for detections (using a max number of windows).
1. Perform classification across candidate windows in parallel to speed up performance.
1. Compute HOG features for the whole image once rather than computing HOG features for small portions of the image.
1. Experiment with rectangular (not square) detection windows.
1. Using PCA or decision tree to find out which features are important and then just using the important features.
1. Fine tuning various hyper params, (may be even learning them)
    1. min and max size of windows,
    1. step size for increasing window size,
    1. maximum number of windows to search,
    1. how many past frames to look back while building a heatmap
    1. threshold for heatmap
1. Build a real time diagnostic view for detection and tracking. Few students in Slack did this and the output is really cool!
1. Try out CNN for classification.


**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
