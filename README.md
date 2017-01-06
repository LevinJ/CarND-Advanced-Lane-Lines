# Self-Driving Car Engineer Nanodegree
# Computer Vision
## Project: Advanced Lane Finding

### Overview
The goals / steps of this project are the following:  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Final Result

The final pipleline has been successfully applied in both project video and challenge video. (https://youtu.be/JmJUB54CtmU)


### Camera Calibration

Using the calibration chessboard images provided in the repository, we successfully obtained the camera matrix and distortion coefficientss by leveraging fuctions like cv2.findChessboardCorners, cv2.calibrateCamera from OpencCV library.   

After camera matrix and distortion matrix are obained, we stored them in a pickle file so that we can quickly retreive them later on without having to calculate them afresh every time we use them to undistort images.

![Camera Calibration](https://github.com/LevinJ/CarND-Advanced-Lane-Lines/blob/master/camera_calibration.png)


class Calibrarte in file calibrate.py implemented camera calibration feature, the main external interface is the Calibrarte::undistort method

### Pipeline (single images)

### Pipeline (video)

### Reflection
