#!/usr/bin/python
# -*- coding:utf-8 -*-

'''

Team ID: Video Stitch
Author List: Chunshan Bai

'''

import cv2
import numpy as np
import imutils
import tqdm
import os
from moviepy.editor import ImageSequenceClip
import time
from openni import openni2


def __init__(self, left_video_in_path, right_video_in_path, video_out_path, video_out_width=800, display=False):
    # Initialize arguments
    self.left_video_in_path = left_video_in_path
    self.right_video_in_path = right_video_in_path
    self.video_out_path = video_out_path
    self.video_out_width = video_out_width
    self.display = display

    # Initialize the saved homography matrix
    self.saved_homo_matrix = None

def stitch(images, ratio=0.75, reproj_thresh=5.0):
    saved_homo_matrix = None
    # Unpack the images
    (image_b, image_a) = images

    # If the saved homography matrix is None, then we need to apply keypoint matching to construct it
    if saved_homo_matrix is None:
        # Detect keypoints and extract
        (keypoints_a, features_a) = detect_and_extract(image_a)
        (keypoints_b, features_b) = detect_and_extract(image_b)

        # Match features between the two images
        matched_keypoints = match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

        # If the match is None, then there aren't enough matched keypoints to create a panorama
        if matched_keypoints is None:
            return None

        # Save the homography matrix
        saved_homo_matrix = matched_keypoints[1]

    # Apply a perspective transform to stitch the images together using the saved homography matrix
    output_shape = (image_a.shape[1] + image_b.shape[1], image_a.shape[0])
    result = cv2.warpPerspective(image_a, saved_homo_matrix, output_shape)
    result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b

    # Return the stitched image
    return result

def detect_and_extract(image):
    # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
    descriptor = cv2.xfeatures2d.SIFT_create() #opencv>3.0
    # descriptor = cv2.SIFT() #opencv2.*
    (keypoints, features) = descriptor.detectAndCompute(image, None)

    # Convert the keypoints from KeyPoint objects to numpy arrays
    keypoints = np.float32([keypoint.pt for keypoint in keypoints])

    # Return a tuple of keypoints and features
    return (keypoints, features)

def match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh):
    # Compute the raw matches and initialize the list of actual matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(features_a, features_b, k=2)
    matches = []

    for raw_match in raw_matches:
        # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
        if len(raw_match) == 2 and raw_match[0].distance < raw_match[1].distance * ratio:
            matches.append((raw_match[0].trainIdx, raw_match[0].queryIdx))

    # Computing a homography requires at least 4 matches
    if len(matches) > 4:
        # Construct the two sets of points
        points_a = np.float32([keypoints_a[i] for (_, i) in matches])
        points_b = np.float32([keypoints_b[i] for (i, _) in matches])

        # Compute the homography between the two sets of points
        (homography_matrix, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_thresh)

        # Return the matches, homography matrix and status of each matched point
        return (matches, homography_matrix, status)

    # No homography could be computed
    return None

def draw_matches(image_a, image_b, keypoints_a, keypoints_b, matches, status):
    # Initialize the output visualization image
    (height_a, width_a) = image_a.shape[:2]
    (height_b, width_b) = image_b.shape[:2]
    visualisation = np.zeros((max(height_a, height_b), width_a + width_b, 3), dtype="uint8")
    visualisation[0:height_a, 0:width_a] = image_a
    visualisation[0:height_b, width_a:] = image_b

    for ((train_index, query_index), s) in zip(matches, status):
        # Only process the match if the keypoint was successfully matched
        if s == 1:
            # Draw the match
            point_a = (int(keypoints_a[query_index][0]), int(keypoints_a[query_index][1]))
            point_b = (int(keypoints_b[train_index][0]) + width_a, int(keypoints_b[train_index][1]))
            cv2.line(visualisation, point_a, point_b, (0, 255, 0), 1)

    # return the visualization
    return visualisation

def run(left,right,fps,video_out_path):
    video_out_width = 400
    display = True
    frames = []
    success = True
    while (success):
        # Stitch the frames together to form the panorama
        stitched_frame = stitch([left, right], ratio=0.75, reproj_thresh=4.0)

        # No homography could not be computed
        if stitched_frame is None:
            print("[INFO]: Homography could not be computed!")
            break

        # Add frame to video
        stitched_frame = imutils.resize(stitched_frame, width=video_out_width)
        frames.append(stitched_frame)

        if display:
            # Show the output images
            cv2.imshow("Result", stitched_frame)

        # If the 'q' key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

        # Save video
        print('[INFO]: Saving {} in {}'.format(video_out_path.split('/')[-1],
                                           os.path.dirname(video_out_path)))
        clip = ImageSequenceClip(frames, fps=1000)
        clip.write_videofile(video_out_path, codec='mpeg4', audio=False, progress_bar=True, verbose=False)
        print('[INFO]: {} saved'.format(video_out_path.split('/')[-1]))

# Main script
if __name__ == '__main__':

    # openni2.initialize()  # can also accept the path of the OpenNI redistribution
    # dev = openni2.Device.open_any()
    # print(dev.get_device_info())
    # depth_stream = dev.create_depth_stream()
    # color_stream = dev.create_color_stream()
    #
    # frame = depth_stream.read_frame()
    # right = color_stream.read_frame()

    camera1 = cv2.VideoCapture(0)
    camera2 = cv2.VideoCapture(1)

    ok, left = camera1.read()
    _, right = camera2.read()

    fps = int(camera1.get(cv2.CAP_PROP_FPS))

    video_out_path='./output/cameral_video.avi'

    # start = time.time()


    run(left,right,fps,video_out_path)

    # end = time.time()
    # total_time = end - start
    # print("total_time: ", total_time)
