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

import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import ImageSequenceClip
import time

# cv2.imwrite('/home/boyce/Desktop/BigData/canku/image/0828/image_car_1015/test/test_A.png', image)
# cv2.waitKey(0)

class VideoStitcher:
    def __init__(self, left_video_in_path, right_video_in_path, video_out_path, video_out_width=800, display=False):
        # Initialize arguments
        self.left_video_in_path = left_video_in_path
        self.right_video_in_path = right_video_in_path
        self.video_out_path = video_out_path
        self.video_out_width = video_out_width
        self.display = display

        # Initialize the saved homography matrix
        self.saved_homo_matrix = None

    def stitch(self, images, ratio=0.75, reproj_thresh=4.0):
        # Unpack the images
        (left_gray, right_gray, image_left, image_right) = images
        # cv2.namedWindow("right_gray", 0)
        # cv2.imshow("right_gray", right_gray)
        # cv2.waitKey(0)
        # cv2.namedWindow("image_right", 0)
        # cv2.imshow("image_right", image_right)
        # cv2.waitKey(0)

        # If the saved homography matrix is None, then we need to apply keypoint matching to construct it
        if self.saved_homo_matrix is None:
            # Detect keypoints and extract
            (keypoints_right, features_right) = self.detect_and_extract(right_gray)
            (keypoints_left, features_left) = self.detect_and_extract(left_gray)

            # Match features between the two images
            matched_keypoints = self.match_keypoints(keypoints_right, keypoints_left, features_right, features_left, ratio, reproj_thresh)

            # If the match is None, then there aren't enough matched keypoints to create a panorama
            if matched_keypoints is None:
                return None

            # Save the homography matrix
            self.saved_homo_matrix = matched_keypoints[1]
            # print(self.saved_homo_matrix)



        # Apply a perspective transform to stitch the images together using the saved homography matrix
        output_shape = (image_right.shape[1] + image_left.shape[1], image_right.shape[0])
        # print("##################", output_shape)

        # result = np.zeros((output_shape[0], output_shape[1]))
        # print("#################", image_right.shape, image_right.shape[0],image_right.shape[1])
        right_transform = cv2.warpPerspective(image_right, self.saved_homo_matrix, (image_right.shape[1],image_right.shape[0]))
        # cv2.namedWindow("right_transform", 0)
        # cv2.imshow("right_transform", right_transform)
        # cv2.waitKey(0)
        left_top_x, left_bottom_x = self.CalcCorners(self.saved_homo_matrix, right_transform)
        # left_top_x, left_top_y, left_bottom_x, left_bottom_y, right_top_x, right_top_y, right_bottom_x, right_bottom_y = self.CalcCorners(self.saved_homo_matrix, right_transform)

        result = cv2.warpPerspective(image_right, self.saved_homo_matrix, output_shape)
        # cv2.namedWindow("result", 0)
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        result[0:image_left.shape[0], 0:image_left.shape[1]] = image_left

        result_final = self.OptimizeSeam(image_left, right_transform, result, left_top_x, left_bottom_x)
        # Return the stitched image#.astype("uint8")
        return result_final

    @staticmethod
    def OptimizeSeam(image_left, trans, result, left_top_x, left_bottom_x):
        #image_left.shape[1] = cols
        #image_left.shape[0] = rows
        start = min(left_top_x, left_bottom_x)
        # print("@@@@@@@@@@@@@", start,image_left.shape[1])
        processWidth = image_left.shape[1] - start
        rows = result.shape[0]
        cols = image_left.shape[1]
        # print("@@@@@@@@@@@@@", rows, cols)
        # alpha = 1

        for row in range(rows):
            for col in range(start, cols):
                # print(trans[row][col] + 2,trans[row,col,:] == 0)
                if (trans[row][col] == 0).all():
                    alpha = 1.0
                    # print("&&&&&&&&&&&&&&&&&&&&&&")
                    # print("alpha#########", alpha)
                else:
                    alpha = float((processWidth - (col - start))) / float(processWidth)
                    # alpha = (processWidth - (col - start)) / processWidth
                    # print("alpha#########", alpha, processWidth, (processWidth - (col - start)))
                # print("&&&&&&&&&&&", (image_left[row][col] * alpha + trans[row][col] * (1 - alpha)) + 10, type(image_left[row][col] * alpha + trans[row][col] * (1 - alpha)))
                result[row][col] = image_left[row][col] * alpha + trans[row][col] * (1 - alpha)
                # print("&&&&&&&&&&&", result[row][col], image_left[row][col] * alpha, trans[row][col] * (1 - alpha))
        return result

    @staticmethod
    def CalcCorners(homo_matrix, image):
        rows, cols, c = image.shape

        v2 = np.asarray([0.0, 0.0, 1.0])
        V2 = np.mat(v2)
        # print(V2.T)

        V1 = np.mat(homo_matrix) * V2.T
        v1 = np.array(V1)
        left_top_x = v1[0] / v1[2]
        # left_top_y = v1[1] / v1[2]

        v2[0] = 0
        v2[1] = rows
        v2[2] = 1
        V1 = np.mat(homo_matrix) * V2.T
        v1 = np.array(V1)
        left_bottom_x = v1[0] / v1[2]
        # left_bottom_y = v1[1] / v1[2]

        # v2[0] = cols
        # v2[1] = 0
        # v2[2] = 1
        # V1 = np.mat(homo_matrix) * V2.T
        # v1 = np.array(V1)
        # right_top_x = v1[0] / v1[2]
        # right_top_y = v1[1] / v1[2]
        #
        # v2[0] = cols
        # v2[1] = rows
        # v2[2] = 1
        # V1 = np.mat(homo_matrix) * V2.T
        # v1 = np.array(V1)
        # right_bottom_x = v1[0] / v1[2]
        # right_bottom_y = v1[1] / v1[2]
        return int(left_top_x), int(left_bottom_x)
        # return int(left_top_x), int(left_top_y), int(left_bottom_x), int(left_bottom_y), int(right_top_x), int(right_top_y), int(right_bottom_x), int(right_bottom_y)




        

    @staticmethod
    def detect_and_extract(image):
        # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
        descriptor = cv2.xfeatures2d.SIFT_create() #opencv>3.0
        # descriptor = cv2.SIFT() #opencv2.*
        (keypoints, features) = descriptor.detectAndCompute(image, None)

        # Convert the keypoints from KeyPoint objects to numpy arrays
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        # Return a tuple of keypoints and features
        return (keypoints, features)

    @staticmethod
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
            # print("homography_matrix: ", homography_matrix)

            # Return the matches, homography matrix and status of each matched point
            return (matches, homography_matrix, status)

        # No homography could be computed
        return None

    @staticmethod
    def draw_matches(image_right, image_left, keypoints_a, keypoints_b, matches, status):
        # Initialize the output visualization image
        (height_a, width_a) = image_right.shape[:2]
        (height_b, width_b) = image_left.shape[:2]
        visualisation = np.zeros((max(height_a, height_b), width_a + width_b, 3), dtype="uint8")
        visualisation[0:height_a, 0:width_a] = image_right
        visualisation[0:height_b, width_a:] = image_left

        for ((train_index, query_index), s) in zip(matches, status):
            # Only process the match if the keypoint was successfully matched
            if s == 1:
                # Draw the match
                point_a = (int(keypoints_a[query_index][0]), int(keypoints_a[query_index][1]))
                point_b = (int(keypoints_b[train_index][0]) + width_a, int(keypoints_b[train_index][1]))
                cv2.line(visualisation, point_a, point_b, (0, 255, 0), 1)

        # return the visualization
        return visualisation

    def run(self):
        # Set up video capture
        left_video = cv2.VideoCapture(self.left_video_in_path)
        right_video = cv2.VideoCapture(self.right_video_in_path)
        n_frames_left = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames_right = int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))

        print('[INFO]: {} and {} loaded'.format(self.left_video_in_path.split('/')[-1],
                                                self.right_video_in_path.split('/')[-1]))
        print('[INFO]: Video stitching starting....')

        # Get information about the videos
        n_frames = min(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT)),
                       int(right_video.get(cv2.CAP_PROP_FRAME_COUNT)))

        fps = int(left_video.get(cv2.CAP_PROP_FPS))
        # print("fps: ", fps)
        frames = []
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        # out = cv2.VideoWriter(self.video_out_path, -1, 30.0, (800, 705))
        out = cv2.VideoWriter(self.video_out_path, fourcc, 30.0, (800, 705))
        for _ in tqdm.tqdm(np.arange(n_frames)):
            # Grab the frames from their respective video streams
            ok, left = left_video.read()
            _, right = right_video.read()

            # cv2.imwrite('/home/boyce/Desktop/VideoStitch/VideoStitcher-master/videos/A_left.png', left)
            # cv2.imwrite('/home/boyce/Desktop/VideoStitch/VideoStitcher-master/videos/A_right.png', right)
            # cv2.waitKey(0)
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            if ok:
                # Stitch the frames together to form the panorama
                stitched_frame = self.stitch([left_gray, right_gray, left, right])
                # cv2.namedWindow("stitched_frame", 0)
                # cv2.imshow("stitched_frame", stitched_frame)
                # cv2.waitKey(0)

                # No homography could not be computed
                if stitched_frame is None:
                    print("[INFO]: Homography could not be computed!")
                    break

                # Add frame to video
                stitched_frame = imutils.resize(stitched_frame, width=self.video_out_width)
                # size = (int(stitched_frame.shape[0]),int(stitched_frame.shape[1]),int(stitched_frame.shape[2]))
                #frames.append(stitched_frame)
                # cv2.namedWindow("stitched_frame", 0)
                # cv2.imshow("stitched_frame", stitched_frame)
                # cv2.waitKey(0)
                out.write(stitched_frame)
                if self.display:
                    # Show the output images
                    cv2.imshow("Result", stitched_frame)

                # If the 'q' key was pressed, break from the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        out.release()
        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

        # Save video
        # print('[INFO]: Saving {} in {}'.format(self.video_out_path.split('/')[-1],
        #                                        os.path.dirname(self.video_out_path)))
        # clip = ImageSequenceClip(frames, fps=fps)
        # clip.write_videofile(self.video_out_path, codec='mpeg4', audio=False, progress_bar=True, verbose=False)
        print('[INFO]: {} saved'.format(self.video_out_path.split('/')[-1]))

    # def run(self):
    #     left_image = cv2.imread('./videos/A_left.png')
    #     right_image = cv2.imread('./videos/A_right.png')
    #     # cols, rows, _ = left_image.shape
    #
    #     left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    #     # cv2.namedWindow("left_gray",0)
    #     # cv2.imshow("left_gray", left_gray)
    #     # cv2.waitKey(0)
    #     right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    #     # cv2.namedWindow("right_gray", 0)
    #     # cv2.imshow("right_gray", right_gray)
    #     # cv2.waitKey(0)
    #
    #     stitched_frame = self.stitch([left_gray, right_gray, left_image, right_image])
    #     cv2.namedWindow("stitched_frame", 0)
    #     cv2.imshow("stitched_frame", stitched_frame)
    #     cv2.waitKey(0)
    #     cv2.imwrite("./output/result123.png", stitched_frame)


# Example call to 'VideoStitcher'
stitcher = VideoStitcher(left_video_in_path='./videos/webwxgetvideo_left.mp4',
                         right_video_in_path='./videos/webwxgetvideo_right.mp4',
                         video_out_path='./output/stitched_video2.avi')
start  = time.time()

stitcher.run()

end = time.time()
total_time = end - start
print("total_time: ", total_time)
