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

def stitch(image, col, raw):
    saved_homo_matrix = np.array([[9.22359134e-01, -1.10679987e-01, 2.64667989e+02],[6.97748005e-02, 9.54478561e-01, 4.47715322e+01],[-1.15977645e-04, -3.17301764e-05, 1.00000000e+00]])
    # print(saved_homo_matrix)
    # (image) = image
    # print("image: ", image.shape, type(image))
    # cv2.namedWindow("image", 0)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    output_shape = (raw + raw + raw, col + col)

    # output_shape = (col + col, raw + raw + raw, 3)
    result = cv2.warpPerspective(image, saved_homo_matrix, output_shape)
    # result = np.zeros(output_shape)
    # print("result_shape: ", result.shape, type(result))

    result[0:col, 0:raw, :] = image[: ,:, :]
    result[col:col + col, 0:raw, :] = image[:, :, :]
    result[0:col, raw:raw+ raw, :] = image[:, :, :]
    result[0:col, raw + raw:raw+ raw + raw, :] = image[:, :, :]
    result[col:col + col, raw:raw + raw, :] = image[:, :, :]
    result[col:col + col, raw + raw:raw + raw + raw, :] = image[:, :, :]

    # cv2.namedWindow("result", 0)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)

    return result

def seg_image(fusion_image, col, raw):

    video_up_left = fusion_image[0:col, 0:raw, :]
    video_up_middle = fusion_image[0:col, raw:raw+ raw, :]
    video_up_right = fusion_image[0:col, raw + raw:raw+ raw + raw, :]
    video_down_left = fusion_image[col:col + col, 0:raw, :]
    video_down_middle =fusion_image[col:col + col, raw:raw + raw, :]
    video_down_right = fusion_image[col:col + col, raw + raw:raw + raw + raw, :]
    return video_up_left, video_up_middle, video_up_right, video_down_left, video_down_middle, video_down_right
    

left_video_in_path='./videos/webwxgetvideo_left.avi'
fusion_video_path='./output/fusion_video.avi'
video_up_left_path = './output/video_up_left.avi'
video_up_middle_path = './output/video_up_middle.avi'
video_up_right_path = './output/video_up_right.avi'
video_down_left_path = './output/video_down_left.avi'
video_down_middle_path = './output/video_down_middle.avi'
video_down_right_path = './output/video_down_right.avi'

video_out_width=800

video = cv2.VideoCapture(left_video_in_path)

n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
print("^^^^^^^^^^", n_frames, fps)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(fusion_video_path, fourcc, 30.0, (800, 705))
# out1 = cv2.VideoWriter(video_up_left_path, fourcc, 30.0, (800, 705))
# out2 = cv2.VideoWriter(video_up_middle_path, fourcc, 30.0, (800, 705))
# out3 = cv2.VideoWriter(video_up_right_path, fourcc, 30.0, (800, 705))
# out4 = cv2.VideoWriter(video_down_left_path, fourcc, 30.0, (800, 705))
# out5 = cv2.VideoWriter(video_down_middle_path, fourcc, 30.0, (800, 705))
# out6 = cv2.VideoWriter(video_down_right_path, fourcc, 30.0, (800, 705))

for _ in tqdm.tqdm(np.arange(n_frames)):
    # print("****************************")
    ok, ori_image = video.read()
    cv2.namedWindow("ori_image", 0)
    cv2.imshow("ori_image", ori_image)
    # cv2.waitKey(0)
    # print("left_shape: ", left.shape)
    col = ori_image.shape[0]
    raw = ori_image.shape[1]
    if ok:
        stitched_frame = stitch(ori_image, col, raw)
        # cv2.namedWindow("stitched_frame", 0)
        # cv2.imshow("stitched_frame", stitched_frame)
        # cv2.waitKey(0)

        if stitched_frame is None:
            print("[INFO]:no video!")
            break
        # stitched_frame = imutils.resize(stitched_frame, width=video_out_width)
        cv2.namedWindow("stitched_frame", 0)
        cv2.imshow("stitched_frame", stitched_frame)
        # cv2.waitKey(0)
        # out.write(stitched_frame)

        video_up_left, video_up_middle, video_up_right, video_down_left, video_down_middle, video_down_right = seg_image(stitched_frame, col, raw)

        cv2.namedWindow("video_up_left", 0)
        cv2.imshow("video_up_left", video_up_left)
        # cv2.waitKey(0)
        cv2.namedWindow("video_up_middle", 0)
        cv2.imshow("video_up_middle", video_up_middle)
        # cv2.waitKey(0)
        cv2.namedWindow("video_up_right", 0)
        cv2.imshow("video_up_right", video_up_right)
        # cv2.waitKey(0)
        cv2.namedWindow("video_down_left", 0)
        cv2.imshow("video_down_left", video_down_left)
        # cv2.waitKey(0)
        cv2.namedWindow("video_down_middle", 0)
        cv2.imshow("video_down_middle", video_down_middle)
        # cv2.waitKey(0)
        cv2.namedWindow("video_down_right", 0)
        cv2.imshow("video_down_right", video_down_right)
        cv2.waitKey(0)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

out.release()
cv2.destroyAllWindows()
print('[INFO]: Video process finished')