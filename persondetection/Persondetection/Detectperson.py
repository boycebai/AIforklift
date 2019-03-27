#!usr/bin/python
#  -*- coding: utf-8 -*-

#  import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from openni import openni2

# openni2.initialize()  # can also accept the path of the OpenNI redistribution
#
# dev = openni2.Device.open_any()
# # print(dev.get_device_info())
#
# color_stream = dev.create_color_stream()
# color_stream.start()


# depth_stream = dev.create_depth_stream()
# depth_stream.start()
# dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
# hog = cv2.FHOGDescriptor()
#　使用opencv默认的SVM分类器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while(1):
    # get a frame from webcarmos
    ret, frame = cap.read()

    # # get a frame from depthcarmos
    # cframe = color_stream.read_frame()
    # frame = np.array(cframe.get_buffer_as_triplet()).reshape(
    #     [color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3])


    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    #　非极大抑制 消除多余的框 找到最佳人体
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.85)
    print("The number of person: ", len(pick))
    # 画出边框
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

