#!/usr/bin/python
# -*- coding:utf-8 -*-

from openni import openni2
import numpy as np
import cv2



openni2.initialize()     # can also accept the path of the OpenNI redistribution

dev = openni2.Device.open_any()
# dev = openni2.Device.open_file("/dev/video0")
print(dev.get_device_info())
depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()

depth_stream.start()
color_stream.start()
dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640//2,480//2))

while True:
    # 显示深度图
    frame = depth_stream.read_frame()
    # print(frame.shape)
    print("###########################")
    # frame_data = frame.get_buffer_as_uint16()
    # print(frame)
    # frame.get_buffer_as_triplet
    dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([depth_stream.get_video_mode().resolutionY, depth_stream.get_video_mode().resolutionX, 2])
    print(dframe_data.shape)
    # print(dframe_data)
    
    dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
    dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
    dpt2 *= 255
    dpt = dpt1  + dpt2
    print(dpt[dpt.shape[0]//2][dpt.shape[1]//2])

    dpt = dpt//20
    dpt = np.asarray(dpt[:,:],dtype="uint8")
    # print(dpt.shape)
    cv2.imshow('dpt', dpt)
    # # 按下q键退出循环
    # key = cv2.waitKey(10)
    # if int(key) == 113:
    #     break


    

    # 显示RGB图像
    cframe = color_stream.read_frame()
    cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3])
    # print(cframe_data.shape)
    R = cframe_data[:, :, 0]
    G = cframe_data[:, :, 1]
    B = cframe_data[:, :, 2]
    cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
    # print(cframe_data.shape)
    # out.write(cframe_data)
    cframe_data = cv2.cvtColor(cframe_data,cv2.COLOR_BGR2GRAY)
    # print(cframe_data.shape)
    # print(cframe_data.dtype)

    cv2.imshow('color', cframe_data)

    overlapping = cv2.addWeighted(cframe_data, 0.5, dpt, 0.5, 0)
    cv2.imshow('depth_color', overlapping)


    # 按下q键退出循环
    key = cv2.waitKey(10)
    if int(key) == 113:
        break

# 人走带门，关闭设备
depth_stream.stop()
color_stream.stop()
dev.close()
out.release()
cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,0)

#         # write the flipped frame
#         out.write(frame)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()
