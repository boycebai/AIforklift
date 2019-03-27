#!/usr/bin/python
# -*- coding:utf-8 -*-

'''

Team ID: Intelligent Warehouse
Author List: Chunshan Bai

1)target_image
2)image_roi
3)persM
4)contourArea
5)resvalue

'''

import cv2
import numpy as np
import time
import os
import math
from openni import openni2
from PIL import Image, ImageDraw, ImageFont
# from skimage import data,filters

dict = {'0': 'Keep on',
        '1': 'Turn',
        '2': 'Go',
        '3': 'Stop',
        '4': 'Go',
        '5': 'Stop'}


def get_distance(x1, y1, x2, y2):
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance


def get_histGBR(path):
    img = cv2.imread(path)
    pixal = img.shape[0] * img.shape[1]
    total = np.array([0])
    for i in range(3):
        histSingle = cv2.calcHist([img], [i], None, [256], [0, 256])
        total = np.vstack((total, histSingle))
    return (total, pixal)


def get_templet(path):
    image_templet = cv2.imread(path)
    templet_gray = cv2.cvtColor(image_templet, cv2.COLOR_BGR2GRAY)

    # 图像自适应阈值二值化
    (_, templet_thresh) = cv2.threshold(templet_gray, 130, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)
    templet_resize = cv2.resize(templet_thresh, (150, 300), interpolation=cv2.INTER_CUBIC)
    # cv2.namedWindow("templet_resize",0)
    # cv2.imshow("templet_resize", templet_resize)
    # cv2.waitKey(0)
    return templet_resize


def hist_similar(lhist, rhist, lpixal, rpixal):
    rscale = rpixal / lpixal
    rhist = rhist / rscale
    assert len(lhist) == len(rhist)
    likely = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lhist, rhist)) / len(lhist)
    if likely == 1.0:
        return [1.0]
    return likely


def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1 - npvec2) ** 2).sum())

def get_perspective_image(image,image_roi):
    # cv2.namedWindow("image1", 0)
    # cv2.imshow("image1", image)
    # cv2.waitKey(0)
    binary, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image_roi, contours, -1, (0, 255, 0), 3)

    # cv2.namedWindow("image_roi2", 0)
    # cv2.imshow("image_roi2", image_roi)
    # cv2.waitKey(0)

    biggest = 0
    max_area = 0
    min_size = image.size / 4
    index1 = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 10000:
            peri = cv2.arcLength(i, True)

        if area > max_area:
            biggest = index1
            max_area = area
        index1 = index1 + 1

    # area_max = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(area_max)

    approx = cv2.approxPolyDP(contours[biggest], 0.05 * peri, True)
    # print(approx.shape)
    # drawing the biggest polyline
    qwer = cv2.polylines(image_roi, [approx], True, (0, 255, 0), 3)

    cv2.namedWindow("qwer", 0)
    cv2.imshow("qwer", qwer)
    cv2.waitKey(0)

    x1 = approx[0][0][0]
    y1 = approx[0][0][1]
    x2 = approx[1][0][0]
    y2 = approx[1][0][1]
    x3 = approx[3][0][0]
    y3 = approx[3][0][1]
    x4 = approx[2][0][0]
    y4 = approx[2][0][1]

    # x1 = x - 5
    # y1 = y - 5
    # x2 = x - 5
    # y2 = y + h + 5
    # x3 = x + w - 5
    # y3 = y + 5
    # x4 = x + w + 5
    # y4 = y + h + 5
    print(x1, y1, x2, y2, x3, y3, x4, y4)
    raw_points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    min_dist = 10000
    #判断四个点的位置
    for i in raw_points:
        x, y = i
        dist = get_distance(x, y, 0, 0)
        print("x1,y1: ", dist)
        if dist < min_dist:
            min_dist = dist
            X1, Y1 = x, y
    min_dist = 10000
    for i in raw_points:
        x, y = i
        dist = get_distance(x, y, 0, 200)
        print("x2,y2: ", dist)
        if dist < min_dist:
            min_dist = dist
            X2, Y2 = x, y
    min_dist = 10000
    for i in raw_points:
        x, y = i
        dist = get_distance(x, y, 210, 0)
        print("x3,y3: ", dist)
        if dist < min_dist:
            min_dist = dist
            X3, Y3 = x, y
    min_dist = 10000
    for i in raw_points:
        x, y = i
        dist = get_distance(x, y, 210, 200)
        print("x4,y4: ", dist)
        if dist < min_dist:
            min_dist = dist
            X4, Y4 = x, y

    pts1 = np.float32([[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]])
    print(X1, Y1, X2, Y2, X3, Y3, X4, Y4)
    pts2 = np.float32([[0, 0], [0, 200], [210, 0], [210, 200]])
    persM = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, persM, (210, 200))
    corner_points = [[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]]

    return (dst, corner_points)

def image_light(image):
    # 求平均亮度
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_lightvalue = cv2.mean(image_gray)
    return image_lightvalue[0]

def compute_angle(a1,b1,a2,b2):
    if a2 == a1:
        return 0
    else:
        k1 = (b2 - b1) / (float(a2 - a1))
        x = np.array([1, k1])
        y = np.array([1, 0])
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        depart_angle = 90 - (np.arccos(x.dot(y) / (float(Lx * Ly))) * 180 / np.pi)
        # depart_angle = 90 - int((np.arccos(x.dot(y) / (float(Lx * Ly))) * 180 / np.pi) + 0.5)
        return depart_angle

def image_recog(image):
    # 图像压缩
    image_resize = cv2.resize(image, (210, 200), interpolation=cv2.INTER_CUBIC)
    # print("$$$$$$$$$$$$$$$$2",image_resize.shape)
    # cv2.namedWindow("image_resize",0)
    # cv2.imshow("image_resize", image_resize)
    # cv2.waitKey(0)

    # 选取感兴趣区域
    heigh, width, dim = image.shape

    a_1 = 0.4
    a_2 = 0.95
    a_3 = 0.1
    a_4 = 0.9
    image_roi = image_resize[int(a_1 * heigh):int(a_2 * heigh), int(a_3 * width):int(a_4 * width), :]
    # cv2.namedWindow("image_roi",0)
    # cv2.imshow("image_roi", image_roi)
    # cv2.waitKey(0)

    # 图像灰度化
    image_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow("image_gray",0)
    # cv2.imshow("image_gray", image_gray)
    # cv2.waitKey(0)

    # image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.namedWindow("image_HSV",0)
    # cv2.imshow("image_HSV", image_HSV)
    # cv2.waitKey(0)
    #
    # H = image_HSV[:,:,0]
    # cv2.namedWindow("H", 0)
    # cv2.imshow("H", H)
    # cv2.waitKey(0)
    # V = image_HSV[:,:,2]
    # cv2.namedWindow("V",0)
    # cv2.imshow("V", V)
    # cv2.waitKey(0)
    # result = np.multiply(np.array(H),np.array(255-V))
    # result[result > 25.5] = 0
    # result = int(result*255.0)
    # cv2.namedWindow("result", 0)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)


    # 图像自适应阈值二值化
    # (_, thresh) = cv2.threshold(image_gray, 80, 255, cv2.THRESH_BINARY)
    ret2, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 7)
    # thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 5)
    cv2.namedWindow("thresh",0)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)


    # 形态学处理
    # 构造一个25*25的结构元素
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    diamond = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diamond[0, 0] = 0
    diamond[0, 2] = 0
    diamond[1, 1] = 0
    diamond[2, 0] = 0
    diamond[2, 2] = 0

    result2 = cv2.erode(thresh, diamond)
    opened = cv2.dilate(result2, cross)
    # cv2.namedWindow("opened", 0)
    # cv2.imshow("opened", opened)
    # cv2.waitKey(0)

    # 中值滤波
    # lbimg = cv2.medianBlur(opened, 3)
    # cv2.namedWindow("lbimg",0)
    # cv2.imshow("lbimg", lbimg)
    # cv2.waitKey(0)
    #
    # height, weight= lbimg.shape
    # print(height, weight)

    # 逆透视
    # dst, corner_points = get_perspective_image(thresh,image_roi)

    # a = 210
    # b = 200
    # pts1 = np.float32([[0 + 20, 0], [0, height - 1], [weight - 1 - 20, 0], [weight - 1, height - 1]])
    # pts2 = np.float32([[0, 0], [0, b], [a, 0], [a, b]])
    # persM = cv2.getPerspectiveTransform(pts1, pts2)
    # dst = cv2.warpPerspective(opened, persM, (a, b))
    # cv2.namedWindow("dst",0)
    # cv2.imshow("dst", dst)
    # cv2.waitKey(0)


    # 边缘轮廓提取
    image_canny = cv2.Canny(opened, 50, 150)
    # cv2.namedWindow("image_canny", 0)
    # cv2.imshow("image_canny", image_canny)
    # cv2.waitKey(0)
    height_canny, width_canny = image_canny.shape
    # print("**************",height_canny, width_canny)
    # 外接矩阵
    image2, cnts, hierarchy = cv2.findContours(image_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print("33333333333333", len(cnts))

    # try:
    #     area = max(cnts, key=cv2.contourArea)
    #     # print("################", cv2.contourArea(area))
    # except Exception as e:
    #     return str(0),-1,-1
    sign_cnt = []
    a = 10
    try:
        for i in range(len(cnts)):
            # cnt = cnts[i]
            # area = cv2.contourArea(cnts[i])

            x1, y1, w1, h1 = cv2.boundingRect(cnts[i])
            # print("############", cv2.contourArea(cnts[i]))
            # print("$$$$$$$$$$$$", (float(h1) / float(w1)))
            if (cv2.contourArea(cnts[i]) < 450*a and cv2.contourArea(cnts[i]) > 130) and (float(h1) / float(w1) > 0.9 and float(h1) / float(w1) < 3.9):
                # area = max(cnts, key=cv2.contourArea)
                # 取得完整的轮廓，不靠近边缘
                # print("############", cv2.contourArea(cnts[i]))
                # print("$$$$$$$$$$$$", (float(h1) / float(w1)))
                if x1 != 0 and x1 + w1 - 1 <= height_canny and y1 != 0 and y1 + h1 - 1 <= width_canny:
                    sign_cnt.append(cnts[i])
                    image_canny[y1:y1 + h1 - 1, x1] = 255
                    image_canny[y1, x1:x1 + w1 - 1] = 255
                    image_canny[y1:y1 + h1 - 1, x1 + w1 - 1] = 255
                    image_canny[y1 + h1 - 1, x1:x1 + w1 - 1] = 255
                    cv2.namedWindow("image_canny", 0)
                    cv2.imshow("image_canny", image_canny)
                    cv2.waitKey(0)
        # print(type(sign_cnt))
        if len(sign_cnt) == 0:
            return str(0), -1, -1, -1, -1, -1, -1
        area = max(sign_cnt, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(area)
        # 提取需要区域
        image_ide = opened[y:y + h, x:x + w]
        # image_ide_heigh, image_ide_width = image_ide.shape
        # print("###############", image_ide_heigh, image_ide_width)
        # cv2.namedWindow("image_ide", 0)
        # cv2.imshow("image_ide", image_ide)
        # cv2.waitKey(0)

        # print("*******************", np.median(np.where(opened[y, x:x + w] == 255.0)[0]))
        # print(np.where(opened[y + 15,x:x + w] == 255.0)[0],width,width/210.0)
        # print("*******************", np.median(np.where(opened[y + h - 1, x:x + w] == 255)[0]))
        # print("*******************", np.median(np.where(opened[y:y + h, x] == 255)[0]))
        # print("*******************", np.median(np.where(opened[y:y + h,x + w - 1] == 255)[0]))

        # pointx_top = int((np.median(np.where(opened[y,x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
        # pointy_top = int((y + a_1 * heigh) * (heigh / 200.0))
        # pointx_down = int((np.median(np.where(opened[y + h - 1,x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
        # pointy_down = int(((y + h - 1) + a_1 * heigh) * (heigh / 200.0))
        #
        # pointx_top_3 = int((np.median(np.where(opened[y + 15,x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
        # pointy_top_3 = int((y + a_1 * heigh) * (heigh / 200.0))
        # pointx_down_3 = int((np.median(np.where(opened[y + h - 1 - 15,x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
        # pointy_down_3 = int(((y + h - 1) + a_1 * heigh) * (heigh / 200.0))

        # img = cv2.circle(image, (pointx_top,pointy_top), 3, (255, 0, 0), -1)
        # img = cv2.circle(img, (pointx_down,pointy_down), 3, (255, 0, 0), -1)
        # cv2.namedWindow("img",0)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        # 归一化大小
        image_norm = cv2.resize(image_ide, (150, 300), interpolation=cv2.INTER_CUBIC)
        # cv2.namedWindow("image_norm",0)
        # cv2.imshow("image_norm", image_norm)
        # cv2.waitKey(0)

        # 图像匹配
        rootdir = "../matchlib"
        resultDict = {}
        for parent, dirnames, filenames in os.walk(rootdir):
            for filename in filenames:
                if (filename[-3:] == 'png'):
                    jpgPath = os.path.join(parent, filename)
                    templet = get_templet(jpgPath)
                    # cv2.namedWindow("templet", 0)
                    # cv2.imshow("templet", templet)
                    # cv2.waitKey(0)
                    # resultDict[jpgPath] = Euclidean(image_norm, templet)
                    # resultDict[jpgPath] = cv2.matchTemplate(image_norm, templet, cv2.TM_CCORR_NORMED)
                    resultDict[jpgPath] = cv2.matchTemplate(image_norm, templet, cv2.TM_CCOEFF_NORMED)

        sortedDict = sorted(resultDict.items(), key=lambda asd: asd[1], reverse=True)
        # print("$$$$$$$$$$$$$4", sortedDict)
        for i in range(4):
            print(sortedDict[i])

        # 图标结果校准
        area_sign = cv2.contourArea(area)
        hw_sign = (float(h) / float(w))
        num_sign = str(sortedDict[0][0][-5:-4])
        # print("the result of sign: ", num_sign,type(num_sign))
        # print("************************: ", sortedDict[0][1][0][0],type(sortedDict[0][1][0][0]))
        # print("%%%%%%%%%%%%%%", area_sign,type(area_sign))
        # print("@@@@@@@@@@@@@@", hw_sign,type(hw_sign))
        pointx = int(((x + (x + w)) / 2 + a_3 * width) * (width / 210.0))
        pointy = int(((y + (y + h)) / 2 + a_1 * heigh) * (heigh / 200.0))
        # print("%%%%%%%%%%%%%%", pointx,type(pointx))
        # print("@@@@@@@@@@@@@@", pointy,type(pointy))
        # img = cv2.circle(image, (pointx,pointy), 3, (255, 0, 0), -1)
        # cv2.namedWindow("img",0)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)


        if sortedDict[0][1][0][0] > 0.5:
            if num_sign == str(1) and (area_sign < 450*a and area_sign > 200) and (hw_sign < 1.5 and hw_sign > 1):
                print("Motor commands: " + dict[num_sign])
                pointx_top = int((np.median(np.where(opened[y, x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
                pointy_top = int((y + a_1 * heigh) * (heigh / 200.0))
                pointx_down = int((np.median(np.where(opened[y + h - 1, x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
                pointy_down = int(((y + h - 1) + a_1 * heigh) * (heigh / 200.0))
                return num_sign, pointx, pointy, pointx_top,pointy_top, pointx_down,pointy_down
            elif num_sign == str(2) and (area_sign < 250*a and area_sign > 120) and (hw_sign < 3.9 and hw_sign > 2.0):
                print("Motor commands: " + dict[num_sign])
                pointx_top = int((np.median(np.where(opened[y, x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
                pointy_top = int((y + a_1 * heigh) * (heigh / 200.0))
                pointx_down = int((np.median(np.where(opened[y + h - 1, x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
                pointy_down = int(((y + h - 1) + a_1 * heigh) * (heigh / 200.0))
                return num_sign, pointx, pointy, pointx_top,pointy_top, pointx_down,pointy_down
            elif num_sign == str(3) and (area_sign < 350*a and area_sign > 150) and (hw_sign < 1.9 and hw_sign > 0.9):
                print("Motor commands: " + dict[num_sign])
                pointx_top_3 = int((np.median(np.where(opened[y + 15, x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
                pointy_top_3 = int((y + a_1 * heigh) * (heigh / 200.0))
                pointx_down_3 = int((np.median(np.where(opened[y + h - 1 - 15, x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
                pointy_down_3 = int(((y + h - 1) + a_1 * heigh) * (heigh / 200.0))
                return num_sign, pointx, pointy, pointx_top_3,pointy_top_3, pointx_down_3,pointy_down_3
            elif num_sign == str(4) and (area_sign < 350*a and area_sign > 120) and (hw_sign < 1.2 and hw_sign > 0.8):
                print("Motor commands: " + dict[num_sign])
                pointx_top = int((np.median(np.where(opened[y, x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
                pointy_top = int((y + a_1 * heigh) * (heigh / 200.0))
                pointx_down = int((np.median(np.where(opened[y + h - 1, x:x + w] == 255.0)[0]) + x + a_3 * width) * (width / 210.0))
                pointy_down = int(((y + h - 1) + a_1 * heigh) * (heigh / 200.0))
                return num_sign, pointx, pointy, pointx_top,pointy_top, pointx_down,pointy_down
            else:
                return str(0), -1, -1, -1, -1, -1, -1
        else:

            print("NO TARGET")
            return str(0), -1, -1, -1, -1, -1, -1

    except Exception as e:
        print("NO TARGET")

        print(e)
        return str(0), -1, -1, -1, -1, -1, -1


# Main script
if __name__ == '__main__':
    # 计时
    start = time.time()

    from optparse import OptionParser

    parser = OptionParser()
    # parser.add_option("-i", "--input_file", dest="input_file",
    #                   help="Input video/image file")
    # parser.add_option("-o", "--output_file", dest="output_file",
    #                   help="Output (destination) video/image file")
    parser.add_option("-I", "--image_only",
                      action="store_true", dest="image_only", default=False,
                      help="Annotate image (defaults to annotating video)")
    options, args = parser.parse_args()
    # input_file = options.input_file
    # output_file = options.output_file
    image_only = options.image_only

    if image_only:
        # 读入图像
        image = cv2.imread("../test/test_B.png")
        # get the lightvalue of image
        # image_lightvalue = image_light(image)
        # if image_lightvalue > 30 and image_lightvalue < 100:
        heigh, width, dim = image.shape
        result,centre_pointx,centre_pointy, pointx_top,pointy_top, pointx_down,pointy_down = image_recog(image)
        # print("distance%%%%%%%%%%:", heigh, width, centre_pointx, centre_pointy)
        print("depart_distance: ", str(width//2 - centre_pointx))
        cv2.line(image, (pointx_top,pointy_top), (pointx_down,pointy_down), (255, 0, 0), thickness=3)
        cv2.namedWindow("image", 0)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        depart_angle = compute_angle(pointx_top,pointy_top, pointx_down,pointy_down)
        print("depart_angle: ", depart_angle)


        # # 读入视频
        # cap = cv2.VideoCapture('../test/output15.avi')
        # frame_count = 1
        # success = True
        # while (success):
        #     success, image = cap.read()
        #     print("###########################")
        #     # print("Read a new frame: ", success)
        #     if success == False:
        #         break
        #     # cv2.imwrite('/home/boyce/Desktop/BigData/canku/image/0828/image_car_1015/test/test_A.png', image)
        #
        #     # 按下q键退出循环
        #     key = cv2.waitKey(10)
        #     if int(key) == 113:
        #         break
        #
        #     params = []
        #     params.append(1)
        #
        #     frame_count = frame_count + 1
        #     print("frame_count: ", frame_count)
        #     #get the lightvalue of image
        #     image_lightvalue = image_light(image)
        #     if image_lightvalue > 30 and image_lightvalue < 100:
        #         result,centre_pointx, centre_pointy, pointx_top,pointy_top, pointx_down,pointy_down = image_recog(image)
        #         cv2.line(image, (pointx_top,pointy_top), (pointx_down,pointy_down), (255, 0, 0), thickness=3)
        #         cv2.imshow('color', image)
        #         depart_angle = compute_angle(pointx_top,pointy_top, pointx_down,pointy_down)
        #         print("depart_angle: ", depart_angle)
        #
        # cap.release()
        # cv2.destroyAllWindows()

    else:
        openni2.initialize()  # can also accept the path of the OpenNI redistribution

        dev = openni2.Device.open_any()
        print(dev.get_device_info())

        depth_stream = dev.create_depth_stream()
        color_stream = dev.create_color_stream()
        depth_stream.start()
        color_stream.start()
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640 // 2, 480 // 2))
        frame_count = 1
        success = True
        while (success):

            # 显示深度图
            frame = depth_stream.read_frame()
            # print(frame.shape)
            print("###########################")
            # frame_data = frame.get_buffer_as_uint16()
            # print(frame)
            # frame.get_buffer_as_triplet
            dframe_data = np.array(frame.get_buffer_as_triplet()).reshape(
                [depth_stream.get_video_mode().resolutionY, depth_stream.get_video_mode().resolutionX, 2])
            # print(dframe_data.shape)
            # print(dframe_data)

            dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
            dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
            dpt2 *= 255
            dpt = dpt1 + dpt2

            # print(dpt[dpt.shape[0] // 2][dpt.shape[1] // 2])

            # dpt = dpt // 20
            # dpt = np.asarray(dpt[:, :], dtype="uint8")
            # # print(dpt.shape)

            # cv2.imshow('dpt', dpt)

            # 显示RGB图像
            cframe = color_stream.read_frame()
            cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape(
                [color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3])
            # cv2.imwrite('/home/boyce/Desktop/BigData/canku/image/0828/image_car_1009/test/test11.png', cframe_data)
            # print(cframe_data.shape)
            # cframe_data = cv2.flip(cframe_data, 1)  # 水平翻转
            R = cframe_data[:, :, 0]
            G = cframe_data[:, :, 1]
            B = cframe_data[:, :, 2]

            heigh, width, dim = cframe_data.shape
            result, centre_pointx, centre_pointy, pointx_top, pointy_top, pointx_down, pointy_down = image_recog(
                cframe_data)

            # print("distance%%%%%%%%%%:", heigh, width, centre_pointx, centre_pointy)
            print("depart_distance: ", str(width // 2 - centre_pointx))

            cv2.line(cframe_data, (pointx_top, pointy_top), (pointx_down, pointy_down), (255, 0, 0), thickness=3)
            depart_angle = compute_angle(pointx_top, pointy_top, pointx_down, pointy_down)
            print("depart_angle: ", depart_angle)

            cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])

            # params = []
            # params.append(1)


            # print("result: ", result)
            # print("centre_pointx: ", centre_pointx)
            # print("centre_pointy: ", centre_pointy)

            frame_count = frame_count + 1
            print("frame_count: ", frame_count)

            distance_dpt = dpt[centre_pointy][centre_pointx]

            # 图像从OpenCV格式转换成PIL格式
            img_PIL = Image.fromarray(cv2.cvtColor(cframe_data, cv2.COLOR_BGR2RGB))
            # 字体 字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc
            font = ImageFont.truetype('NotoSansCJK-Black.ttc', 20)
            # font = ImageFont.truetype("C:\\WINDOWS\\Fonts\\simsun.ttc", 20)
            if centre_pointx != -1 and centre_pointy != -1:
                print("dpt_distance: ", distance_dpt)
                fillColor = (0, 255, 0) # 字体颜色
                str_word = '标志距离: ' + str(distance_dpt) + '\n' + "输出指示信息： " + dict[result] + '\n' + "偏离角度： " + str(depart_angle) + '\n' + "偏离距离: " + str(width // 2 - centre_pointx)# 输出内容
                # str_word = '输出内容: ' + str(distance_dpt)  # 输出内容
            else:
                distance_dpt = -1
                print("dpt_distance: ", distance_dpt)
                fillColor = (255, 0, 0)# 字体颜色
                str_word = '无标志' + '\n' + "输出指示信息： " + dict[result] + '\n' + "偏离角度： " + str(depart_angle) + '\n' + "偏离距离： " + str(0)# 输出内容
            position = (10, 10)  # 文字输出位置
            # 需要先把输出的中文字符转换成Unicode编码形式
            if not isinstance(str_word, unicode):
                str_word = str_word.decode('utf8')
            draw = ImageDraw.Draw(img_PIL)
            draw.text(position, str_word, font=font, fill=fillColor)
            # 转换回OpenCV格式
            img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

            cv2.imshow('color', img_OpenCV)

            # 按下q键退出循环
            key = cv2.waitKey(10)
            if int(key) == 113:
                break


    end = time.time()
    total_time = end - start
    print("total_time: " + str(total_time))
