# -*- coding=utf-8 -*-
import os
import numpy as np
import cv2
# import cv2
import copy
import datetime
from PIL import Image, ImageDraw, ImageFont
# import zbarlight
import zbar

scaling = 1
img_path='images'
img_result='results'
def reshape_image(image):
    '''归一化图片尺寸：短边400，长边不超过800，短边400，长边超过800以长边800为主'''
    width,height=image.shape[1],image.shape[0]
    min_len=width
    scale=width*1.0/400
    new_width=400
    new_height=int(height/scale)
    if new_height>800:
        new_height=800
        scale=height*1.0/800
        new_width=int(width/scale)
    out=cv2.resize(image,(new_width,new_height))
    return out
def detecte(image):
    '''提取所有轮廓'''
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,gray=cv2.threshold(gray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    # cv2.namedWindow("gray", 0)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)
    img,contours,hierachy=cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return image,contours,hierachy

def compute_1(contours,i,j):
    '''最外面的轮廓和子轮廓的比例'''
    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])
    if area2==0:
        return False
    ratio = area1 * 1.0 / area2
    # print("%%%%%%%%%%ratio1:",ratio)
    # print("%%%%%%%%%%ratio1_abs:", abs(ratio - 49.0 / 25))
    if abs(ratio - 49.0 / 25):
        # print("((((((((((((((((((((")
        return True
    return False
def compute_2(contours,i,j):
    '''子轮廓和子子轮廓的比例'''
    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])
    if area2==0:
        return False
    ratio = area1 * 1.0 / area2
    # print("%%%%%%%%%%ratio2:",ratio)
    # print("%%%%%%%%%%ratio2_abs:", abs(ratio - 25.0 / 9))
    if abs(ratio - 25.0 / 9):
        # print("$$$$$$$$$$$$$$$$$$")
        return True
    return False
def compute_center(contours,i):
    '''计算轮廓中心点'''
    M=cv2.moments(contours[i])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx,cy
def detect_contours(vec):
    '''判断这个轮廓和它的子轮廓以及子子轮廓的中心的间距是否足够小'''
    distance_1=np.sqrt((vec[0]-vec[2])**2+(vec[1]-vec[3])**2)
    distance_2=np.sqrt((vec[0]-vec[4])**2+(vec[1]-vec[5])**2)
    distance_3=np.sqrt((vec[2]-vec[4])**2+(vec[3]-vec[5])**2)
    if sum((distance_1,distance_2,distance_3))/3<3:
        return True
    return False
def juge_angle(rec):
    '''判断寻找是否有三个点可以围成等腰直角三角形'''
    if len(rec)<3:
        return -1,-1,-1
    for i in range(len(rec)):
        for j in range(i+1,len(rec)):
            for k in range(j+1,len(rec)):
                distance_1 = np.sqrt((rec[i][0] - rec[j][0]) ** 2 + (rec[i][1] - rec[j][1]) ** 2)
                distance_2 = np.sqrt((rec[i][0] - rec[k][0]) ** 2 + (rec[i][1] - rec[k][1]) ** 2)
                distance_3 = np.sqrt((rec[j][0] - rec[k][0]) ** 2 + (rec[j][1] - rec[k][1]) ** 2)
                # print("**************: ", rec[i][0],rec[i][1],rec[j][0],rec[j][1],rec[k][0],rec[k][1])
                # print("distance_1: ", distance_1)
                # print("distance_2: ", distance_2)
                # print("distance_3: ", distance_3)
                if abs(distance_1-distance_2)<25:
                    if abs(np.sqrt(np.square(distance_1)+np.square(distance_2))-distance_3)<5:
                        return i,j,k
                elif abs(distance_1-distance_3)<25:
                    if abs(np.sqrt(np.square(distance_1)+np.square(distance_3))-distance_2)<5:
                        return i,j,k
                elif abs(distance_2-distance_3)<25:
                    if abs(np.sqrt(np.square(distance_2)+np.square(distance_3))-distance_1)<5:
                        return i,j,k
    return -1,-1,-1
def find(image,image_name,contours,hierachy,root=0):
    '''找到符合要求的轮廓'''
    rec=[]
    for i in range(len(hierachy)):
        child = hierachy[i][2]
        # print("#########55555#",child)
        child_child=hierachy[child][2]
        # print("#########66666#", child_child)
        if child!=-1 and hierachy[child][2]!=-1:
            if compute_1(contours, i, child) and compute_2(contours,child,child_child):
                # print("&&&&&&&&&&&&&&22222222222222222")
                cx1,cy1=compute_center(contours,i)
                cx2,cy2=compute_center(contours,child)
                cx3,cy3=compute_center(contours,child_child)
                if detect_contours([cx1,cy1,cx2,cy2,cx3,cy3]):
                    # print("&&&&&&&&&&&&&&333333333333333")
                    rec.append([cx1,cy1,cx2,cy2,cx3,cy3,i,child,child_child])
    '''计算得到所有在比例上符合要求的轮廓中心点'''
    i,j,k=juge_angle(rec)
    # print("aaaaaaaaaaaaaaaaaa",i,j,k)
    if i==-1 or j== -1 or k==-1:
        print("This is no QRCode")
        cv2.imshow('img',cv2.resize(image,(int(image.shape[1]/scaling),int(image.shape[0]/scaling))))
        return
    ts = np.concatenate((contours[rec[i][6]], contours[rec[j][6]], contours[rec[k][6]]))
    rect = cv2.minAreaRect(ts)
    box = cv2.boxPoints(rect)
    # print("$$$$$$$$$$$$$", rect[0][1])
    # print("image.shape:",image.shape[0] - rect[0][1])
    box = np.int0(box)
    result=copy.deepcopy(image)
    cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
    cv2.drawContours(image,contours,rec[i][6],(255,0,0),2)
    cv2.drawContours(image,contours,rec[j][6],(255,0,0),2)
    cv2.drawContours(image,contours,rec[k][6],(255,0,0),2)
    # cv2.imshow('img',cv2.resize(image,(image.shape[1]//scaling,image.shape[0]//scaling)))
    # cv2.waitKey(0)
    # print("imshow")


    # cv2.namedWindow("img", 0)
    cv2.imshow('img',cv2.resize(result,(int(result.shape[1]/scaling),int(result.shape[0]/scaling))))
    cv2.waitKey(0)


    # path=os.path.join(img_result,image_name)
    # cv2.imwrite(path,result)

    min=np.min(box, axis=0)
    max=np.max(box, axis=0)
    kz = 10
    roi = image[min[1]-kz:max[1]+kz,min[0]-kz:max[0]+kz]

    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("roi", 0)
    cv2.imshow("roi", roi)
    cv2.waitKey(0)
    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')

    pil = Image.fromarray(roi).convert('L')

    width, heigh = pil.size

    print(width, heigh)
    # raw = pil.tostring()
    raw = pil.tobytes()

    zarimage = zbar.Image(width, heigh, 'Y800', raw)

    results = scanner.scan(zarimage)
    # print("results: ", results)
    # print("$$$$$$$$$$$$$$$$$$$$$$$$")
    if results:
        print('QR codes: %s' % results)
        # print('QR codes: %s' % results[0].decode())
        print("distance_left: ", image.shape[1] - rect[0][0])
        print("distance_right: ", rect[0][0])
        print("distance_up: ", rect[0][1])
        print("distance_down: ", image.shape[0] - rect[0][1])
        print("$"*20)
    else:
        print("二维码无法识别")

    # # create a reader
    # scanner = zbar.ImageScanner()
    # # configure the reader
    # scanner.parse_config('enable')
    # pil = Image.fromarray(roi).convert('L')
    # width, heigh = pil.size
    #
    # raw = pil.topytes()
    # print("**********************")
    # zarimage = zbar.Image(width, heigh, 'Y800', raw)
    #
    # scanner.scan(zarimage)
    # print("codes: ", zarimage)

    # # print("%%%%%%%%%%%%", roi.shape[1], roi.shape[0])
    # # # wrap image data
    # # zarimage = zbar.Image(roi.shape[1], roi.shape[0], 'Y800', pil)
    # # print("**********************")
    # # codes = scanner.scan(zarimage)
    # # print("**********************", codes)
    # print("**********************")
    # codes = ZbarDecoder(pil)
    # # codes = zbar.scan_codes(['qrcode'], pil)
    #
    # # codes = zbarlight.scan_codes(['qrcode'], pil)
    # print("codes: ",codes)


    # if codes:
    #     print('QR codes: %s' % codes[0].decode())
    #     print("distance_left: ", image.shape[1] - rect[0][0])
    #     print("distance_right: ", rect[0][0])
    #     print("distance_up: ", rect[0][1])
    #     print("distance_down: ", image.shape[0] - rect[0][1])
    # else:
    #     print("二维码无法识别")
    return


if __name__ == '__main__':

    camera = cv2.VideoCapture("./video/video4.mpeg")
    # camera = cv2.VideoCapture("http://admin:admin@10.238.40.243:8081/")
    # camera = cv2.VideoCapture(1)
    print(camera.isOpened())

    while (camera.isOpened()):
        (grabbed, image) = camera.read()
        if not grabbed:
            break
        # print("#######")
        starttime = datetime.datetime.now()
        image,contours,hierachy = detecte(image)
        # print(hierachy.shape)
        try:
            find(image,"file_name",contours,np.squeeze(hierachy))
            # print("&&&&&&&&&&&&&&&&&")
        except Exception as e:
            print("Exception: ", e)
            # cv2.imshow('img', image)
            # cv2.waitKey(0)


        endtime = datetime.datetime.now()
        # print((endtime - starttime).microseconds)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
        # cv2.imshow('frame', image)


    camera.release()
    cv2.destroyAllWindows()
