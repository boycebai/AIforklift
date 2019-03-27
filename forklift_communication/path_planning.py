#!/usr/bin/python
# -*- coding:utf-8 -*-

'''

Team ID: Intelligent Warehouse
Author List: Chunshan Bai

输出指令为[0,1]，其中
第一位数表示如下：
#0 直行
#1 转弯
#2 停止
第二位数表示如下：
#0 指令不成立
#1 指令成立

'''

def image_message(i):
    return image_order[i][0],image_order[i][1]


if __name__== '__main__':
    P = [0, 0, 1, 0, 0, 2] #服务器给出的仓库行走指令
    image_order = [[0, 0], [0, 1], [0, 1], [1, 0], [1, 1], [1, 0], [0, 1], [0, 0],
                   [0, 1], [2, 0], [2, 1]]  #摄像头输出的指令

    # try:
    for i in range(len(image_order)):
        ret0, ret1 = image_message(i)
        print(image_order[i], ret0, ret1)

        if ret0 == 0 and ret1 == 0:  #指令为直行但不生效
            print("keep on")
        elif ret0 == 0 and ret1 == 1: #指令为直行且生效
            print("forklift can go")
            pass #go
        elif ret0 == 1 and ret1 == 0: #指令为转弯但不生效
            print("keep on")
        elif ret0 == 1 and ret1 == 1: #指令为转弯且生效
            print("forklift can turn")
            pass #turn
        elif ret0 == 2 and ret1 == 0: #指令为停止但不生效
            print("keep on")
        elif ret0 == 2 and ret1 == 1: #指令为停止且生效
            print("forklift can stop")
            continue #stop

        if ret1 == 1:
            P.pop(0)
        # print(len(P))
        if len(P) == 1 and ret0 == 2 and ret1 == 1:
            print("Plan planning finished!")
            break