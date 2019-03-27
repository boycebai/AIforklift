#!/usr/bin/python
# -*- coding:utf-8 -*-

'''

Team ID: Intelligent Warehouse

run_distance_cm :run in the distance
turn_angle :turn
up_down_cm :up and down
run_speed :always run
stop_run :stop

'''


import serial
# import pymysql
# import threading
import time
cmd_serial = serial.Serial('COM5',9600,timeout=5)
# eg [0X5A,  0X01,0X00,  0X08,0X00,  0X01,      0X05,  0X01,   0XA1,0XA2,0XA3,0XA4,  0X45,    0XA5]
#    5A包头, 版本号       协议号       数据流向   长度    序列号  数据                   校验和    包尾
head = bytearray([0X5A])
version = bytearray([0X01,0X00])
protocol_version = bytearray([0X00,0X00])
data_to_client = bytearray([0X01])
data_to_server = bytearray([0X02])
tail = bytearray([0XA5])
number = 0

# flag_dirct = ["HEAD":0,""]
def print_hex(bytes):
    l = [hex(int(i)) for i in bytes]
    print(" ".join(l))


def check_sum(data_bytes):
    pass
    data_sum = 0
    for data in data_bytes:
        data_sum += data
    data_sum = data_sum & 0xFF
    return data_sum

def rec_one_frame():
    while(True):
        ret_bytes = bytearray(cmd_serial.read(1))
        print_hex(ret_bytes)
        # print(type(ret_bytes))
        if head == ret_bytes:
            ret_bytes = ret_bytes + bytearray(cmd_serial.read(6))
            print_hex(ret_bytes)
            data_len = ret_bytes[6]
            print(data_len)
            ret_bytes = ret_bytes + bytearray(cmd_serial.read(data_len + 2))
            # break
            data_sum = check_sum(ret_bytes[1:6+data_len+1])
            # print("######")
            print(data_sum)
            print 'data_sum=0x%x'%(data_sum)
            print_hex(ret_bytes)
            print_hex(ret_bytes[1:6+data_len])
            if(data_sum == ret_bytes[-2]):
                # print_hex(ret_bytes)
                print("check sum ok")
                break
            else:
                print("check sum error")
                continue
    return ret_bytes



# direction 0x00 前进，0x01后退
def run_distance_cm(direction, val):
    global number
    # sub_cmd = 0
    protocol_version[0] = 2
    if direction !=0 and direction !=1 :
        print("run_cm direction error!!!")
        return False
    data_len = 3
    cmd_bytes = version + protocol_version + data_to_client + bytearray([data_len]) + bytearray([number]) + bytearray([direction]) + bytearray([val])
    data_sum = check_sum(cmd_bytes)
    cmd_bytes = head + cmd_bytes + bytearray([data_sum]) + tail
    print(cmd_bytes)
    cmd_serial.write(cmd_bytes)
    # ret_bytes = rec_one_frame()


    number = (number + 1) & 0xFF
    return True

# direction 0x00 左转，0x01右转
def turn_angle(direction, val):
    global number
    # sub_cmd = 1
    protocol_version[0] = 1
    if direction !=0 and direction !=1 :
        print("run_cm direction error!!!")
        return False
    if val > 90:
        val = 90
    data_len = 3
    cmd_bytes = version + protocol_version + data_to_client + bytearray([data_len]) + bytearray([number]) + bytearray([direction]) + bytearray([val])
    data_sum = check_sum(cmd_bytes)
    cmd_bytes = head + cmd_bytes + bytearray([data_sum]) + tail
    print_hex(cmd_bytes)
    cmd_serial.write(cmd_bytes)
    # ret_bytes = rec_one_frame()


    number = (number + 1) & 0xFF
    return True

# direction 0x00 上升，0x01下降
def up_down_cm(direction, val):
    global number
    # sub_cmd = 2
    protocol_version[0] = 3
    if direction !=0 and direction !=1 :
        print("run_cm direction error!!!")
        return False
    data_len = 3
    cmd_bytes = version + protocol_version + data_to_client + bytearray([data_len]) + bytearray([number]) + bytearray([direction]) + bytearray([val])
    data_sum = check_sum(cmd_bytes)
    cmd_bytes = head + cmd_bytes + bytearray([data_sum]) + tail
    print_hex(cmd_bytes)
    cmd_serial.write(cmd_bytes)
    # ret_bytes = rec_one_frame()


    number = (number + 1) & 0xFF
    return True

# direction 0x00 前进，0x01后退,val cm/s
def run_speed(direction, val):
    global number
    # sub_cmd = 3
    protocol_version[0] = 4
    if direction !=0 and direction !=1 :
        print("run_cm direction error!!!")
        return False
    data_len = 3
    cmd_bytes = version + protocol_version + data_to_client + bytearray([data_len]) + bytearray([number]) + bytearray([direction]) + bytearray([val])
    data_sum = check_sum(cmd_bytes)
    cmd_bytes = head + cmd_bytes + bytearray([data_sum]) + tail
    print(cmd_bytes)
    cmd_serial.write(cmd_bytes)
    # ret_bytes = rec_one_frame()
    number = (number + 1) & 0xFF
    return True

def stop_run():
    global number
    # sub_cmd = 4
    protocol_version[0] = 5
    data_len = 1
    cmd_bytes = version + protocol_version + data_to_client + bytearray([data_len]) + bytearray([number])
    # print(type(cmd_bytes))
    # print(type(cmd_bytes.decode()))
    data_sum = check_sum(cmd_bytes)
    cmd_bytes = head + cmd_bytes + bytearray([data_sum]) + tail
    print(cmd_bytes)
    cmd_serial.write(cmd_bytes)
    # ret_bytes = rec_one_frame()
    number = (number + 1) & 0xFF
    return True

def stop_turn():
    global number
    # sub_cmd = 5
    protocol_version[0] = 6
    data_len = 1
    cmd_bytes = version + protocol_version + data_to_client + bytearray([data_len]) + bytearray([number])
    data_sum = check_sum(cmd_bytes)
    cmd_bytes = head + cmd_bytes + bytearray([data_sum]) + tail
    print(cmd_bytes)
    cmd_serial.write(cmd_bytes)
    # ret_bytes = rec_one_frame()
    number = (number + 1) & 0xFF
    return True

def stop_updown():
    global number
    # sub_cmd = 6
    protocol_version[0] = 7
    data_len = 1
    cmd_bytes = version + protocol_version + data_to_client + bytearray([data_len]) + bytearray([number])
    data_sum = check_sum(bytearray(cmd_bytes, encoding='utf8'))
    cmd_bytes = head + cmd_bytes + bytearray([data_sum]) + tail
    print(cmd_bytes)
    cmd_serial.write(cmd_bytes)
    # ret_bytes = rec_one_frame()
    number = (number + 1) & 0xFF
    return True

# def send():
#     while True:
#         time.sleep(3)
#         myinput= bytes([0X5A,0X01,0X01,0X08,0X00,0X01,0X05,0X01,0XA1,0XA2,0XA3,0XA4,0X45,0XA5])
#         x.write(myinput)
# def jieshou():#接收函数
#     while True:
#        while x.inWaiting()>0:
#            myout=x.read(7)#读取串口传过来的字节流，这里我根据文档只接收7个字节的数据



if __name__== '__main__':
    # run_cm(0, 255)
    # run_distance_cm(0, 100)

    # up_down_cm(1,10)

    # for i in range(1,4):
    #     print "######"
    #     up_down_cm(0,5)
    #     time.sleep(4)
    #     up_down_cm(1,5)
    # time.sleep(4)
    stop_run()
    time.sleep(2)
    # run_speed(1,5)
    # time.sleep(3)
    run_speed(0,5)
    time.sleep(4)
    # run_speed(1,5)
    # time.sleep(2)
    stop_run()
    # time.sleep(2)
    # up_down_cm(0,20)
    # time.sleep(4)
    # up_down_cm(1,20)
    # time.sleep(4)

    # stop_turn()
    # stop_updown()
    # run_cm(1, 100)



    # turn_angle(1,30)
    # rec_one_frame()
    # stop_run()
    # ret_bytes = cmd_serial.read(2)
    # ret_bytes = bytearray(ret_bytes)

    # print(type(ret_bytes))
    # print(ret_bytes)


    # data_sum = check_sum(bytes([0X01,0X01,  0X08,0X00,  0X01,      0X05,  0X01,   0X01,0X02,0X03,0X04]))
    # print(data_sum)
    #  t1 = threading.Thread(target=jieshou,name="jieshou")
    #  t2= threading.Thread(target=send, name="send")
    #  t2.start()#开启线程1
    #  t1.start()#开启线程2
