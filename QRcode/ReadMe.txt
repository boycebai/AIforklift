文件所需配置环境如下：

- python    `2.7.12`

- cv2       `3.4.2`

- numpy     `1.15.4`

- Pillow    `3.1.2`
- zbar

zbar安装教程：
pip install zbar
（注意事项：必须用python27 64位 windows）
地址：'C:\\Python27\\lib\\site-packages\\zbar.pyd'

'/usr/local/lib/python2.7/dist-packages/zbar.so'

"test_cv2_new_distance_zbar.py"进行了二维码位置定位及内容识别，并计算了二维码离摄像头上下左右边缘的距离，
速度可达到实时，执行命令：
python test_cv2_new_distance_zbar.py


其中，# camera = cv2.VideoCapture(1)放开注释即可调用摄像头
camera = cv2.VideoCapture("./video4.mpeg")为调用录制视频文件


屏幕打印示例如下：
QR codes: ABC_001_abcd
('distance_left: ', 547.870361328125)
('distance_right: ', 412.129638671875)
('distance_up: ', 328.18426513671875)
('distance_down: ', 215.81573486328125)
