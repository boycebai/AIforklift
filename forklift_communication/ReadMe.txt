文件所需配置环境如下：

- python    `2.7.12`

- pyserial  `3.4`



"serial.py"进行了python控制叉车，并将运动指令传输给AGV控制模块，执行命令：
python serial.py


"path_planning.py"进行了根据服务器指令处理摄像头输出的指令，执行命令：
python path_planning.py
