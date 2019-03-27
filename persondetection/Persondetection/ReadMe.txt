文件所需配置环境如下：

- python    `2.7.12`

- cv2       `3.4.2`

- numpy     `1.15.4`
- imtils    `0.5.1`


"Detectperson.py"根据opencv默认的SVM分类器实现在摄像头中找到人体，但实测发现效果一般，执行命令：
python Detectperson.py


其中，# get a frame from webcarmos
    ret, frame = cap.read()
使用普通web摄像头进行图像读取

    # get a frame from depthcarmos
    cframe = color_stream.read_frame()
    frame = np.array(cframe.get_buffer_as_triplet()).reshape(
        [color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3])
使用深度摄像头进行图像读取
