�ļ��������û������£�

- python    `2.7.12`

- cv2       `3.4.2`

- numpy     `1.15.4`
- imtils    `0.5.1`


"Detectperson.py"����opencvĬ�ϵ�SVM������ʵ��������ͷ���ҵ����壬��ʵ�ⷢ��Ч��һ�㣬ִ�����
python Detectperson.py


���У�# get a frame from webcarmos
    ret, frame = cap.read()
ʹ����ͨweb����ͷ����ͼ���ȡ

    # get a frame from depthcarmos
    cframe = color_stream.read_frame()
    frame = np.array(cframe.get_buffer_as_triplet()).reshape(
        [color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3])
ʹ���������ͷ����ͼ���ȡ
