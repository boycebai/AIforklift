�ļ��������û������£�

- python    `2.7.12`

- cv2       `3.4.2`

- numpy     `1.15.4`

- Pillow    `3.1.2`
- zbar

zbar��װ�̳̣�
pip install zbar
��ע�����������python27 64λ windows��
��ַ��'C:\\Python27\\lib\\site-packages\\zbar.pyd'

'/usr/local/lib/python2.7/dist-packages/zbar.so'

"test_cv2_new_distance_zbar.py"�����˶�ά��λ�ö�λ������ʶ�𣬲������˶�ά��������ͷ�������ұ�Ե�ľ��룬
�ٶȿɴﵽʵʱ��ִ�����
python test_cv2_new_distance_zbar.py


���У�# camera = cv2.VideoCapture(1)�ſ�ע�ͼ��ɵ�������ͷ
camera = cv2.VideoCapture("./video4.mpeg")Ϊ����¼����Ƶ�ļ�


��Ļ��ӡʾ�����£�
QR codes: ABC_001_abcd
('distance_left: ', 547.870361328125)
('distance_right: ', 412.129638671875)
('distance_up: ', 328.18426513671875)
('distance_down: ', 215.81573486328125)
