#!/usr/bin/env python3
import cv2
import time


formaty= 'pot_6_%s.jpg'
i = 0
while 1:
     vcap = cv2.VideoCapture("http://192.168.0.92/image/jpeg.cgi")
     ret, frame = vcap.read()
     print(ret)
     cv2.imwrite(formaty % i, frame)
     time.sleep(5)
     i += 1
