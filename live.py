import time
import argparse  # 命令行模块
import cv2
import numpy as np
import configparser
from retinaface import Retinaface
import pymysql
import base64
import os
import multiprocessing
import logging

import logging.config
a = str(round(time.time()))
print(a)

video_path = 'rtmp://yunlive4.nbs.cn/live/2021njnews'
capture = cv2.VideoCapture(video_path)
while (True):
    t1 = time.time()
    # 读取某一帧
    ref, frame = capture.read()
    if not ref:
        break
    cv2.imshow("video", frame)
    c = cv2.waitKey(1) & 0xff
