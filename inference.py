import os
import cv2
import numpy as np
import time
from utils.predictor import predictor
from utils.detector import *
from utils.plot import *
import dlib


res = 256
video = cv2.VideoCapture('1.mp4')
frame_width = int(video.get(3))
frame_height = int(video.get(4))
output = cv2.VideoWriter('output.avi',
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            30, (frame_width,frame_height))

detector = mtcnn_detector()
predictor = predictor()
frame_idx = 0
crop_idx = 0
while video.grab(): 
    start = time.time()
    _, im = video.retrieve()
    boxs = detector.detect(im)
    if len(boxs) == 0:
        continue
    for box in boxs:
        crop_idx += 1
        crop_img = im[box[1]:box[3],box[0]:box[2],:]
        cv2.imwrite('crop/{}.png'.format(crop_idx),crop_img)
        params = predictor.predict(crop_img)
        kpt = predictor.pst68(params, box).transpose()
        im = plot_kpt(im,kpt)
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)

    end = time.time()
    print("time: {}s, fps: {}".format(end-start, 1/(end-start)))
    output.write(im)
    
output.release()
