import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import cv2
import numpy as np
import time
from utils.predictor import predictor
from utils.plot import *
from utils.utils import *
from deep_sort import DeepSort
from utils.detector import YOLOv3

class video_loader():
    def __init__(self, path):
        self.list = []
        if os.path.isdir(path):
            path_lst = []
            for filename in os.listdir(path):
                if filename[-3:] == 'bmp':
                    path_lst.append(os.path.join(path, filename))
            path_lst.sort()
            for path in path_lst:
                self.list.append(cv2.imread(path))
        else:
            video = cv2.VideoCapture(path)
            while video.grab(): 
                _, im = video.retrieve()
                self.list.append(im)
        print(self.list[0].shape)
        self.width = self.list[0].shape[1]
        self.height = self.list[0].shape[0]
    



res = 256
video = video_loader('test1.mp4')
frame_width = int(video.width)
frame_height = int(video.height)
output = cv2.VideoWriter('output.avi',
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            20, (frame_width,frame_height))

detector = YOLOv3(0.5,0.4)
deepsort = DeepSort()
predictor = predictor()

frame_idx = 0
for im in video.list:
    start = time.time()
    #print('detection:')
    detections = detector.detect(im)
    imgs = []
    
    for d in detections:
        d = d[:4].astype(np.int)
        #print(d)
        imgs.append(im[d[1]:d[3],d[0]:d[2],:])
    detections, ids = deepsort.update(detections, imgs)

    
    for detection,id in zip(detections,ids):
        detection = detection.astype(np.int)
        crop_img = im[detection[1]:detection[3],detection[0]:detection[2],:]
        cv2.rectangle(im, 
            (detection[0], detection[1]), 
            (detection[2], detection[3]), 
            COLORS_10[id], 2)

        label = "id:{}".format(id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.putText(im,label,
            (detection[0],detection[1]+t_size[1]+4), 
            cv2.FONT_HERSHEY_PLAIN, 
            2, COLORS_10[id], 2)
        params = predictor.predict(crop_img)
        kpt = predictor.pst68(params, detection).transpose()
        im = plot_kpt(im,kpt)
    
    #print('deep sort:')
    #print(detections)
    print(time.time() - start)
    output.write(im)
    cv2.imwrite('test.png',im)
output.release()
    