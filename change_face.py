import os
import sys
import cv2
import numpy as np
import time
from utils.predictor import predictor
from utils.plot import *
from utils.utils import *
from deep_sort import DeepSort
from utils.detector import YOLOv3
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

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

if __name__ == "__main__":
    deepsort = DeepSort()
    predictor = predictor()
    res = 256
    video = video_loader(sys.argv[1])
    mod = SourceModule(open('render.cu').read().
                        replace('WIDTH',str(video.height)).replace('HEIGHT',str(video.width))
                        )
    render_image = mod.get_function("render")
    frame_width = int(video.width)
    frame_height = int(video.height)
    output = cv2.VideoWriter(sys.argv[2],
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                20, (frame_width,frame_height))

    detector = YOLOv3(0.5,0.4)


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
            img = crop_img(im,detection[:4])
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
            params = predictor.predict(img)
            kpt = predictor.pst68(params, detection)
            new_box = parse_roi_box_from_landmark(kpt)
            detection = new_box.astype(np.int)
            img = crop_img(im,detection[:4])
            params = predictor.predict(img)
            kpt = predictor.pst68(params, detection[:4])
            im = plot_kpt(im,kpt)
            if id == 0:
                vertices = predictor.dense_vertices(params, detection[:4])
                tris = predictor.tri
                colors = predictor.colors
                render_image(
                    drv.In(vertices.astype(np.float32)), 
                    drv.In(np.ascontiguousarray(tris).astype(np.int32)), 
                    drv.In(colors.astype(np.float32)),
                    drv.In((np.zeros((im.shape[0],im.shape[1]))-99999).astype(np.int32)),
                    drv.InOut(im),
                    block=(400,1,1), grid=(500,1)
                )
        #print('deep sort:')
        #print(detections)
        print(time.time() - start)
        output.write(im)

    output.release()
        