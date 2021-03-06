import os
os.environ["CUDA_DEVICES"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
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


def show_flag(lst):
    for idx,i in enumerate(lst):
        print(idx)
        print(i.flags)

res = 256

detector = YOLOv3(0.5,0.4)
predictor = predictor()

frame_idx = 0
im = cv2.imread('1.jpg')
#im = cv2.imread('1.jpg')
mod = SourceModule(open('render.cu').read().
                    replace('WIDTH',str(im.shape[0])).replace('HEIGHT',str(im.shape[1]))
                    )
render_image = mod.get_function("render")
detections = detector.detect(im)
imgs = []

for detection in detections :
    detection = detection.astype(np.int)
    img = crop_img(im,detection[:4])
    params = predictor.predict(img)
    kpt = predictor.pst68(params, detection[:4])
    new_box = parse_roi_box_from_landmark(kpt)
    detection = new_box.astype(np.int)
    img = crop_img(im,detection[:4])
    params = predictor.predict(img)
    vertices = predictor.dense_vertices(params, detection[:4])
    tris = predictor.tri
    colors = np.load('texture.npy')
    render_image(
        drv.In(vertices.astype(np.float32)), 
        drv.In(np.ascontiguousarray(tris).astype(np.int32)), 
        drv.In(colors.astype(np.float32)),
        drv.In((np.zeros((im.shape[0],im.shape[1]))-99999).astype(np.int32)),
        drv.InOut(im),
        block=(400,1,1), grid=(1000,1))
    cv2.rectangle(im, 
        (detection[0], detection[1]), 
        (detection[2], detection[3]), 
        COLORS_10[0], 2)
cv2.imwrite('test.png',im)

    