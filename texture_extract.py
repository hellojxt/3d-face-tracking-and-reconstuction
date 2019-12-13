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


def distance(p1, p2, length):
    d = np.sum((p1-p2)*(p1-p2)*[1,0.6])
    d = d*4/(length**2)
    return np.exp(-np.power(d, 2.) / (2 * np.power(0.4, 2.)))
res = 256

detector = YOLOv3(0.5,0.4)
predictor = predictor()

frame_idx = 0
im = cv2.imread('trump.jpg')
#im = cv2.imread('1.jpg')
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
    colors = np.zeros((4,vertices.shape[1]))
    print(vertices.shape)
    print(im.shape)
    center = (detection[:2] + detection[2:])/2
    length = detection[2] - detection[0]
    for i in range(vertices.shape[1]):
        colors[:3,i] = im[int(vertices[1,i]),int(vertices[0,i]),:]
        colors[3,i] = distance(vertices[:2,i],center,length)
    np.save('Data/texture',colors)
    im = render_texture(vertices, colors ,tris,im.shape[0],im.shape[1])
    cv2.rectangle(im, 
        (detection[0], detection[1]), 
        (detection[2], detection[3]), 
        COLORS_10[0], 2)
    print(vertices.shape)
cv2.imwrite('test.png',im)

    