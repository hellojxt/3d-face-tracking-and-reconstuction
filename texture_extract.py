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
    tris = predictor.tri - 1
    colors = np.zeros_like(vertices)
    print(vertices.shape)

    for i in range(vertices.shape[1]):
        colors[:,i] = im[int(vertices[1,i]),int(vertices[0,i]),:]
    np.save('trump_texture',colors)
    im = render_texture(vertices, colors ,tris,im.shape[0],im.shape[1])
    cv2.rectangle(im, 
        (detection[0], detection[1]), 
        (detection[2], detection[3]), 
        COLORS_10[0], 2)
    print(vertices.shape)
cv2.imwrite('test.png',im)

    