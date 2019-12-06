from utils import predictor
import cv2
from utils.plot import plot_kpt
from utils.detector import *
p = predictor.predictor()
detector = mtcnn_detector()
im = cv2.imread('crop/45.png')
boxs = detector.detect(im)
box = boxs[0]
crop_img = im[box[1]:box[3],box[0]:box[2],:]
tmp = crop_img.astype(np.int)*4
tmp = tmp - np.mean(tmp)
crop_img = np.uint8(np.clip(tmp, 0, 255))
cv2.imwrite('crop.png',crop_img)
output = p.predict(crop_img)

pst = p.pst68(output,box).transpose()
im = plot_kpt(im,pst)
cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
cv2.imwrite('out.png',im)