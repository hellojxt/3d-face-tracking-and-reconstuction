import cv2
import os
import numpy as np
import dlib
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

class mtcnn_detector():
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MTCNN(keep_all=True,device=device)
    def detect(self, img):
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        boxes, probs= self.model.detect(img)
        if boxes is None:
            boxes = []
        
        boxs = []
        for bbox in boxes:
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

            llength = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)*0.7
            center_x = (bbox[2] + bbox[0]) / 2
            center_y = (bbox[3] + bbox[1]) / 2

            roi_box = [0] * 4
            if center_x - llength / 2 < 0 :
                llength = center_x * 2
            if center_y - llength / 2 < 0 :
                llength = center_y * 2
            roi_box[0] = center_x - llength / 2
            roi_box[1] = center_y - llength / 2
            roi_box[2] = roi_box[0] + llength
            roi_box[3] = roi_box[1] + llength
            boxs.append(roi_box)

        return np.array(boxs).astype(np.int)
        
class cv2_detector():
    def __init__(self):
        prototxt_path = 'Data/model_data/deploy.prototxt'
        caffemodel_path = 'Data/model_data/weights.caffemodel'
        self.model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    def detect(self, img ,confidence_ = 0.4):
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 
            1.0, 
            (300, 300), 
            (0, 0, 0)
        )
        self.model.setInput(blob)
        boxs = []
        detections = self.model.forward()
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            confidence = detections[0, 0, i, 2]
            if (confidence > confidence_):
                boxs.append(box.astype("int"))
        return np.array(boxs).astype(np.int)

class dlib_detector():
    def __init__(self,use_landmark = False):
        self.use_lk = use_landmark
        if use_landmark:
            dlib_landmark_model = 'Data/model_data/shape_predictor_68_face_landmarks.dat'
            self.face_regressor = dlib.shape_predictor(dlib_landmark_model)
        self.face_detector = dlib.get_frontal_face_detector()

    def detect(self, img):
        rects = self.face_detector(img, 1)
        boxs = []
        for rect in rects:
            if self.use_lk:
                pts = self.face_regressor(img, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                bbox = [min(pts[0,:]), min(pts[1,:]), max(pts[0,:]), max(pts[1,:])]
            else:
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]

            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

            llength = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
            center_x = (bbox[2] + bbox[0]) / 2
            center_y = (bbox[3] + bbox[1]) / 2

            roi_box = [0] * 4
            if center_x - llength / 2 < 0 :
                llength = center_x * 2
            if center_y - llength / 2 < 0 :
                llength = center_y * 2
            roi_box[0] = center_x - llength / 2
            roi_box[1] = center_y - llength / 2
            roi_box[2] = roi_box[0] + llength
            roi_box[3] = roi_box[1] + llength
            boxs.append(roi_box)
        return np.array(boxs).astype(np.int)


if __name__ == "__main__":
    d = mtcnn_detector()
    img = cv2.imread('1.jpg')
    boxs = d.detect(img)
    print(boxs)