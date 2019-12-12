import cv2
import os
import numpy as np
import time
import torch
from PIL import Image
from utils.yolo import *
from utils.utils import *
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest")
    return image

class YOLOv3():
    def __init__(self,conf_thres = 0.5, nms_thres = 0.4):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.img_size = 416
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet('Data/cfg/detectHeadInference.cfg', img_size=self.img_size).to(device)
        model.load_darknet_weights('Data/cfg/detectHead_58000.weights')
        model.eval()
        self.model = model

    def detect(self, ori_img):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        img = transforms.ToTensor()(ori_img).cuda()
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.img_size)
        with torch.no_grad():
            detections = self.model(img)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)[0]
            if detections is None:
                return []
            
        detections = rescale_boxes(detections, self.img_size, ori_img.shape[:2])
        detections =  detections.detach().cpu().numpy()
        #detections[:4] = detections[:4].astype(np.int)
        return detections


# class mtcnn_detector():
#     def __init__(self):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = MTCNN(keep_all=True,device=device)
#     def detect(self, img):
#         img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#         boxes, probs= self.model.detect(img)
#         if boxes is None:
#             boxes = []
        
#         boxs = []
#         for bbox in boxes:
#             center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
#             radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
#             bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

#             llength = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)*0.7
#             center_x = (bbox[2] + bbox[0]) / 2
#             center_y = (bbox[3] + bbox[1]) / 2

#             roi_box = [0] * 4
#             if center_x - llength / 2 < 0 :
#                 llength = center_x * 2
#             if center_y - llength / 2 < 0 :
#                 llength = center_y * 2
#             roi_box[0] = center_x - llength / 2
#             roi_box[1] = center_y - llength / 2
#             roi_box[2] = roi_box[0] + llength
#             roi_box[3] = roi_box[1] + llength
#             boxs.append(roi_box)

#         return np.array(boxs).astype(np.int)
        
# class cv2_detector():
#     def __init__(self):
#         prototxt_path = 'Data/model_data/deploy.prototxt'
#         caffemodel_path = 'Data/model_data/weights.caffemodel'
#         self.model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
#     def detect(self, img ,confidence_ = 0.4):
#         (h, w) = img.shape[:2]
#         blob = cv2.dnn.blobFromImage(
#             cv2.resize(img, (300, 300)), 
#             1.0, 
#             (300, 300), 
#             (0, 0, 0)
#         )
#         self.model.setInput(blob)
#         boxs = []
#         detections = self.model.forward()
#         for i in range(0, detections.shape[2]):
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             confidence = detections[0, 0, i, 2]
#             if (confidence > confidence_):
#                 boxs.append(box.astype("int"))
#         return np.array(boxs).astype(np.int)

# class dlib_detector():
#     def __init__(self,use_landmark = False):
#         self.use_lk = use_landmark
#         if use_landmark:
#             dlib_landmark_model = 'Data/model_data/shape_predictor_68_face_landmarks.dat'
#             self.face_regressor = dlib.shape_predictor(dlib_landmark_model)
#         self.face_detector = dlib.get_frontal_face_detector()

#     def detect(self, img):
#         rects = self.face_detector(img, 1)
#         boxs = []
#         for rect in rects:
#             if self.use_lk:
#                 pts = self.face_regressor(img, rect).parts()
#                 pts = np.array([[pt.x, pt.y] for pt in pts]).T
#                 bbox = [min(pts[0,:]), min(pts[1,:]), max(pts[0,:]), max(pts[1,:])]
#             else:
#                 bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]

#             center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
#             radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
#             bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

#             llength = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
#             center_x = (bbox[2] + bbox[0]) / 2
#             center_y = (bbox[3] + bbox[1]) / 2

#             roi_box = [0] * 4
#             if center_x - llength / 2 < 0 :
#                 llength = center_x * 2
#             if center_y - llength / 2 < 0 :
#                 llength = center_y * 2
#             roi_box[0] = center_x - llength / 2
#             roi_box[1] = center_y - llength / 2
#             roi_box[2] = roi_box[0] + llength
#             roi_box[3] = roi_box[1] + llength
#             boxs.append(roi_box)
#         return np.array(boxs).astype(np.int)


# if __name__ == "__main__":
#     d = mtcnn_detector()
#     img = cv2.imread('1.jpg')
#     boxs = d.detect(img)