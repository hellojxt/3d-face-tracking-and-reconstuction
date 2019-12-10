from deep_sort.track import Track
from deep_sort.feature_extractor import Extractor
import numpy as np
from scipy.optimize import linear_sum_assignment

def boxs_to_xyrhs(boxs):
    xyrhs = []
    for box in boxs:
        width = box[2] - box[0]
        height = box[3] - box[1]
        xyrhs.append([
            box[0]+width/2, 
            box[1]+height/2,
            width/height,
            height
        ])
    return np.array(xyrhs)

def xyrhs_to_boxs(xyrhs):
    boxs = []
    for xyrh in xyrhs:
        width = xyrh[2]*xyrh[3]
        height = xyrh[3]
        boxs.append([
            xyrh[0] - width / 2,
            xyrh[1] - height / 2,
            xyrh[0] + width / 2,
            xyrh[1] + height / 2,
        ])
    return np.array(boxs)

fe = Extractor('Data/ckpt.t7')


class DeepSort():
    def __init__(self):
        self.tracks = []
        
    def update(self, boxs, imgs):
        for track in self.tracks:
            track.predict()

        if len(boxs) == 0:
            return [],[]
        xyrhs = boxs_to_xyrhs(boxs)
        features = fe(imgs)
        track_ids, m_ids = [],[]

        if len(self.tracks) > 0 and len(xyrhs) > 0: 
            cost = np.zeros((len(self.tracks), len(xyrhs)))
            for i,track in enumerate(self.tracks):
                for j,feature in enumerate(features):
                    cost[i,j] = track.similarity(xyrhs[j],feature)
            track_ids, m_ids = linear_sum_assignment(cost)
            #print(track_ids,m_ids)
            #print(xyrhs)
            for id1,id2 in zip(track_ids, m_ids):
                self.tracks[id1].update(xyrhs[id2],features[id2])
        
        for i,xyrh in enumerate(xyrhs):
            if i not in m_ids:
                self.tracks.append(Track(xyrh, features[i]))

        return xyrhs_to_boxs(xyrhs[m_ids]), track_ids



        