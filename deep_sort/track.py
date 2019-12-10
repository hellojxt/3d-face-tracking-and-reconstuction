from deep_sort.kalman_filter import KalmanFilter

import torch

kf = KalmanFilter()

class Track():
    def __init__(self, measurement, feature, dim = 512):
        self.mean,self.cov = kf.initiate(measurement)
        self.maxnum = 60
        self.feature = torch.zeros((self.maxnum,dim)).cuda()
        self.feature_full = 1
        self.feature_index = 1
        self.feature[0] = feature

    def similarity(self, measurement, feature):
        d1 = kf.gating_distance(self.mean, self.cov, measurement)
        d2 = 1 - (self.feature[:self.feature_full]*feature).sum(-1).max().item()
        lamda = 0.5
        if d1 > 9.48:
            d1 = 100
        d = d1*lamda + d2*(1-lamda)
        return d

    def update(self, measurement, feature):
        if self.feature_full < self.maxnum:
            self.feature_full += 1
        self.feature[self.feature_index] = feature
        self.feature_index = (self.feature_index + 1) % self.maxnum
        self.mean,self.cov = kf.update(self.mean, self.cov, measurement)

    
    def predict(self):
        self.mean,self.cov = kf.predict(self.mean, self.cov)
