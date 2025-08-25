import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

class Z_score(object):
    def __init__(self, strid, channel, threshold=0.001):
        self.length = int(1/strid)
        self.grid_martix = np.ones((channel, self.length, self.length, self.length))
        self.threshold = threshold


    def diff_deal(self, data):
        diff_tensor = torch.diff(data, dim=2)
        max_value = (torch.max(diff_tensor, dim=2, keepdim=True).values * self.length).to(dtype=torch.int)
        min_value = (torch.min(diff_tensor, dim=2, keepdim=True).values * self.length).to(dtype=torch.int)
        median_abs_mean = (torch.mean(torch.abs(diff_tensor), dim=2, keepdim=True) * self.length).to(dtype=torch.int)
        feats = torch.stack([max_value, min_value, median_abs_mean], dim=2)
        return feats.squeeze()

    def update(self, data):
        batch, channel, window_size = data.shape

        feats = self.diff_deal(data)
        for i in tqdm(range(batch), desc="Processing batch"):
            for j in range(channel):
                feat = feats[i, j, :]
                self.get_neighbors((j, feat[0], feat[1], feat[2]))


    def compute(self, sample):
        batch, channel, window_size = sample.shape
        rc = torch.zeros([batch, channel])
        feats = self.diff_deal(sample)
        for i in range(batch):
            for j in range(channel):
                feat = feats[i, j, :]
                rc[i, j] = float(self.grid_martix[j, feat[0], feat[1], feat[2]])
        return rc


    def get_neighbors(self, coord):

        j, x, y, z = coord
        neighbors = [
            (x, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y, z + 1),
            (x, y, z - 1)
        ]

        valid_neighbors = [
            (nx, ny, nz) for nx, ny, nz in neighbors
            if 0 <= nx < self.length and 0 <= ny < self.length and 0 <= nz < self.length
        ]
        for _ in valid_neighbors:
            if self.grid_martix[j, _[0], _[1], _[2]] == 1:
                self.grid_martix[j, _[0], _[1], _[2]] = self.threshold



