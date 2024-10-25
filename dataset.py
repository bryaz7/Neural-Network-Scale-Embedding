import glob
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

from torchvision import transforms
from torchvision.utils import make_grid

####
class DatasetSerial(data.Dataset):
    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)
               
    def __init__(self, pair_list, shape_augs=None, input_augs=None):
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):

        pair = self.pair_list[idx]

        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1] # normal is 0

        # shape must be deterministic so it can be reused
        shape_augs = self.shape_augs.to_deterministic()
        input_img = shape_augs.augment_image(input_img)

        # additional augmentation just for the input
        if self.input_augs is not None:
            input_img = self.input_augs.augment_image(input_img)

        return input_img, img_label
        
    def __len__(self):
        return len(self.pair_list)
    
####
def prepare_data():

    data_files = '/mnt/dang/data/SMHTMAs/core_grade.txt'
    tma_list = ['160003', '161228', '162350', '163542', '164807']

    # label -1 means exclude
    grade_pair = {'B': 0,
                '3+3': 1, '3+4': 1, '4+3': 1,
                '4+4': 1, '3+5': 1, '5+3': 1,
                '4+5': 1, '5+4': 1, '5+5': 1}
    with open(data_files, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data_info = [line for line in reader] # [[path, label], etc]
    data_info = [list(t) for t in zip(*data_info)] # [path_list, label_list]
    data_info[1] = [grade_pair[label] for label in data_info[1]]
    data_info = [list(t) for t in zip(*data_info)] # [[path, label], etc]
    
    train_tma = ['160003', '161228', '162350']
    valid_tma = ['163542', '164807']
    print(valid_tma, train_tma)

    train_pairs = []
    for tma in train_tma:
        filtered = [pair for pair in data_info if tma in pair[0]]
        train_pairs.extend(filtered)

    valid_pairs = []
    for tma in valid_tma:
        filtered = [pair for pair in data_info if tma in pair[0]]
        valid_pairs.extend(filtered)

    return train_pairs, valid_pairs

####
def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size+1):
            sample = ds[data_idx+j]
            img = sample[0]
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size
