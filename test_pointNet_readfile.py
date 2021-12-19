import os
import sys
import torch
import numpy as np
print(sys.path)
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from numpy import genfromtxt


from har_dataset_train_test_split import train_test_split, HARDataset

data_path = '../har_data'
activity_list = ['walk', 'sit', 'fall']
train_files, test_files = train_test_split(activity_list, 'point' ,data_path, 0.75)

x = np.array([])
y = []

for file in train_files:
    sample = []
    print(file)
    point_data = genfromtxt(file, delimiter=',')  # list无法通过 slice 直接获取某一列，所以别用tolist()转成list
    print(point_data.shape)
    frame_id = point_data[1:, 0]
