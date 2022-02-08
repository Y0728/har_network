from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from numpy import genfromtxt
import os
import glob
import torch
import sys


class HARDataset(Dataset):

    def __init__(self, path, type, act_list, seq_len = 10):
        """可以在初始化函数当中对数据进行一些操作，比如读取、归一化等"""
        # self.data = np.loadtxt(path)  # 读取 txt 数据
        # self.x = self.data[:, 1:]  # 输入变量
        # self.y = self.data[:, 0]  # 输出变量
        self.act_dir = {'none': 0, 'walk': 1, 'run': 2, 'sit': 3, 'fall': 4}
        self.activity_list = act_list  # ['walk','run','sit','run','stand','none']
        self.datapath = path  # '/har_data/'
        self.seq_len = seq_len
        # self.type = type # 'target' or 'point'

        if type == 'target':
            self.x, self.y = self.createTargetDataset(self.activity_list)
        elif type == 'point':
            self.x, self.y = self.createPointDataset(self.activity_list)
        else:
            print('The type of dataset is not supported!')
            sys.exit(1)

    def check_exists(self, path):
        return os.path.exists(path)

    def createTargetDataset(self, activity_list):
        x = np.array([])
        y = []

        for activity in activity_list:
            act_data = np.array([])
            a_path = os.path.join(self.datapath, activity)

            if not self.check_exists(a_path):
                print('Warning:Dataset of '+ activity+' not found.')
                break

            for root, dirs, _ in os.walk(a_path):
                for dir in dirs:
                    raw_path = os.path.join(root, dir, 'raw/*target*.csv')
                    raw_data_files = glob.glob(raw_path)

                    for file in raw_data_files:
                        data = genfromtxt(file, delimiter=',')[1:, 2:11]  # pos_x, pos_y, pos_z, vx, vy, vz, ax, ay, az
                        if len(data) < self.seq_len:
                            print("Warning: Total frame number is less than seq_len.")
                            break
                        for i in range(len(data)):
                            if i + self.seq_len < len(data):
                                if act_data.size != 0:
                                    act_data = np.concatenate((act_data, data[i:i+self.seq_len][np.newaxis,:,:]))
                                else:
                                    act_data = data[i:i+self.seq_len][np.newaxis,:,:]
                            else:
                                break
                        # if act_data.size != 0:
                        #     act_data = np.concatenate((act_data,data))
                        # else:
                        #     act_data = data
                break

            if act_data.size == 0:
                print('Warning: The raw data of ' + activity + ' is not found!')
            else:
                if x.size == 0:
                    x = act_data
                else:
                    x = np.concatenate((x, act_data))
                y = y + ([self.act_dir[activity] for i in range(len(act_data))])
                # if y.size == 0:
                #     y = np.ones((len(act_data), 1)) * self.act_dir[activity]
                # else:
                #     y = np.concatenate((y, np.ones((len(act_data), 1)) * self.act_dir[activity]))

        if x.size == 0:
            print('Error: Dataset is not found.')
            sys.exit(1)

        return torch.tensor(x.tolist()), y

    def createPointDataset(self, activity_list):
        pass

    def __len__(self):
        """返回数据集当中的样本个数"""
        return len(self.x)

    def __getitem__(self, index):
        """返回样本集中的第 index 个样本；输入变量在前，输出变量在后"""
        return self.x[index, :], self.y[index]

dataDir = "../har_data/"
data_set = HARDataset(dataDir, 'target', ['walk', 'sit', 'fall'],5)
data_loader = DataLoader(dataset=data_set, batch_size=200, shuffle=True, drop_last=False)
