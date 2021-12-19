from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from numpy import genfromtxt
import os
import glob
import torch
import random
import sys

act_dir = {'none': 0, 'walk': 1, 'run': 2, 'sit': 3, 'fall': 4}


def check_exists(path):
    return os.path.exists(path)


def getLabel(path):
    for a in ['None', 'walk', 'run', 'sit', 'fall']:
        if a in path:
            return act_dir[a]
    return 0


# 目前是按照文件划分，可以考虑增加功能，按照env划分，按照date划分，按照人员划分。。。
def train_test_split(activity_list, datatype, har_data_dir, train_ratio):
    '''
    Splits the data files to training set and test set for every activity according to the given ratio
    :param activity_list: lists activities that will be involved in HAR.
    :param datatype: 'point' or 'target'
    :param dir: root dir of HAR data files.
    :param train_ratio: The ratio of training set files
    :return:
    '''

    train_file_list = []
    test_file_list = []

    for activity in activity_list:
        act_file_list = []
        a_path = os.path.join(har_data_dir, activity)

        if not check_exists(a_path):
            print('Warning:Dataset of ' + activity + ' not found.')
            break

        for root, dirs, _ in os.walk(a_path):
            for dir in dirs:
                raw_path = os.path.join(root, dir, 'raw/*'+datatype+'*.csv')
                raw_data_files = glob.glob(raw_path)
                act_file_list.extend(raw_data_files)

        random.Random(4).shuffle(act_file_list)
        # random.shuffle(act_file_list)
        split_num = round(len(act_file_list) * train_ratio)
        train_file_list.extend(act_file_list[:split_num])
        test_file_list.extend(act_file_list[split_num:])

    return train_file_list, test_file_list


class HARDataset(Dataset):

    def __init__(self, path, type, act_list, concat_framNum = 1, seq_len=10, file_list=None, pointLSTM = False):
        """可以在初始化函数当中对数据进行一些操作，比如读取、归一化等"""
        # self.data = np.loadtxt(path)  # 读取 txt 数据
        # self.x = self.data[:, 1:]  # 输入变量
        # self.y = self.data[:, 0]  # 输出变量
        self.act_dir = act_dir
        self.type = type
        self.activity_list = act_list  # ['walk','run','sit','run','stand','none']
        self.datapath = path  # '/har_data/'
        self.seq_len = seq_len
        self.concat_frameNum = concat_framNum
        if file_list != None:
            self.file_list = file_list
        # self.type = type # 'target' or 'point'

        if self.type == 'target':
            if file_list != None:
                self.x, self.y = self.createTargetDetesetWithFileList(self.file_list)
            else:
                self.x, self.y = self.createTargetDeteset(self.activity_list)
        elif self.type == 'point':
            if pointLSTM == False:
                self.x, self.y = self.createPointDatasetWithFileList_v1(self.file_list)
            else:
                self.x, self.y = self.createPointDatasetWithFileList_v2(self.file_list)
        elif self.type == 'PAT':
            self.x_p, self.x_t, self.y = self.createPATDatasetWithFileList(self.file_list)  #file_list for points
        else:
            print('The type of dataset is not supported!')
            sys.exit(1)

    def createTargetDetesetWithFileList(self, file_list):
        '''
        Creates target dataset according to the given list
        :param file_list: lists the file paths in the dataset you want to create.
        :return: dataset including data and label in tensor format.
        '''
        x = np.array([])
        y = []
        act_data = np.array([])
        for file in file_list:
            data = genfromtxt(file, delimiter=',')[1:, 2:11]  # pos_x, pos_y, pos_z, vx, vy, vz, ax, ay, az
            if len(data) < self.seq_len:
                print("Warning: Total frame number is less than seq_len.")
                break
            for i in range(len(data)):
                if i + self.seq_len < len(data):
                    if x.size != 0:
                        x = np.concatenate((x, data[i:i + self.seq_len][np.newaxis, :, :]))
                    else:
                        x = data[i:i + self.seq_len][np.newaxis, :, :]
                    y = y + [getLabel(file)]
                else:
                    break

        if x.size == 0:
            print('Warning: Dataset is not found!')
            sys.exit(1)

        return torch.tensor(x.tolist()), y

    def createTargetDataset(self, activity_list):
        x = np.array([])
        y = []

        for activity in activity_list:
            act_data = np.array([])
            a_path = os.path.join(self.datapath, activity)

            if not check_exists(a_path):
                print('Warning:Dataset of ' + activity + ' not found.')
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
                                    act_data = np.concatenate((act_data, data[i:i + self.seq_len][np.newaxis, :, :]))
                                else:
                                    act_data = data[i:i + self.seq_len][np.newaxis, :, :]
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

        if x.size == 0:
            print('Error: Dataset is not found.')
            sys.exit(1)

        return torch.tensor(x.tolist()), y

    def createPointDatasetWithFileList_v1(self, file_list):
        '''
        Creates dataset for point data.
        Concatenate points in the recent self.concat_frameNum frames as a sample.
        Pads the total number of points to 16*self.concat_frameNum.
        The shape of a sample is (16*self.concat_frameNum)*5
        :param file_list: lists the file paths in the dataset you want to create.
        :return: point dataset
        '''
        # 待添加筛选出有效点，去掉 target_id 大于254的点
        x = np.array([])
        y = []

        for file in file_list:
            point_data = genfromtxt(file, delimiter=',') # list无法通过 slice 直接获取某一列，所以别用tolist()转成list
            frame_id = point_data[1:,0]
            point_attri = point_data[1:, 2:7]
            end_frame_id = frame_id[-1]
            i = 0
            while i < len(point_data):
                current_frame_id = frame_id[i]
                c = 0
                threshold = current_frame_id + self.concat_frameNum
                if threshold > end_frame_id:  # if the left frames is not enough to construct a sample.
                    break
                while frame_id[i+c] < threshold:
                    c += 1

                tmp_array = point_attri[i:i+c]
                if len(tmp_array) == 0:
                    continue
                # pad the point number to 16*self.concat_frameNum
                if len(tmp_array) > (16*self.concat_frameNum):
                    sample = tmp_array[:16*self.concat_frameNum]
                else:
                    sample = np.tile(tmp_array,(int(np.ceil((16*self.concat_frameNum)/len(tmp_array))),1))[:16*self.concat_frameNum]
                # concatenate the sample to dataset
                if len(x) == 0:
                    x = sample[np.newaxis, :, :]
                else:
                    x = np.concatenate((x, sample[np.newaxis, :, :]))
                    # x = np.array([x, sample])# 这样会报栈溢出的错 "Fatal Python error: Cannot recover from stack overflow"

                # get the sample label
                y = y + [getLabel(file)]

                # find the next frame_id
                while frame_id[i] == current_frame_id:
                    i += 1

        if len(x) == 0:
            print('Warning: Dataset is not found!')
            sys.exit(1)

        return torch.tensor(x.tolist()), y  # x.shape = (16*self.concat_framNum, 5)

    def createPointDatasetWithFileList_v2(self, file_list):
        x = np.array([])
        y = []

        for file in file_list:
            point_data = genfromtxt(file, delimiter=',')  # list无法通过 slice 直接获取某一列，所以别用tolist()转成list
            frame_id = point_data[1:, 0]
            point_attri = point_data[1:, 2:7]
            end_frame_id = frame_id[-1]
            i = 0
            while i < len(point_data):
                curSeq_frame_id = frame_id[i]

                thre = curSeq_frame_id + (self.concat_frameNum - 1) + self.seq_len
                if thre >= end_frame_id:  # if the left frames is not enough to construct a sample.
                    break
                j = i
                c_sample = 0
                samples_array = np.array([])

                while c_sample < self.seq_len:
                    curSam_frame_id = frame_id[j]
                    thre_1 = curSam_frame_id + self.concat_frameNum
                    c_frame = 0
                    frame_len = len(frame_id)
                    while j+c_frame < frame_len and frame_id[j + c_frame] < thre_1:
                        c_frame += 1

                    tmp_sample = point_attri[j:j + c_frame]
                    if len(tmp_sample) == 0:
                        continue
                    # pad the point number to 16*concat_frameNum
                    if len(tmp_sample) > (16 * self.concat_frameNum):
                        sample = tmp_sample[:16 * self.concat_frameNum]
                    else:
                        sample = np.tile(tmp_sample, (int(np.ceil((16 * self.concat_frameNum) / len(tmp_sample))), 1))[
                             :16 * self.concat_frameNum]
                    if len(samples_array) == 0:
                        samples_array = sample[np.newaxis, :, :]
                    else:
                        samples_array = np.concatenate((samples_array, sample[np.newaxis, :, :]))

                    c_sample += 1

                    if frame_id[j] == end_frame_id:
                        break
                    while frame_id[j] == curSam_frame_id:
                        j += 1

                if len(samples_array) != 10:
                    break
                # concatenate the sample to dataset
                if len(x) == 0:
                    # x.shape=(sampleNums, seq_len, 16*concat_framNum, attriNum)
                    x = samples_array[np.newaxis, :, :]  # samples_array.shape = (seq_len, pointNum=(16*concat_framNum, arrtiNum)
                else:
                    x = np.concatenate((x, samples_array[np.newaxis, :, :]))
                    # x = np.array([x, sample])# 这样会报栈溢出的错 "Fatal Python error: Cannot recover from stack overflow"

                # get the sample label
                y = y + [getLabel(file)]

                # find the next frame_id
                while frame_id[i] == curSeq_frame_id:
                    i += 1

        if len(x) == 0:
            print('Warning: Dataset is not found!')
            sys.exit(1)

        return torch.tensor(x.tolist()), y  #   x.shape = (self.seq_len, 16*self.concat_framNum, 5)


    def createPATDatasetWithFileList(self, file_list):
        '''
        Creates the dataset contains both point and target information
        :param file_list: lists the point file to create dataset
        :return: a dataset contains point samples, target samples and labels.
        '''
        x_data = np.array([])
        x_target = np.array([])
        y = []

        for file in file_list:
            target_data = genfromtxt(file.replace('point','target'), delimiter=',')
            t_frame_id = target_data[1:,0]
            target_attri = target_data[1:,2:11]
            point_data = genfromtxt(file, delimiter=',')  # list无法通过 slice 直接获取某一列，所以别用tolist()转成list
            frame_id = point_data[1:, 0]
            point_attri = point_data[1:, 2:7]
            end_frame_id = frame_id[-1]
            i = 0
            while i < len(point_data):
                curSeq_frame_id = frame_id[i]

                thre = curSeq_frame_id + (self.concat_frameNum - 1) + self.seq_len
                if thre >= end_frame_id:  # if the left frames is not enough to construct a sample.
                    break
                j = i
                c_sample = 0
                samples_array = np.array([])

                while c_sample < self.seq_len:
                    curSam_frame_id = frame_id[j]
                    thre_1 = curSam_frame_id + self.concat_frameNum
                    c_frame = 0
                    frame_len = len(frame_id)
                    while j+c_frame < frame_len and frame_id[j + c_frame] < thre_1:
                        c_frame += 1

                    tmp_sample = point_attri[j:j + c_frame]
                    if len(tmp_sample) == 0:
                        continue
                    # pad the point number to 16*concat_frameNum
                    if len(tmp_sample) > (16 * self.concat_frameNum):
                        sample = tmp_sample[:16 * self.concat_frameNum]
                    else:
                        sample = np.tile(tmp_sample, (int(np.ceil((16 * self.concat_frameNum) / len(tmp_sample))), 1))[
                             :16 * self.concat_frameNum]
                    if len(samples_array) == 0:
                        samples_array = sample[np.newaxis, :, :]
                    else:
                        samples_array = np.concatenate((samples_array, sample[np.newaxis, :, :]))

                    c_sample += 1

                    if frame_id[j] == end_frame_id:
                        break
                    while frame_id[j] == curSam_frame_id:
                        j += 1

                if len(samples_array) != self.seq_len:
                    break
                # concatenate the sample to dataset
                idx = np.where(t_frame_id[:] == frame_id[i])[0][0]
                if len(x_data) == 0:    # 因为目前data_collecting 逻辑是有一个target才会记录数据，所以point的帧数小于等于target
                    # 所以记录的时候先以point为准，有point才做成sample
                    # x_data.shape=(sampleNums, seq_len, 16*concat_framNum, attriNum)
                    x_data = samples_array[np.newaxis, :, :]  # samples_array.shape = (seq_len, pointNum=(16*concat_framNum, arrtiNum)
                    x_target = target_attri[idx:idx+self.seq_len][np.newaxis,:,:]
                else:
                    x_data = np.concatenate((x_data, samples_array[np.newaxis, :, :]))
                    # x = np.array([x, sample])# 这样会报栈溢出的错 "Fatal Python error: Cannot recover from stack overflow"
                    if len(x_target) == 0:
                        x_target = target_attri[idx:idx+self.seq_len][np.newaxis,:,:]
                    else:
                        x_target = np.concatenate((x_target,target_attri[idx:idx+self.seq_len][np.newaxis,:,:]))
                # get the sample label
                y = y + [getLabel(file)]

                # find the next frame_id
                while frame_id[i] == curSeq_frame_id:
                    i += 1

        if len(x_data) == 0:
            print('Warning: Dataset is not found!')
            sys.exit(1)

        return torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), y
        #   x_data.shape = (sampleNum, self.seq_len, 16*self.concat_framNum, 5)
        #   x_target.shape = (sampleNum, self.seq_len, attriNum=9)




    def __len__(self):
        """返回数据集当中的样本个数"""
        return len(self.y)

    def __getitem__(self, index):
        """返回样本集中的第 index 个样本；输入变量在前，输出变量在后"""
        if self.type == 'PAT':
            return self.x_p[index,:], self.x_t[index,:], self.y[index]
        else:
            return self.x[index, :], self.y[index]


# dataDir = "../har_data/"
# activity_list = ['walk', 'sit', 'fall']
# train_files, test_files = train_test_split(activity_list, 'target', dataDir, 0.75)
# print(len(train_files))
# train_files, test_files = train_test_split(activity_list, 'point', dataDir, 0.75)
# train_dataset = HARDataset(dataDir, 'point', ['walk', 'sit', 'fall'], 10, train_files)

# print(len(train_files))

# train_dataset = MyDataSet(dataDir, 'target', ['walk', 'sit', 'fall'], 10, train_files)
# data_set = MyDataSet(dataDir, 'target', ['walk', 'sit', 'fall'],5)
# data_loader = DataLoader(dataset=data_set, batch_size=200, shuffle=True, drop_last=False)
