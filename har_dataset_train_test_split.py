#!/usr/bin/env Python
# coding=utf-8
import logging
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from numpy import genfromtxt
import os
import glob
import torch
import random
import sys
import h5py

act_dir = {'none': 0, 'walk': 1, 'run': 2, 'sit': 3, 'fall': 4,'stand':5,'squat':6,'bend':7, 'jump':8, 'lie_down':9}
act_list = []

def check_exists(path):
    return os.path.exists(path)


def getLabel(path):
    for a in ['None', 'walk', 'run', 'sit', 'fall','stand','squat','bend','jump','lie_down']:
        if a in path:
            return act_dir[a]
    return 0
# #包含None
# def getLabel_v2(path):
#     for a in ['None', 'walk', 'run', 'sit', 'fall','stand','squat','bend','jump','lie_down']:
#         if a in path:
#             if a in act_list:
#                 return act_list.index(a) + 1
#             else:
#                 act_list.append(a)
#                 return act_list.index(a) + 1
#
#     return 0

#不含None：
def getLabel_v2(path):
    for a in ['walk', 'run', 'sit', 'fall','stand','squat','bend','jump','lie_down']:
        if a in path:
            if a in act_list:
                return act_list.index(a)+1
            else:
                act_list.append(a)
                return act_list.index(a)+1   #已经做好的数据集就是从1开始的，只能在网络代码里边改了？


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

def save_dataset_to_h5(seq_len, concat_frame_num,act_list, x_point, x_target, y, dataset_type = None):
    '''
    Save the dataset to h5 file, and name it by seq_len, concate_frame_num, act_list
    :param seq_len:
    :param concat_frame_num:
    :param act_list:
    :param x_point:
    :param x_target:
    :param y:
    :param dataset_type:
    :return:
    '''
    if dataset_type == None:
        dataset_type = 'None'
    act_string = '_'.join(act_list)
    if DEBUG:
        dataset_dir = '../har_dataset/'
    else:
        dataset_dir = '../../har_dataset/'
    Path(dataset_dir).mkdir(exist_ok=True)
    f = h5py.File(f'{dataset_dir}{dataset_type}_{seq_len}_{concat_frame_num}_{act_string}.h5', 'w')
    # f = h5py.File(f'../../har_dataset/{dataset_type}_{seq_len}_{concat_frame_num}_{act_string}.h5', 'w')
    f.create_dataset('x_point', data=x_point)
    f.create_dataset('x_target', data=x_target)
    f.create_dataset('y',data=y)
    f.create_dataset('valid',data=1)
    f.close()

def dataset_exists(activity_list, seq_len, concat_frameNum, dataset_type = None):
    '''
    Check if the dataset has already existed.
    :param activity_list:
    :param seq_len:
    :param concat_frameNum:
    :param dataset_type:
    :return:
    '''
    if isinstance(activity_list,str):
        activity_list=[activity_list]
    if dataset_type == None:
        dataset_type = 'None'
    partition = "train"
    if DEBUG:
        file_list = glob.glob(f'../har_dataset/{dataset_type}_{seq_len}_{concat_frameNum}_*.h5')
    else:
        file_list = glob.glob(f'../../har_dataset/{dataset_type}_{seq_len}_{concat_frameNum}_*.h5')
    for file in file_list:
        if set(os.path.basename(file).strip('.h5').split('_',3)[3].split('_')) == set(activity_list):
            return file   # contains root dir
    return None

class HARDataset(Dataset):

    def __init__(self, path, type, act_list, concat_framNum = 1, seq_len=10, file_list=None, pointLSTM = False, dataset_type=None, pc_channel = 5):
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
        self.dataset_type=dataset_type
        self.pointLSTM = pointLSTM
        self.pc_channel = pc_channel
        if file_list != None:
            self.file_list = file_list
        # self.type = type # 'target' or 'point'

        if self.type == 'target':
            if file_list != None:
                self.x, self.y = self.createTargetDetesetWithFileList(self.file_list)
            else:
                self.x, self.y = self.createTargetDeteset(self.activity_list)
        elif self.type == 'point':   # 20200220改成不做seq_len的point,target,y
            if self.pointLSTM == False:
                self.x, self.y = self.createPointDatasetWithFileList_v1(self.file_list)
                # self.x_p,self.x_t,self.y=self.createPointTargetDatasetWithFileList(self.file_list)
            else:
                # 可以考虑共用，只在取数据的时候取point的数据
                # self.x, self.y = self.createPointDatasetWithFileList_v2(self.file_list)
                self.x_p, self.x_t, self.y = self.createPATDatasetWithFileList_save_h5(self.file_list)
        elif self.type == 'PAT':
            # self.x_p, self.x_t, self.y = self.createPATDatasetWithFileList(self.file_list)  #file_list for points
            self.x_p, self.x_t, self.y = self.get_dataset()
            # self.x_p, self.x_t, self.y = self.createPATDatasetWithFileList_save_h5(self.file_list)  #file_list for points
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
                y = y + [getLabel_v2(file)]

                # find the next frame_id
                while frame_id[i] == current_frame_id:
                    i += 1

        if len(x) == 0:
            print('Warning: Dataset is not found!')
            sys.exit(1)
        return torch.tensor(x.tolist()), [label-1 for label in y]  # x.shape = (16*self.concat_framNum, 5)

        # return torch.tensor(x.tolist()), y  # x.shape = (16*self.concat_framNum, 5)

    def createPointTargetDatasetWithFileList(self, file_list):
        '''
        Creates dataset for point data.
        Concatenate points in the recent self.concat_frameNum frames as a sample.
        Pads the total number of points to 16*self.concat_frameNum.
        The shape of a sample is (16*self.concat_frameNum)*5
        :param file_list: lists the file paths in the dataset you want to create.
        :return: point dataset
        '''
        # 待添加筛选出有效点，去掉 target_id 大于254的点
        fileName = dataset_exists(self.activity_list, self.seq_len, self.concat_frameNum, self.dataset_type,do_seq=self.pointLSTM)
        if fileName and h5py.File(fileName, 'r')['valid']:
            f = h5py.File(fileName, 'r')
            return torch.tensor(f['x_point'][()].tolist()), torch.tensor(f['x_target'][()].tolist()), f['y'][()]
            # torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), y
        # else:
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

                if len(samples_array) != self.seq_len:
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
                y = y + [getLabel_v2(file)]

                # find the next frame_id
                while frame_id[i] == curSeq_frame_id:
                    i += 1

        if len(x_data) == 0:
            print('Warning: Dataset is not found!')
            sys.exit(1)

        return torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), y
        # return x_data.tolist(), (x_target.tolist()), y

        #   x_data.shape = (sampleNum, self.seq_len, 16*self.concat_framNum, 5)
        #   x_target.shape = (sampleNum, self.seq_len, attriNum=9)

    def createPATDatasetWithFileList_save_h5(self, file_list):
        '''
        Creates the dataset contains both point and target information
        :param file_list: lists the point file to create dataset
        :return: a dataset contains point samples, target samples and labels.
        '''
        fileName = dataset_exists(self.activity_list, self.seq_len, self.concat_frameNum, self.dataset_type)
        if fileName and h5py.File(fileName, 'r')['valid']:
            f = h5py.File(fileName, 'r')
            return torch.tensor(f['x_point'][()].tolist()),torch.tensor(f['x_target'][()].tolist()), f['y'][()]-1
            # torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), y
        # else:
        x_data = np.array([])
        x_target = np.array([])
        y = []

        for file in file_list:
            print(file)
            # delimiter=None if '100' in file and 'run' in file else ','
            target_data = genfromtxt(file.replace('point', 'target'), delimiter=',')
            t_frame_id = target_data[1:, 0]
            target_attri = target_data[1:, 2:11]
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
                    while j + c_frame < frame_len and frame_id[j + c_frame] < thre_1:
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
                if len(x_data) == 0:  # 因为目前data_collecting 逻辑是有一个target才会记录数据，所以point的帧数小于等于target
                    # 所以记录的时候先以point为准，有point才做成sample
                    # x_data.shape=(sampleNums, seq_len, 16*concat_framNum, attriNum)
                    x_data = samples_array[np.newaxis, :,
                             :]  # samples_array.shape = (seq_len, pointNum=(16*concat_framNum, arrtiNum)
                    x_target = target_attri[idx:idx + self.seq_len][np.newaxis, :, :]
                else:
                    x_data = np.concatenate((x_data, samples_array[np.newaxis, :, :]))
                    # x = np.array([x, sample])# 这样会报栈溢出的错 "Fatal Python error: Cannot recover from stack overflow"
                    if len(x_target) == 0:
                        x_target = target_attri[idx:idx + self.seq_len][np.newaxis, :, :]
                    else:
                        x_target = np.concatenate((x_target, target_attri[idx:idx + self.seq_len][np.newaxis, :, :]))
                # get the sample label
                y = y + [getLabel_v2(file)]

                # find the next frame_id
                while frame_id[i] == curSeq_frame_id:
                    i += 1
            print(f'{file} processed')
        if len(x_data) == 0:
            print('Warning: Dataset is not found!')
            sys.exit(1)
        if self.seq_len == 1:
            x_data = np.squeeze(x_data)
            x_target=np.squeeze(x_target)
        # save_dataset_to_h5(self.seq_len, self.concat_frameNum, self.activity_list,x_data, x_target, y, self.dataset_type)
        # return torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), y
        save_dataset_to_h5(self.seq_len, self.concat_frameNum, self.activity_list,x_data, x_target, y, self.dataset_type)
        return torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), [i-1 for i in y]


    def createPATDatasetWithFileList_save_h5_v2(self,act, file_list):
        '''
        增加了act作为输入，把返回值类型改了，因为只需要v2制作指定的某一个act的数据集，保存也只保存一个act的数据集
        调用v2的是代码只需要处理一种状态数据，然后把v2返回的结果合到整体数据集上
        Creates the dataset contains both point and target information
        :param file_list: lists the point file to create dataset
        :return: a dataset contains point samples, target samples and labels.
        '''
        fileName = dataset_exists(act, self.seq_len, self.concat_frameNum, self.dataset_type)
        if fileName and h5py.File(fileName, 'r')['valid']:
            f = h5py.File(fileName, 'r')
            logging.info(f'{fileName} is loaded.')
            print(f'{fileName} is loaded.')
            return f['x_point'][()],f['x_target'][()], f['y'][()]-1
            # return torch.tensor(f['x_point'][()].tolist()),torch.tensor(f['x_target'][()].tolist()), f['y'][()]-1
            # torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), y
        # else:
        x_data = np.array([])
        x_target = np.array([])
        y = []
        for file in file_list:
            print(file)
            # delimiter=None if '100' in file and 'run' in file else ','
            target_data = genfromtxt(file.replace('point', 'target'), delimiter=',')
            t_frame_id = target_data[1:, 0]
            target_attri = target_data[1:, 2:11]
            point_data = genfromtxt(file, delimiter=',')  # list无法通过 slice 直接获取某一列，所以别用tolist()转成list
            if len(point_data[0,:]) < 2 + self.pc_channel:
                logging.info(f'{file} does not contain 7 attributes.')
                print(f'{file} does not contain 7 attributes.')
                continue
            if self.pc_channel == 7:  # columns = frame_id, target_id, pos_x, pos_y, pos_z, doppler, snr, range, azimuth, elevation
                point_data = point_data[:,[0,1,2,3,4,7,8,9,5,6]]
            frame_id = point_data[1:, 0]
            # point_attri = point_data[1:, 2:7]
            point_attri = point_data[1:, 2:2+self.pc_channel]

            end_frame_id = frame_id[-1]
            i = 0
            frame_len = len(frame_id)
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

                    while j + c_frame < frame_len and frame_id[j + c_frame] < thre_1:
                        c_frame += 1

                    tmp_sample = point_attri[j:j + c_frame]
                    ## 以下两行代码有问题，如果是只用既有point又有target的数据，那不会出现(len(tmp_sample)=0)的情况，如果是会用到有target而没有point的数据，那下面的continue会导致死循环，因为i没加1，所以下一个sample还是会continue
                    # if len(tmp_sample) == 0:
                    #     continue
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
                    last_sample_frame = frame_id[j]
                    while frame_id[j] == curSam_frame_id:
                        j += 1
                    if j < frame_len and frame_id[j] > last_sample_frame + 10:   #加，如果一个seq中两个相邻的sample之间的帧号大于10，认为中断，该seq作废？
                        break
                idx = np.where(t_frame_id[:] == frame_id[i])[0][0]
                if frame_id[j] == end_frame_id:
                    break
                if len(samples_array) != self.seq_len:
                    # 防止因为len(samples_array) != self.seq_len而break导致的死循环
                    while frame_id[i] < frame_id[j]:
                        i += 1
                    continue  # 220318 修改，break -> continue 如果samples_array的长度不到seq_len的话，继续下一个sample
                # concatenate the sample to dataset
                if len(x_data) == 0:  # 因为目前data_collecting 逻辑是有一个target才会记录数据，所以point的帧数小于等于target
                    # 所以记录的时候先以point为准，有point才做成sample
                    # x_data.shape=(sampleNums, seq_len, 16*concat_framNum, attriNum)
                    x_data = samples_array[np.newaxis, :,
                             :]  # samples_array.shape = (seq_len, pointNum=(16*concat_framNum, arrtiNum)
                    x_target = target_attri[idx:idx + self.seq_len][np.newaxis, :, :]
                else:
                    x_data = np.concatenate((x_data, samples_array[np.newaxis, :, :]))
                    # x = np.array([x, sample])# 这样会报栈溢出的错 "Fatal Python error: Cannot recover from stack overflow"
                    if len(x_target) == 0:
                        x_target = target_attri[idx:idx + self.seq_len][np.newaxis, :, :]
                    else:
                        x_target = np.concatenate((x_target, target_attri[idx:idx + self.seq_len][np.newaxis, :, :]))
                # get the sample label
                y = y + [getLabel_v2(file)]

                # find the next frame_id
                while frame_id[i] == curSeq_frame_id:
                    i += 1
            print(f'{file} processed')
        if len(x_data) == 0:
            logging.warning(f'{act} generates 0 data sample.')
            print(f'{act} generates 0 data sample.')
            # print('Warning: Dataset is not found!')
            # sys.exit(1)
        if self.seq_len == 1:
            x_data = np.squeeze(x_data)
            x_target=np.squeeze(x_target)
        # save_dataset_to_h5(self.seq_len, self.concat_frameNum, self.activity_list,x_data, x_target, y, self.dataset_type)
        # return torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), y
        # 感觉不用保存，因为已经拼接了，还存啥呀
        save_dataset_to_h5(self.seq_len, self.concat_frameNum, [act],x_data, x_target, y, self.dataset_type)
        # return torch.tensor(x_data.tolist()), torch.tensor((x_target.tolist())), [i-1 for i in y]
        return x_data, x_target, y


    def __len__(self):
        """返回数据集当中的样本个数"""
        return len(self.y)

    def __getitem__(self, index):
        """返回样本集中的第 index 个样本；输入变量在前，输出变量在后"""
        if self.type == 'PAT':
            return self.x_p[index,:], self.x_t[index,:], self.y[index]
        elif self.type=='point' and self.pointLSTM:
            return self.x_p[index,:], self.x_t[index,:], self.y[index]
        else:
            return self.x[index, :], self.y[index]

    def get_dataset(self):
        point_data = np.array([])
        target_data = np.array([])
        y = []
        act_dic = dict(zip(self.activity_list, range(len(self.activity_list))))
        logging.info(f'the act_dict is: {act_dic}.')
        for act in self.activity_list:
            if DEBUG:
                train_files_point, test_files_point = train_test_split([act], 'point', '../har_data', 0.75)
            else:
                train_files_point, test_files_point = train_test_split([act], 'point', '../../har_data', 0.75)

            if self.dataset_type=='train':
                file_list_for_one_act = train_files_point
            else:
                file_list_for_one_act = test_files_point

            x_p, x_t, y_label = self.createPATDatasetWithFileList_save_h5_v2(act, file_list_for_one_act)

            # file = f'../../har_dataset/{self.dataset_type}_{self.seq_len}_{self.concat_num}_{act}.h5'
            if len(point_data)==0:
                point_data=x_p
                target_data=x_t
            else:
                point_data = np.concatenate((point_data, x_p))
                target_data = np.concatenate((target_data, x_t))
            # assign label for act
            label = act_dic[act]
            y = y + [label for i in range(len(x_p))]
            # # 如果act的.h数据集已存在，直接读取
            # if os.path.exists(file):
            #     f = h5py.File(file, 'r')['valid']
            #     point_data = np.concatenate((point_data, f['x_point'][()]))
            #     target_data = np.concatenate(target_data, f['x_target'][()])
            #     label = act_dic[act]
            #     y = y + [label for i in range(len(f['x_point'][()]))]
            # else:
            #     # 制作act的数据集并保存
            #     train_file, test_file = train_test_split([act], 'PAT', '../../har_data', 0.75)
            #     if dataset_type == 'train':
            #         x_p, x_t, y_label = self.createPATDatasetWithFileList_save_h5_v2(act,train_file)
            #         # HARDataset(data_path, 'PAT', activity_list, concat_framNum=args.concat_frame_num, seq_len=args.seq_len,
            #         #            file_list=train_files_point, pointLSTM=args.pointLSTM, dataset_type='train')
            #     else:
            #         x_p, x_t, y_label = self.createPATDatasetWithFileList_save_h5_v2(act,test_file)
            #     point_data = np.concatenate((point_data, x_p))
            #     target_data = np.concatenate(target_data, x_t)
            #     label = act_dic[act]
            #     y = y + [label for i in range(len(x_p))]

        return torch.tensor(point_data.tolist()), torch.tensor((target_data.tolist())), y#[i-1 for i in y]



def dataset_split(source_file):
    '''
    把原来各种状态和在一起的状态集拆分成按状态分别存储，读取的时候根据activity_list合并就行
    :param source_file:
    :param target_act:
    :return:
    '''
    if os.path.exists(source_file):
        dataset_info=os.path.basename(source_file).strip('.h5').split('_',3)
        dataset_type=dataset_info[0]
        seq_len=dataset_info[1]
        concat_frame_num=dataset_info[2]
        source_act_list=dataset_info[3].split('_')
        act_dic=dict(zip(source_act_list,range(len(source_act_list))))
        print(f'act dict is {act_dic}')
        f = h5py.File(source_file, 'r')
        if f['valid']:
            for act in source_act_list:
                label=act_dic[act]
                act_index=np.where(f['y'][()] == label+1)
                f_act = h5py.File(f'../har_dataset/{dataset_type}_{seq_len}_{concat_frame_num}_{act}.h5', 'w')
                f_act.create_dataset('x_point', data=f['x_point'][()][act_index])
                f_act.create_dataset('x_target', data=f['x_target'][()][act_index])
                f_act.create_dataset('y', data=[1 for i in range(len(act_index))])
                f_act.create_dataset('valid', data=1)
                f_act.close()
    else:
        print(f'Source file {source_file} does not exist.')



DEBUG=False
if __name__ == '__main__':
    DEBUG=True
    dataDir = "../har_data/"
    activity_list = ['stand']
    train_files, test_files = train_test_split(activity_list, 'point', dataDir, 0.75)
    # train_dataset = HARDataset(dataDir, 'point', activity_list, concat_framNum=3,seq_len=1, file_list=train_files, pointLSTM=False)
    print('!')

    # train_dataset = MyDataSet(dataDir, 'target', ['walk', 'sit', 'fall'], 10, train_files)
    # data_set = MyDataSet(dataDir, 'target', ['walk', 'sit', 'fall'],5)
    # data_loader = DataLoader(dataset=data_set, batch_size=200, shuffle=True, drop_last=False)

    # source_data='/home/mz/Human_activity/yj/har_dataset/train_20_3_stand_jump_sit_fall_run_walk_bend.h5'
    # dataset_split(source_data)

    test_dataset = HARDataset(dataDir, 'PAT', activity_list, concat_framNum=3,
                              seq_len=20,
                              file_list=test_files, pointLSTM=True, dataset_type='test',pc_channel=7)
