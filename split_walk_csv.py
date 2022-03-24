# 由于walk原本的csv数量少，将一个文件按照chunksize拆成多个，这样划分train_test以及制作数据集就不会有问题了
import os
import numpy as np
import pandas as pd
import glob

def split_dataset(file_dir, chunksize):
    '''
    Split walk raw csv data into small files
    :param file_dir: the dir saves walk csv files
    :param chunksize: the size of each splited csv file
    :return:
    '''
    # point_file_list = glob.glob(os.path.join(file_dir,'*point.csv'))
    target_file_list = glob.glob(os.path.join(file_dir,'*target.csv'))
    for file in target_file_list:
        c = 0
        no = 0
        df_target = pd.read_csv(file)
        frame_id = np.array(df_target['frame_id'])
        df_point = pd.read_csv(file.replace('target','point'))
        point_frame_id = np.array(df_point['frame_id'])
        while c < len(df_target):
            start = c
            start_frameNo = frame_id[start]
            while c < len(df_target) and frame_id[c] < start_frameNo+chunksize:
                c += 1

            df_target.iloc[start:c].to_csv(file.replace('.csv',f'-{no}.csv'),index=False)
            p_start = np.where(point_frame_id[:] == start_frameNo)[0][0]
            if c >= len(df_target):
                df_target.iloc[start:].to_csv(file.replace('.csv', f'-{no}.csv'), index=False)
                df_point.iloc[p_start:].to_csv(file.replace('target.csv',f'point-{no}.csv'),index=False)
                break
            end_frameNo = frame_id[c]
            p_end = p_start
            while p_end < len(point_frame_id) and point_frame_id[p_end] < end_frameNo:
                p_end += 1
            df_point.iloc[p_start:p_end].to_csv(file.replace('target.csv',f'point-{no}.csv'),index=False)
            no += 1
        # if start < len(df_target):
        #     df_target.iloc[start:].to_csv(file.replace('.csv',f'-{no}.csv'),index=False)
        #     df_point.iloc[p_end:].to_csv(file.replace('target.csv',f'point-{no}.csv'),index=False)


if __name__ == '__main__':
    file_dir = '/home/zlc/yj/HAR/har_data/walk/2021_12_07/raw'
    chunksize = 500
    split_dataset(file_dir,chunksize)

