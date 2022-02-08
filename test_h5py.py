from har_dataset_train_test_split import train_test_split, HARDataset
import h5py
import numpy as np
data_path = '../har_data'
activity_list = ['jump']
concat_frame_num = 3
seq_len = 10
train_files_point, test_files_point = train_test_split(activity_list, 'point', data_path, 0.75)

train_dataset = HARDataset(data_path, 'PAT', activity_list, concat_framNum=concat_frame_num, seq_len=seq_len,
                           file_list=train_files_point, pointLSTM=True,dataset_type='train')


# X= np.random.rand(100, 1000, 1000).astype('float32')
# y = np.random.rand(1, 1000, 1000).astype('float32')

# Create a new file
# f = h5py.File(f'train_{seq_len}_{concat_frame_num}_jump.h5', 'w')
# f.create_dataset('train', data=train_dataset)
# # f.create_dataset('y_train', data=y)
# f.close()
#
# # Load hdf5 dataset
# f = h5py.File('data.h5', 'r')
# X = f['X_train']
# print(X == train_dataset)
# # Y = f['y_train']
# f.close()