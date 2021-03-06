"""
Author: Benny
Date: Nov 2019
"""

import os  #注意！！需要放到最前面，这两句放到torch后就不起作用。。
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
import sys
import torch
import numpy as np

print(sys.path)
import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
# from data_utils.ModelNetDataLoader import ModelNetDataLoader
from torch.utils.data import Dataset, DataLoader
from har_dataset_train_test_split import train_test_split, HARDataset
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
seqLen = 1#40
concat_framNum = 3 #1


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=16 * concat_framNum, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--seq_len', default=seqLen, help='default is 10')
    parser.add_argument('--activity_list', default=['stand','jump','sit','fall','run','walk','bend'], help='activity types')  #['walk','sit','fall','run','bend','jump','stand']
    parser.add_argument('--concat_frame_num', type=int, default=concat_framNum,
                        help='The number of frames that are concatenated together as one sample')
    parser.add_argument('--pointLSTM', default=False, help='Create dataset for lstm if it is true, default is False')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))  # 这里声明了
    classifier = model.eval()

    for j, (points,target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            # 表示在当前batch中 target为 cat的这类里边预测正确的个数，calassacc is the number of correct predicts of 类别为cat的 in this batch
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])  # 这里分母为啥不直接用target为cat的样本的个数？
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]    # ori 如果结果不含某一类，这里的分母就会是0，然后整体的acc就得不出来
    class_acc[np.where(class_acc[:, 1] != 0), 2] = class_acc[np.where(class_acc[:, 1] != 0), 0] / class_acc[
        np.where(class_acc[:, 1] != 0), 1]
    # class_acc = np.mean(class_acc[:, 2])  # ori
    class_acc = np.mean(class_acc[np.where(class_acc[:, 1] != 0), 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../../har_data'
    activity_list = args.activity_list
    train_files, test_files = train_test_split(activity_list, 'point', data_path, 0.75)
    log_string('train_files:%s' % train_files)
    log_string('test_files:%s' % test_files)
    # data_set = MyDataSet(dataDir, 'target', ['walk', 'sit', 'fall'])
    # data_loader = DataLoad/er(dataset=data_set, batch_size=batch_size, shuffle=True, drop_last=False)

    train_dataset = HARDataset(data_path, 'point', activity_list, concat_framNum=args.concat_frame_num,
                               file_list=train_files, pointLSTM=args.pointLSTM)
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataset = HARDataset(data_path, 'point', activity_list, concat_framNum=args.concat_frame_num,
                              file_list=test_files, pointLSTM=args.pointLSTM)
    testDataLoader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # train_dataset = HARDataset(data_path, 'PAT', activity_list, concat_framNum=args.concat_frame_num,
    #                            seq_len=args.seq_len,
    #                            file_list=train_files, pointLSTM=args.pointLSTM, dataset_type='train')
    # trainDataLoader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # test_dataset = HARDataset(data_path, 'PAT', activity_list, concat_framNum=args.concat_frame_num,
    #                           seq_len=args.seq_len,
    #                           file_list=test_files, pointLSTM=args.pointLSTM, dataset_type='test')
    # testDataLoader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = len(args.activity_list)#args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./%s.py' % args.model, str(exp_dir))
    # shutil.copy('pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification_pointnet.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                               smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
