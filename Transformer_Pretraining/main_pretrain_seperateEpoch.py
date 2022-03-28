"""
Author: Benny
Date: Nov 2019
"""
# mae的timm需要0.3.2版本，否则报错
'''
Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
TypeError: __init__() got an unexpected keyword argument 'qk_scale'
'''
# 但timm=0.3.2的话又会报错
'''
from torch._six import container_abcs
ImportError: cannot import name 'container_abcs' from 'torch._six' (/home/zlc/.conda/envs/har-network/lib/python3.8/site-packages/torch/_six.py)

'''
import os
from typing import Iterable

import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append(os.pardir)
import torch
import numpy as np

print(sys.path)
import datetime
import logging
from pointnet import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
# from data_utils.ModelNetDataLoader import ModelNetDataLoader
from torch.utils.data import Dataset, DataLoader
from har_dataset_train_test_split_pretrain import train_test_split, HARDataset_Pretrain,dataset_exists
import random

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import math
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import time
import util.misc as misc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
seqLen = 20
concat_framNum = 3

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--model', default='models_mae', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=8, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=80, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=32 * concat_framNum, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--seq_len', default=seqLen, help='default is 10')
    parser.add_argument('--activity_list', default=['6'], help='activity types') #'6','7','8','9','10'['fall', 'run']','stand','jump','sit','fall','run','walk','bend'#'walk','fall','stand','sit'
    parser.add_argument('--concat_frame_num', type=int, default=concat_framNum,
                        help='The number of frames that are concatenated together as one sample')
    parser.add_argument('--pointLSTM', default=True, help='Create dataset for lstm if it is true, default is False')
    # parser.add_argument('--rnn_type', default='rnn', help='The type of recurrent network')
    parser.add_argument('--plot_result_in_tensorboard', default=True, help='Plot the acc and loss in tensorboard')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--channels', default=7, help='The number of attributes of points.')
    parser.add_argument('--pretrain',default=True,help='Whether to load pretrain dataset')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True



def train_one_epoch(classifier: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,scheduler,
                    log_writer=None,
                    args=None,logger=None):
    classifier = classifier.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 150
    scheduler.step()
    accum_iter = 1 # 待修改？原来是args.accum_iter
    optimizer.zero_grad()


    # for batch_id, (points, target, y) in tqdm(enumerate(data_loader, 0), total=len(data_loader),
    #                                           smoothing=0.9):
    for batch_id, dataset in enumerate(metric_logger.log_every(data_loader, print_freq, header,logger)):

        points = dataset[0]
        target = dataset[1]

        # optimizer.zero_grad()   # pre-training改到后面了

        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        if args.pointLSTM == False:
            points = points.transpose(2, 1)
        else:
            points = points.transpose(3, 2)

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        with torch.cuda.amp.autocast():
            loss, pred, mask, trans_feat_array = classifier(points, target, 0.75)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=classifier.parameters(),
                    update_grad=(batch_id + 1) % accum_iter == 0)

        if (batch_id + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        # print(f'lr is :{lr}' )
        # print(f'schedular.get_lr() is :{scheduler.get_last_lr()}')   #不知道为啥，每到该更新的那个epoch的时候，这个get_last_lr的值总是会多乘一个0.7，过了这个epoch就有和lr = optimizer.param_groups[0]["lr"]一样了
        metric_logger.update(lr=lr)
        # loss_value_reduce = loss_value  # misc.all_reduce_mean(loss_value)
        loss_value_reduce=misc.all_reduce_mean(loss_value)
        if args.plot_result_in_tensorboard:
            epoch_1000x = int((batch_id / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args = parse_args()
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
    args.output_dir=checkpoints_dir
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    # dir to save result visualization
    res_dir = exp_dir.joinpath('result/')
    res_dir.mkdir(exist_ok=True)

    '''LOG'''

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
    # data_path = '../../har_data'
    data_path = '../../har_pretrain_dataset'
    activity_list = args.activity_list
    train_files_point, test_files_point = train_test_split(activity_list, 'point', data_path, 0.75)
    # train_files_target = [p.replace('point','target') for p in train_files_point]
    # test_files_target = [p.replace('point','target') for p in test_files_point]
    # log_string('train_files_point:%s' % train_files_point)
    # log_string('test_files_point:%s' % test_files_point)
    # data_set = MyDataSet(dataDir, 'target', ['walk', 'sit', 'fall'])
    # data_loader = DataLoad/er(dataset=data_set, batch_size=batch_size, shuffle=True, drop_last=False)

    if file_exists:=dataset_exists(activity_list,args.seq_len,args.concat_frame_num,'train'):
        log_string(f'{file_exists} was founded.')
        act_order=os.path.basename(file_exists).strip('.h5').split('_', 3)[3].split('_')
        log_string(f'The order of act is:{act_order}')
    else:
        log_string(f'{file_exists} was not founded.')
        act_order=activity_list
        log_string(f'The order of act is:{activity_list}')

    log_writer = SummaryWriter(log_dir=res_dir)
    if not args.pretrain:
        train_dataset = HARDataset_Pretrain( 'PAT', activity_list, concat_framNum=args.concat_frame_num,seq_len = args.seq_len,
                                   file_list=train_files_point, pointLSTM=args.pointLSTM,dataset_type='train')
        trainDataLoader= DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # test_dataset = HARDataset_Pretrain( 'PAT', activity_list, concat_framNum=args.concat_frame_num, seq_len=args.seq_len,
    #                           file_list=test_files_point, pointLSTM=args.pointLSTM,dataset_type='test')
    # testDataLoader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    else:
        train_dataset = HARDataset_Pretrain( 'pretrain', activity_list, concat_framNum=args.concat_frame_num,
                                   seq_len=args.seq_len,
                                   file_list=None, pointLSTM=args.pointLSTM, dataset_type='train')
        trainDataLoader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # test_dataset = HARDataset_Pretrain( 'PAT', activity_list, concat_framNum=args.concat_frame_num,
    #                           seq_len=args.seq_len,
    #                           file_list=test_files_point, pointLSTM=args.pointLSTM, dataset_type='test')
    # testDataLoader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    '''MODEL LOADING'''
    num_class = len(activity_list)#args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./%s.py' % args.model, str(exp_dir))
    # shutil.copy('pointnet2_utils.py', str(exp_dir))
    shutil.copy(f'./{os.path.basename(__file__)}', str(exp_dir))
    ## 0037
    classifier = model.MaskedAutoencoderViT(embed_dim=128, decoder_embed_dim=16, in_chans=args.channels)
    # classifier = model.PointNet_Vit(  num_classes= num_class, normal_channel=args.use_normals,seq_len = seqLen)

    # classifier = model.get_model( num_layers = 2, hidden_size = 2048, k = num_class, normal_channel=args.use_normals, model=args.rnn_type)
    # criterion = model.get_loss()
    classifier.apply(inplace_relu)

    model_without_ddp = classifier
    logger.info("Model = %s" % str(model_without_ddp))

    if not args.use_cpu:
        classifier = classifier.cuda()
        # criterion = criterion.cuda()

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
    logger.info(optimizer)
    loss_scaler = NativeScaler() # pre-training 加
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    global_epoch = 0

    '''TRANING'''
    logger.info('Start training...')
    logger.info(f"Start training for {args.epoch} epochs")
    start_time = time.time()
    if log_writer is not None:
        logger.info('log_dir: {}'.format(log_writer.log_dir))
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []


        train_stats = train_one_epoch(
            classifier, trainDataLoader,
            optimizer, device, epoch, loss_scaler,scheduler,
            log_writer=log_writer,
            args=args,
            logger=logger
        )

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epoch):
            misc.save_model(
                args=args, model=classifier, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }


        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == '__main__':
    args = parse_args()
    main(args)
