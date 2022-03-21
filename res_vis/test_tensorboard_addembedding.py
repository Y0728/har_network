import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.pardir)
import argparse
import importlib
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import shutil
from torch.utils.data import Dataset, DataLoader
from har_dataset_train_test_split import train_test_split, HARDataset,dataset_exists
from tqdm import tqdm
from pathlib import Path
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

seqLen=20
concat_framNum=3
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=True, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size in training')
    parser.add_argument('--model', default='PointNet_mlp_Vit', help='model name [default: pointnet_cls]')
    parser.add_argument('--model_path', default='pointnet_mlp_transformer.log.classification.2022-02-21_11-52.', help='model name [default: pointnet_cls]')
    # parser.add_argument('--num_category', default=10, type=int, choices=[10, 40], help='training on ModelNet10/40')
    # parser.add_argument('--epoch', default=80, type=int, help='number of epoch in training')
    # parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    # parser.add_argument('--num_point', type=int, default=32 * concat_framNum, help='Point Number')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    # parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    # parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--seq_len', default=seqLen, help='default is 10')
    parser.add_argument('--activity_list', default=['stand','jump','sit','fall','run','walk'], help='activity types') #['fall', 'run']',,'walk','stand','fall','bend'
    parser.add_argument('--concat_frame_num', type=int, default=concat_framNum,
                        help='The number of frames that are concatenated together as one sample')
    parser.add_argument('--pointLSTM', default=True, help='Create dataset for lstm if it is true, default is False')
    # parser.add_argument('--rnn_type', default='rnn', help='The type of recurrent network')
    # parser.add_argument('--plot_result_in_tensorboard', default=True, help='Plot the acc and loss in tensorboard')
    return parser.parse_args()





def test(model, criterion,loader, num_class):

    mean_correct = []
    class_acc = np.zeros((num_class, 3))  # 这里声明了
    classifier = model.eval()

    all_preds = torch.tensor([])
    all_truth = torch.tensor([])
    total_correct=0
    # criterion = importlib.import_module(args.model).get_loss()  # 加，为了计算val_loss
    if not args.use_cpu:
        criterion = criterion.cuda()
    for j, (points, target, y) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target, y,all_preds,all_truth = points.cuda(), target.cuda(), y.cuda(),all_preds.cuda(),all_truth.cuda()
        if args.pointLSTM == False:
            points = points.transpose(2, 1)
        else:
            points = points.transpose(3, 2)
        # points = points.transpose(2, 1)
        # pred, _ = classifier(points)
        pred, trans_feat = classifier(points, target)
        pred_choice = pred.data.max(1)[1]

        #混淆矩阵
        all_preds = torch.cat(
            (all_preds, pred_choice)
            , dim=0
        )
        all_truth=torch.cat((
            all_truth,y
        ),dim=0)
        # 加，计算val_loss
        val_loss = criterion(pred, y.long(), trans_feat)

        for cat in np.unique(y.cpu()):
            # 表示在当前batch中 y为 cat的这类里边预测正确的个数，calassacc is the number of correct predicts of 类别为cat的 in this batch
            classacc = pred_choice[y == cat].eq(y[y == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[y == cat].size()[0])  # 这里分母为啥不直接用y为cat的样本的个数？
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(y.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        total_correct+=correct
    # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]    # ori 如果结果不含某一类，这里的分母就会是0，然后整体的acc就得不出来
    class_acc[np.where(class_acc[:, 1] != 0), 2] = class_acc[np.where(class_acc[:, 1] != 0), 0] / class_acc[
        np.where(class_acc[:, 1] != 0), 1]

    # instance_acc = np.mean(mean_correct)
    instance_acc=total_correct/len(loader.dataset)


    sns.set()
    f, ax = plt.subplots()
    cf = confusion_matrix(all_truth.cpu(), all_preds.cpu())
    # print(cf)  # 打印出来看看
    # sns.heatmap(C2, annot=True, ax=ax)  # 画热力图
    # plt.show()
    return instance_acc, class_acc, val_loss,cf,total_correct



def test_main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('test')
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
    # dir to save result visualization
    res_dir = exp_dir.joinpath('result/')
    res_dir.mkdir(exist_ok=True)


    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'test_res'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../../har_data'
    activity_list = args.activity_list
    _, test_files_point = train_test_split(activity_list, 'point', data_path, 0.75)
    log_string('test_files_point:%s' % test_files_point)

    activity_list=args.activity_list

    test_dataset = HARDataset(data_path, 'PAT', activity_list, concat_framNum=args.concat_frame_num, seq_len=args.seq_len,
                              file_list=test_files_point, pointLSTM=args.pointLSTM,dataset_type='test')
    testDataLoader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    '''MODEL LOADING'''
    num_class = len(activity_list)  # args.num_category
    model = importlib.import_module(args.model_path+args.model)
    shutil.copy('../%s.py' % (args.model_path.replace('.','/')+args.model), str(exp_dir))
    shutil.copy(f'./{os.path.basename(__file__)}', str(exp_dir))
    # classifier = model.PointNet_Vit(num_layers=2, hidden_size=2048, k=num_class, normal_channel=args.use_normals,
    #                                 model=args.rnn_type, seq_len=seqLen)
    classifier = model.PointNet_Vit( num_layers = 2, hidden_size = 2048, k = num_class, normal_channel=args.use_normals, seq_len = seqLen)


    # classifier = model.get_model( num_layers = 2, hidden_size = 2048, k = num_class, normal_channel=args.use_normals, model=args.rnn_type)
    criterion = model.get_loss()#nn.CrossEntropyLoss()
    # classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(args.model_path.replace('.','/')) + 'checkpoints/best_model.pth')
        # start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    with torch.no_grad():
        instance_acc, class_acc_array, val_loss,cf, total_correct = test(classifier.eval(), criterion,testDataLoader, num_class=num_class)
        class_acc = np.mean(class_acc_array[np.where(class_acc_array[:, 1] != 0), 2])
        print(instance_acc)
        print(class_acc_array)

    log_string(f'best_instance_acc_cf is {cf}')
    log_string(f'total_correct is:{total_correct}')
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(cf, annot=True, ax=ax, cmap='BuPu', fmt='d', xticklabels=activity_list,
                yticklabels=activity_list)  # 画热力图
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Prediction Value')
    ax.set_ylabel('Truth Value')
    plt.savefig('%s/confusion_matrix_count.png' % exp_dir)

    # percetage
    percetage = np.array(
        [[cf[i][j] / cf[i].sum() for j in range(len(cf[i]))] for i in
         range(len(cf))])
    f1, ax1 = plt.subplots()
    sns.heatmap(percetage, annot=True, ax=ax1, cmap='BuPu', fmt='.2f', xticklabels=activity_list,
                yticklabels=activity_list)  # 画热力图
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Prediction Value')
    ax1.set_ylabel('Truth Value')

if __name__ == '__main__':
    args = parse_args()
    test_main(args)