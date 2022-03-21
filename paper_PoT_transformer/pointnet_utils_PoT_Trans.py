import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)    # ??? iden 在做什么？？
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3) # ori  后边的stn好像只针对xyz来做的，所以这里还保留3*3
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True, channel=5,fea_dim=64,pn_fea_dim=256,stn3d=True):
        super(PointNetEncoder, self).__init__()
        self.do_stn3d=stn3d
        self.stn = STN3d(channel)
        if fea_dim==64:
            self.conv1 = torch.nn.Conv1d(channel, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.conv3 = torch.nn.Conv1d(128, 256, 1)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(256)
        else:
            self.conv1 = torch.nn.Conv1d(channel, fea_dim, 1)
            self.conv2 = torch.nn.Conv1d(fea_dim, 32, 1)
            # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.conv3 = torch.nn.Conv1d(32, pn_fea_dim, 1)
            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(32)
            self.bn3 = nn.BatchNorm1d(64)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        # if self.feature_transform:
        #     self.fstn = STNkd(k=64)
        if self.feature_transform:
            self.fstn = STNkd(k=fea_dim)
        self.pn_fea_dim=pn_fea_dim

    def forward(self, x):
        B, D, N = x.size()   # B: batch_size, D: fea_dim, N: num_point
        if self.do_stn3d:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)   # 应该是STN只作用于三维坐标，不对其他fea做变换的意思？所以上面stn里边不应该改成5*5，应该还保留3*3
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        else:
            trans=None
        x = F.relu(self.bn1(self.conv1(x)))  #self.conv1(x)=[256, 64, 48]

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # x.shape = torch.Size([batch_size, 1024, 160])
        x = torch.max(x, 2, keepdim=True)[0]   # 对称函数，maxpooling？
        # x.shape = torch.Size([batch_size, 1024, 1])
        # x = x.view(-1, 1024)
        x = x.view(-1, self.pn_fea_dim)
        #x.shape = torch.Size([batch_size, 1024])
        if self.global_feat:
            return x, trans, trans_feat
        else:   # 什么情况下不用global_feat??
            x = x.view(-1, self.pn_fea_dim, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat   # 将 point-level 和 global_fea 拼接到一起？


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss