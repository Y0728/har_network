import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import nn as nn



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
        batchsize = x.size()[0]   #x.shape torch.Size([512, 5, 48])
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
        batchsize = x.size()[0]   # x.shape=[512, 16, 48] (batch_size,embed_dim,pointnum)
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
        B, D, N = x.size()   # B: batch_size, D: fea_dim, N: num_point  [512, 5, 48])
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





class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    # def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
    def __init__(self, img_size=224, patch_size=48, seq_len=20,in_chans=5, embed_dim=64, norm_layer=None, flatten=True):

        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        # self.patch_size = patch_size
        self.patch_size = patch_size
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = seq_len
        # self.flatten = flatten
        #
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=in_chans,fea_dim=16,pn_fea_dim=embed_dim,stn3d=True)


    def forward(self, point_data):
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        # x = self.proj(x)
        # if self.flatten:
        #     x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)

        sample_fea = np.array([])
        # for i in range(seq_len):
        trans_fea_array = np.array([])
        point_data = point_data.transpose(1, 0)  # 从（batch_size, self.seq_len,attriNum)->(self.seq_len, batch_size, attriNum)
        for i, point_sample in enumerate(point_data):
            # Apply pointNet to each frame. point_sample.shape = (batch_size,attriNum, pointNum)
            point_fea, trans, trans_feat = self.feat(point_sample)  # point_fea.shape = batch_size,1024)
            if len(sample_fea) == 0:
                sample_fea = point_fea[np.newaxis, :, :]
                trans_fea_array = trans_feat[np.newaxis, :, :, :]
            else:
                sample_fea = torch.cat((sample_fea, point_fea[np.newaxis, :, :]),
                                       0)  # sample_fea = (seq_len, batch_size, 1024)
                trans_fea_array = torch.cat((trans_fea_array, trans_feat[np.newaxis, :, :, :]),
                                            0)  # trans_fea_array.shape = (seq_len, batch_size,64,64)
        sample_fea = sample_fea.transpose(0, 1)  # BNC?

        return sample_fea, trans_fea_array