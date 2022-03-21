""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from point_transformer.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from point_transformer.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from point_transformer.registry import register_model

from pointnet_utils_PAT import PointNetEncoder, feature_transform_reguliarzer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

_logger = logging.getLogger(__name__)




class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # loss = nn.CrossEntropyLoss()(pred, target)
        # loss = nn.NLLLoss()(pred, target)
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale

        # mat_diff_loss = 0
        # for i, trans_feat in enumerate(trans_feat_array):
        #     # print(trans_feat_array)
        #     # print(f"the size of trans_feat_array {trans_feat_array.shape}")  #10, 512, 64, 64]
        #
        #     mat_diff_loss += feature_transform_reguliarzer(trans_feat)  #trans_feat.shape = (batch_size,64,64)
        #
        # total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss



class PointNet_Target(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, num_classes,normal_channel = True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 5  # 3
        self.num_classes = num_classes

        # self.embed_target = nn.Sequential(
        #         nn.Linear(9, 16),
        #         # nn.BatchNorm1d(16),
        #         nn.ReLU(),
        #         nn.Linear(16, 64),
        #         # nn.BatchNorm1d(64)
        #         # nn.ReLU(),
        #         # nn.Linear(128, 256),
        #     )
        self.embed_target = nn.Sequential(
            nn.Conv1d(9, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32,1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            # nn.BatchNorm1d(64)
            # nn.ReLU(),
            # nn.Linear(128, 256),
        )

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        #原
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, num_classes)
        # self.dropout = nn.Dropout(p=0.4)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.relu = nn.ReLU()
        #改
        # self.fc1 = nn.Linear(64, 32)
        # self.fc2 = nn.Linear(32, 16)
        # self.fc3 = nn.Linear(16, num_classes)
        # self.dropout = nn.Dropout(p=0.4)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(16)
        # self.relu = nn.ReLU()

        # self.fc1 = nn.Linear(64+9, 32)
        # self.fc2 = nn.Linear(32, 16)
        # self.fc3 = nn.Linear(16, num_classes)
        # self.dropout = nn.Dropout(p=0.4)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(16)
        # self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()


    def forward(self, point_data,target_data):
        point_fea, trans, trans_feat = self.feat(point_data)
        x=point_fea
        target_data=target_data[:,:,np.newaxis]
        embeded_target_data = self.embed_target(target_data)
        embeded_target_data=embeded_target_data.squeeze()
        # x=point_fea+embeded_target_data
        # x = torch.cat((point_fea,embeded_target_data), 1).to(device)
        x = torch.cat((embeded_target_data, point_fea), 1).to(device)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


