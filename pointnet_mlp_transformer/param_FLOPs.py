import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import importlib
import torch
from thop import profile
import torch.nn as nn

from pointnet_utils_PAT_pnmlp_Trans_tf_vit import PointNetEncoder, STN3d, STNkd
from PointNet_mlp_Vit import Attention
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from point_transformer.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from point_transformer.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from point_transformer.registry import register_model

device ='cpu'# torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_file='PointNet_mlp_Vit'
activity_list=['stand','jump','sit','fall','run','walk','bend']
num_class=len(activity_list)
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# thop
input_point=torch.randn(1,20,5,48).to(device)
input_target=torch.randn(1,20,9).to(device)
modelFile=importlib.import_module(model_file)
model = modelFile.PointNet_Vit( num_classes = num_class, seq_len = 20 ,normal_channel=False).to(device)
print(get_parameter_number(model))


flops, params = profile(model, inputs=((input_point),(input_target),))
print('model\'s flops:{}'.format(flops))
print('model\'s params:{}'.format(params))

# model_atten=modelFile.Attention(128, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0)
# att_input=
# flops, params = profile(model_atten, inputs=((input_point),(input_target),))
# print('model\'s flops:{}'.format(flops))
# print('model\'s params:{}'.format(params))


model_PointNetEncoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=5, fea_dim=16, pn_fea_dim=64,
                            stn3d=True)
PointnetEncoder_input=torch.randn(1,5,48)
flops_pointnet, params_pointnet = profile(model_PointNetEncoder, inputs=((PointnetEncoder_input),))

print('PointNetEncoder\'s flops:{}'.format(flops_pointnet))
print('PointNetEncoder\'s params:{}'.format(params_pointnet))

model_stn3d = STN3d(5)
stn3d_input=torch.randn(1,5,48)
flops_stn3d, params_stn3d = profile(model_stn3d, inputs=((stn3d_input),))
print('stn3d\'s flops:{}'.format(flops_stn3d))
print('stn3d\'s params:{}'.format(params_stn3d))


model_stnkd = STNkd(16)
stnkd_input=torch.randn(1,16,48)
flops_stnkd, params_stnkd = profile(model_stnkd, inputs=((stnkd_input),))
print('stnkd\'s flops:{}'.format(flops_stnkd))
print('stnkd\'s params:{}'.format(params_stnkd))

dim=128
mlp_ratio=4
mlp_hidden_dim = int(dim * mlp_ratio)
model_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
mlp_input=torch.randn(1,21,128)
flops_mlp, params_mlp = profile(model_mlp, inputs=((mlp_input),))
print('mlp\'s flops:{}'.format(flops_mlp))
print('mlp\'s params:{}'.format(params_mlp))

num_heads=8
att_model=Attention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=0, proj_drop=0)
att_input=torch.randn(1,21,128)
flops_att, params_att = profile(att_model, inputs=((att_input),))
print('mlp\'s flops:{}'.format(flops_att))
print('mlp\'s params:{}'.format(params_att))
# from torchsummary import summary
# from tensorboardX import SummaryWriter
# summary(model, [(20,5,48),(20,9)])
# with SummaryWriter(comment='Net') as w:
#     w.add_graph(Net, (dummy_input01, dummy_input02))

# # torchstat
# from torchstat import stat
#
# model = importlib.import_module(model_file).PointNet_Vit( num_classes = num_class, seq_len = 20 ,normal_channel=False).to(device)
# stat(model,((20,5,48),(20,9)))

# # torchsummary
# import torchsummary as summary
# model = importlib.import_module(model_file).PointNet_Vit( num_classes = num_class, seq_len = 20 ,normal_channel=False).to(device)
# summary.summary(model,[(20,5,48),(20,9)])

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from pytorch_model_summary import summary
# model = importlib.import_module(model_file).PointNet_Vit( num_classes = num_class, seq_len = 20 ,normal_channel=False).to(device)
# print(summary(model,torch.zeros((1, 20, 5, 48)),torch.zeros((1,20,9)), show_input=False, show_hierarchical=True))