import importlib
import torch
from thop import profile
# from pointnet_utils_PAT_pnmlp_Trans_tf_vit import PointNetEncoder, feature_transform_reguliarzer
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from point_transformer.helpers import build_model_with_cfg, named_apply, adapt_input_conv
# from point_transformer.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
# from point_transformer.registry import register_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_file='Target_SaT'
activity_list=['stand','jump','sit','fall','run','walk','bend']
num_class=len(activity_list)
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# thop
input_point=torch.randn(1,20,5,48).to(device)
input_target=torch.randn(1,20,9).to(device)
model = importlib.import_module(model_file).PointNet_Vit( num_classes = num_class, seq_len = 20 ,normal_channel=False,embed_dim=64).to(device)

# print(get_parameter_number(model))
flops, params = profile(model, inputs=(input_target,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))
#
# # from torchsummary import summary
# # from tensorboardX import SummaryWriter
# # summary(model, [(20,5,48),(20,9)])
# # with SummaryWriter(comment='Net') as w:
# #     w.add_graph(Net, (dummy_input01, dummy_input02))
#
# # # torchstat
# # from torchstat import stat
# #
# # model = importlib.import_module(model_file).PointNet_Vit( num_classes = num_class, seq_len = 20 ,normal_channel=False).to(device)
# # stat(model,((20,5,48),(20,9)))
#
# # # torchsummary
# # import torchsummary as summary
# # model = importlib.import_module(model_file).PointNet_Vit( num_classes = num_class, seq_len = 20 ,normal_channel=False).to(device)
# # summary.summary(model,[(20,5,48),(20,9)])
#
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np
# # from pytorch_model_summary import summary
# # model = importlib.import_module(model_file).PointNet_Vit( num_classes = num_class, seq_len = 20 ,normal_channel=False).to(device)
# # print(summary(model,torch.zeros((1, 20, 5, 48)),torch.zeros((1,20,9)), show_input=False, show_hierarchical=True))
#
#
# # from fvcore.nn import FlopCountAnalysis
# # flops=FlopCountAnalysis(RNN, torch.randn(1,28,28))
# # flops.total()
# from flopth import flopth
# # sum_flops=flopth(model, in_size=[[10], [10]])
# sum_flops=flopth(model,[[20,9]])
#
# print(sum_flops)



# from flopth import flopth
# import torch
# import torch.nn as nn
# from ptflops import get_model_complexity_info
#
#
# class Siamese(nn.Module):
#     def __init__(self):
#         super(Siamese, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, 3, 1)
#         self.conv2 = nn.Conv2d(1, 10, 3, 1)
#
#     def forward(self, x):
#         # assume x is a list
#         return self.conv1(x[0]) + self.conv2(x[1])

# def prepare_input(resolution):
#     x1 = torch.FloatTensor(1, *resolution)
#     x2 = torch.FloatTensor(1, *resolution)
#     return dict(x = [x1, x2])
#
# if __name__ == '__main__':
#     model = Siamese()
#     flops, params = get_model_complexity_info(model, input_res=(1, 224, 224),
#                                               input_constructor=prepare_input,
#                                               as_strings=True, print_per_layer_stat=False)
#     print('      - Flops:  ' + flops)
#     print('      - Params: ' + params)