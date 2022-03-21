import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils_pn_woFeaTrans_tf_vit import PointNetEncoder, feature_transform_reguliarzer
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
class get_model(nn.Module):
    def __init__(self, num_layers, hidden_size, k=5, normal_channel=True, model='rnn'):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 5  # 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.t_input_size = 9
        self.t_hidden_size = 256
        self.t_num_layers = 3
        self.model = model
        if self.model == 'rnn':
            self.rnn_target = nn.RNN(self.t_input_size, self.t_hidden_size, self.t_num_layers, batch_first=True)
            self.rnn = nn.RNN(self.t_hidden_size+256, hidden_size, num_layers, batch_first=True)
        elif self.model == 'gru':
            self.rnn_target = nn.GRU(self.t_input_size, self.t_hidden_size, self.t_num_layers, batch_first=True)
            self.rnn = nn.GRU(self.t_hidden_size + 256, hidden_size, num_layers, batch_first=True)
        elif self.model == 'lstm':
            self.rnn_target = nn.LSTM(self.t_input_size, self.t_hidden_size, self.t_num_layers, batch_first=True)
            self.rnn = nn.LSTM(self.t_hidden_size + 256, hidden_size, num_layers, batch_first=True)

    def forward(self, point_data, target_data):
        sample_fea = np.array([])
        # for i in range(seq_len):
        trans_fea_array = np.array([])
        point_data = point_data.transpose(1,0) #从（batch_size, self.seq_len,attriNum)->(self.seq_len, batch_size, attriNum)
        for i, point_sample in enumerate(point_data):
            # Apply pointNet to each frame. point_sample.shape = (batch_size,attriNum, pointNum)
            point_fea, trans, trans_feat = self.feat(point_sample)  # point_fea.shape = batch_size,1024)
            if len(sample_fea) == 0:
                sample_fea = point_fea[np.newaxis, :, :]
                trans_fea_array = trans_feat[np.newaxis, :, :, :]
            else:
                sample_fea = torch.cat((sample_fea, point_fea[np.newaxis, :, :]),0)  #sample_fea = (seq_len, batch_size, 1024)
                trans_fea_array = torch.cat((trans_fea_array, trans_feat[np.newaxis, :, :, :]),0)   #trans_fea_array.shape = (seq_len, batch_size,64,64)
        sample_fea = sample_fea.transpose(0,1)
        t_h0 = torch.zeros(self.t_num_layers, target_data.size(0), self.t_hidden_size).to(device)
        if self.model == 'lstm':
            t_c0 = torch.zeros(self.t_num_layers, target_data.size(0), self.t_hidden_size).to(device)
            target_data, _ = self.rnn_target(target_data, (t_h0,t_c0))
        else:
            target_data, _ = self.rnn_target(target_data, t_h0)
        input = torch.cat((target_data, sample_fea),2).to(device)

        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        if self.model == 'lstm':
            c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
            out, _ = self.rnn(input, (h0,c0))
        else:
            out, _ = self.rnn(input, h0)
        out = out[:, -1, :]
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)
        return out,trans_fea_array   #如果feature_transform是False的话， trans_feat = None?


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat_array):
        loss = nn.NLLLoss()(pred, target)
        # loss = F.nll_loss(pred, target)
        mat_diff_loss = 0
        for i, trans_feat in enumerate(trans_feat_array):
            mat_diff_loss += feature_transform_reguliarzer(trans_feat)  #trans_feat.shape = (batch_size,64,64)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
