import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn.pytorch import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder

from src.NF import MAF, RealNVP
from src.dlutils import *
from src.constants import *
torch.manual_seed(1)

class GDN(nn.Module):
    def __init__(self, feats):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.batch = 128
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats

        self.feature_gat = GATConv(1, 1, feats)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
        )
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
        )

    def forward(self, data):
        B=1
        att_score = self.attention(data).view(-1, self.n_window, 1)
        data = data.view(-1, self.n_feats, self.n_window)
        data_r = torch.matmul(data, att_score)
        # GAT convolution on complete graph
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        src_ids = np.repeat(np.array(list(range(self.n_feats))), self.n_feats)
        dst_ids = np.array(list(range(self.n_feats))*self.n_feats)

        g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
        g = dgl.add_self_loop(g)
        g = g.to(device)
        g_list = []
        for i in range(B):
            g.ndata['feat'] = data_r[i]
            g_list.append(g)
        g = dgl.batch(g_list)

        feat_r = self.feature_gat(g, g.ndata['feat'])
        feat_r = feat_r.view(B, self.n_feats, self.n_feats)
        x = self.fcn(feat_r)
        return x.view(B, -1)


import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer


class TranAD(nn.Module):
    """TranAD model for anomaly detection using Transformer architecture."""

    def __init__(self, feats, batch=128, wd=10):

        super().__init__()

        self.name = 'TranAD'
        self.batch_size = batch
        self.n_feats = feats
        self.n_window = wd
        self.n = self.n_feats * self.n_window

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=2 * feats,
            dropout=0.1,
            max_len=self.n_window
        )

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # Transformer decoders
        decoder_layers_1 = TransformerDecoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers_1, num_layers=1)

        decoder_layers_2 = TransformerDecoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers_2, num_layers=1)

        # Output fully connected layer
        self.fcn = nn.Sequential(
            nn.Linear(2 * feats, feats),
            nn.Sigmoid()
        )

    def encode(self, src, context, tgt):

        # Concatenate source and context
        src_combined = torch.cat((src, context), dim=2)
        src_combined = src_combined * math.sqrt(self.n_feats)
        src_combined = self.pos_encoder(src_combined)

        # Encode through transformer
        memory = self.transformer_encoder(src_combined)

        # Prepare target
        tgt_processed = tgt.repeat(1, 1, 2)

        return tgt_processed, memory

    def forward(self, src, tgt):
        context_zeros = torch.zeros_like(src)
        tgt_processed, memory = self.encode(src, context_zeros, tgt)
        x1 = self.transformer_decoder1(tgt_processed, memory)
        x1 = self.fcn(x1)

        reconstruction_error = (x1 - src) ** 2
        tgt_processed, memory = self.encode(src, reconstruction_error, tgt)
        x2 = self.transformer_decoder2(tgt_processed, memory)
        x2 = self.fcn(x2)

        return x1, x2
class MoEGDN(nn.Module):
    def __init__(self, feats):
        super(MoEGDN, self).__init__()
        self.name = 'MoEGDN'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 5
        self.n = self.n_feats * self.n_window
        self.n_classes = len(feats)
        self.main_model = nn.ModuleList([GDN(self.n_feats[i]) for i in range(self.n_classes)])

    def forward(self, data):
        outlist = []
        lb = 0
        ub = 0
        for i in range(len(self.main_model)):
            lb = lb+self.n_feats[i-1] if i > 0 else 0
            ub = ub+self.n_feats[i]
            _data = data[:, int(lb*self.n_window):int(ub*self.n_window)]
            x = self.main_model[i](_data)
            outlist.append(x)
        output = torch.cat(outlist, dim=1)
        return output

class MoETransAD(nn.Module):
    def __init__(self, feats, batch = 128, wd = 10):
        super(MoETransAD, self).__init__()
        self.name = 'MoETransAD'
        self.router = 'MAF'
        self.lr = lr
        self.batch = batch
        self.n_feats = feats
        self.alpha = 0.01
        self.n_window = wd
        self.n = self.n_feats * self.n_window
        self.n_classes = len(feats)
        self.main_model = nn.ModuleList([TranAD(self.n_feats[i], batch=batch, wd=wd) for i in range(self.n_classes)])
        if self.router=="MAF":
            self.route = MAF(6,  self.n_feats.sum()*self.n_window, 32, 1, cond_label_size=self.n_feats.sum(), batch_norm=self.batch,activation='tanh')
        else:
            self.route = RealNVP(6, self.n_feats.sum(), 32, 1, cond_label_size=32, batch_norm=self.batch)

    def forward(self, src, tgt):
        batch = src.shape[1]
        out_list1 = []
        out_list2 = []
        lb = 0
        ub = 0
        # if self.training:
        #     t_src = src.transpose(0, 1).reshape(batch, -1)
        #     log_prob = self.route.log_prob(t_src, tgt.squeeze())
        #     src = self.topk_mask(src, log_prob)
        for i in range(len(self.main_model)):
            lb = lb+self.n_feats[i-1] if i > 0 else 0
            ub = ub+self.n_feats[i]
            _src, _tgt = src[:, :, lb:ub], tgt[:, :, lb:ub]
            x1, x2 = self.main_model[i](_src, _tgt)
            out_list1.append(x1)
            out_list2.append(x2)
        t1 = torch.cat(out_list1, dim=2)
        t2 = torch.cat(out_list2, dim=2)

        return t1, t2

    def topk_mask(self,
            input_tensor: torch.Tensor,
            mask_tensor: torch.Tensor,
            k=3,
    ) -> torch.Tensor:


        x_dim, len_dim, y_dim = input_tensor.shape

        _, topk_indices = torch.topk(mask_tensor, k, largest=True)

        bool_mask = torch.zeros(len_dim, dtype=torch.bool, device=input_tensor.device)
        bool_mask[topk_indices] = True

        expanded_mask = bool_mask.view(1, len_dim, 1).expand( x_dim, len_dim, y_dim)
        input_tensor[expanded_mask] = 0
        return input_tensor

class MoEGNN(nn.Module):
    def __init__(self, feats):
        super(MoEGNN, self).__init__()
        self.name = 'MoETransAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.n_classes = len(feats)
        self.main_model = nn.ModuleList([TranAD(self.n_feats[i]) for i in range(self.n_classes)])



    def forward(self, src, tgt):
        out_list1 = []
        out_list2 = []
        lb = 0
        ub = 0
        for i in range(len(self.main_model)):
            lb = lb+self.n_feats[i-1] if i > 0 else 0
            ub = ub+self.n_feats[i]
            _src, _tgt = src[:, :, lb:ub], tgt[:, :, lb:ub]
            x1, x2 = self.main_model[i](_src, _tgt)
            out_list1.append(x1)
            out_list2.append(x2)
        t1 = torch.cat(out_list1, dim=2)
        t2 = torch.cat(out_list2, dim=2)
        return t1, t2

    def compute_cosine_similarity_matrix(self, feature, threshold = 0.5):

        dot_product = torch.bmm(feature, torch.transpose(feature, 1, 2))
        norm_squared = torch.sum(feature ** 2, dim=2, keepdim=True)
        denominator = torch.sqrt(torch.bmm(norm_squared, norm_squared.transpose(1, 2)))
        denominator[denominator == 0] = 1
        dot_product[dot_product == 0] = -1
        cosine_similarity = dot_product / denominator
        binarized_tensor = (cosine_similarity > threshold).int()
        return binarized_tensor

    def DFT(self, x):
        mean = x.mean()
        std = x.std()
        norm_x = (x - mean) / std
        fft_whole = torch.fft.fft(norm_x, dim=-1, norm='forward')
        real_part = fft_whole.real
        imag_part = fft_whole.imag

        merge_fft = torch.sqrt(real_part ** 2 + imag_part ** 2)
        return merge_fft


class Expert(nn.Module):
    def __init__(self, input_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(32, input_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Gating(nn.Module):
    def __init__(self, input_dim,
                 num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

 # Layers
        self.layer1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(64, 32)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(32, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        return torch.softmax(self.layer4(x), dim=2)


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MoE, self).__init__()
        self.name = 'MoE'
        self.batch = 64
        input_dim = input_dim[0]
        self.lr = lr
        self.n_window = 10
        self.num_experts = 7
        self.backbone = TranAD(input_dim)
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(self.num_experts)])
        self.gating_network = Gating(self.n_window, self.num_experts)

    def forward(self, x, tgt):
        # Compute gating weights
        gating_weights = self.gating_network(torch.transpose(x, 0, 2))
        gating_weights = torch.transpose(gating_weights, 0, 2)
        x1, x2 = self.backbone(x, tgt)
        # Compute expert outputs
        expert_outputs1 = torch.stack([expert(x1) for expert in self.experts], dim=1)
        expert_outputs2 = torch.stack([expert(x2) for expert in self.experts], dim=1)
        # Combine expert outputs using gating weights
        x1 = torch.sum(gating_weights.unsqueeze(0) * expert_outputs1, dim=1)
        x2 = torch.sum(gating_weights.unsqueeze(0) * expert_outputs2, dim=1)
        return x1, x2


class MoE2(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MoE2, self).__init__()
        self.name = 'MoE2'
        self.batch = 64
        input_dim = input_dim
        self.lr = lr
        self.n_window = 10
        self.num_experts = 6
        self.topk = int(input_dim/self.num_experts*1.2)
        self.backbone = TranAD(input_dim)
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(self.num_experts)])
        self.gating_network = Gating(self.n_window, self.num_experts)

    def forward(self, x, tgt):
        n_window, batch_size, input_dim = x.shape

        gating_weights = self.gating_network(torch.transpose(x, 0, 2))  # (batch_size, num_tokens, num_experts)
        gating_weights = torch.transpose(gating_weights, 0, 2)

        expert_outputs = torch.zeros(batch_size, n_window, input_dim, device=x.device)

        for expert_idx in range(self.num_experts):

            expert_gating_weights = gating_weights[:, :, expert_idx]  # (batch_size, num_tokens)

            topk_weights, topk_indices = torch.topk(expert_gating_weights, k=self.topk, dim=1)  # (batch_size, topk)

            topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)  # (batch_size, topk)

            for i in range(batch_size):
                selected_tokens = x[i, topk_indices[i]]

                expert_output = self.experts[expert_idx](selected_tokens)
                expert_outputs[i, topk_indices[i]] += topk_weights[i].unsqueeze(-1) * expert_output

        final_output = expert_outputs.sum(dim=1)
        return final_output
