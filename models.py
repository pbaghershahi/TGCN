import argparse, time, torch, logging, os
import numpy as np, torch.nn as nn, torch.nn.functional as F
from torch_scatter import scatter


class TGCNLayer(nn.Module):
    def __init__(
            self,
            in_feat,
            out_feat,
            n_bases=-1,
            bias=True,
            activation=None,
            self_loop=True,
            normalize_output=False,
            cp_decompose=False,
            dropout_rates={
                'dr_input': 0.2,
                'dr_hid1': 0.3,
                'dr_hid2': 0.2,
                'dr_output': 0.2
            },
            split_rate=30000
    ):
        super(TGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.n_bases = n_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.normalize_output = normalize_output
        self.cp_decompose = cp_decompose
        self.split_rate = split_rate
        self.input_dropout = nn.Dropout(dropout_rates['dr_input'])
        self.hidden_dropout1 = nn.Dropout(dropout_rates['dr_hid1'])
        self.hidden_dropout2 = nn.Dropout(dropout_rates['dr_hid2'])
        self.output_dropout = nn.Dropout(dropout_rates['dr_output'])
        self.bn0 = nn.BatchNorm1d(in_feat)
        self.bn1 = nn.BatchNorm1d(in_feat)
        if normalize_output:
            self.out_bn_norm = nn.BatchNorm1d(out_feat)
        self._init_params()

    def _init_params(self,):
        if self.cp_decompose:
            self.W1 = nn.Parameter(torch.Tensor(self.n_bases, self.in_feat, 1))
            self.W2 = nn.Parameter(torch.Tensor(self.n_bases, 1, self.out_feat))
            self.W3 = nn.Parameter(torch.Tensor(self.in_feat, self.n_bases))
            nn.init.xavier_uniform_(self.W1, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.W2, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.W3, gain=nn.init.calculate_gain('relu'))
        else:
            self.W = nn.Parameter(torch.Tensor(self.in_feat, self.in_feat, self.in_feat))
            nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(self.out_feat))
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

    def message_func(self, feats, edges, etypes, norms, small_graph=True):
        msg = []
        e1 = feats[edges[:, 0]]
        r = etypes
        device = e1.device
        if self.cp_decompose:
            core_weight = torch.bmm(self.W1, self.W2)
            core_weight = torch.einsum('ib,bho->iho', self.W3, core_weight).view(r.size(1), -1)
        else:
            core_weight = self.W.view(r.size(1), -1)
        for i in range(int(np.ceil(e1.size(0) / self.split_rate))):
            x = e1[i * self.split_rate:min((i + 1) * self.split_rate, e1.size(0))]
            x = self.bn0(x)
            x = self.input_dropout(x)
            x = x.view(-1, 1, e1.size(1))
            xr = r[i * self.split_rate:min((i + 1) * self.split_rate, r.size(0))]
            xr = torch.mm(xr, core_weight)
            xr = xr.view(-1, e1.size(1), e1.size(1))
            xr = self.hidden_dropout1(xr)
            x = torch.bmm(x, xr)
            x = x.view(-1, e1.size(1))
            x = self.bn1(x)
            x = self.hidden_dropout2(x)
            msg.append(x)
        msg = torch.cat(msg)
        msg = msg * norms
        return msg

    def forward(self, g_edges, feat, etypes, norms, small_graph=False):
        if self.self_loop:
            node_repr = torch.matmul(feat, self.loop_weight)
        msgs = self.message_func(
            feat,
            g_edges,
            etypes,
            norms
        )
        msgs = scatter(msgs, g_edges[:, 2], dim=0, reduce="sum")
        node_repr[:msgs.size(0)] += msgs
        if self.bias:
            node_repr += self.h_bias
        if self.normalize_output:
            node_repr = self.out_bn_norm(node_repr)
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.output_dropout(node_repr)
        return node_repr


class LinkPredict(nn.Module):
    def __init__(
            self,
            num_nodes,
            num_rels,
            d_embd,
            n_layers=1,
            cp_decompose=False,
            n_bases=-1,
            reg_param=0.0,
            dropout_rates={},
            decoder='tucker'
    ):
        super(LinkPredict, self).__init__()
        self.embedding = nn.Parameter(torch.Tensor(num_nodes, d_embd))
        nn.init.xavier_normal_(self.embedding,
                        gain=nn.init.calculate_gain('relu'))
        self.reg_param = reg_param
        self.num_rels = num_rels
        self.rel_embed = nn.Parameter(torch.Tensor(2 * num_rels, d_embd))
        nn.init.xavier_uniform_(self.rel_embed,
                                gain=nn.init.calculate_gain('relu'))
        self.tgcn = nn.ModuleList([
            TGCNLayer(
                d_embd,
                d_embd,
                n_bases=n_bases,
                activation=F.relu if i < n_layers - 1 else None,
                self_loop=True,
                cp_decompose=cp_decompose,
                normalize_output=False if i < n_layers - 1 else True,
                dropout_rates=dropout_rates
            ) for i in range(n_layers)
        ])
        self.n_bases = n_bases
        self.decoder = decoder
        if self.decoder == 'distmult':
            self.cal_scores = self.distmult_decode
            print('Using distmult decoder')
        else:
            self.W = nn.Parameter(torch.Tensor(d_embd, d_embd, d_embd))
            nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
            self.input_dropout = nn.Dropout(dropout_rates['dr_decoder'])
            self.bn0 = nn.BatchNorm1d(d_embd)
            self.cal_scores = self.tucker_decode
            print('Using tucker decoder')
    
    def distmult_decode(self, feats, edges, mode, temperature=1.0, split_size=15000):
        positives = []
        negatives = []
        r = self.rel_embed[edges[:, 1]]
        for i in range(int(np.ceil(edges.size(0) / split_size))):
            first_idx = i * split_size
            last_idx = min((i + 1) * split_size, edges.size(0))
            temp_edges = edges[first_idx:last_idx]
            x = feats[temp_edges[:, 0]]
            xr = r[first_idx:last_idx]
            xxr = x * xr
            sim_mat = torch.mm(xxr, feats.transpose(0, 1))
            if mode == 'test':
                return sim_mat
            sim_mat /= temperature
            temp_neg = torch.logsumexp(sim_mat, dim=1)
            negatives.append(temp_neg)
            temp_pos = torch.gather(sim_mat, 1, temp_edges[:, 2][:, None])
            positives.append(temp_pos)
        negatives = torch.cat(negatives)
        positives = torch.cat(positives).squeeze(1)
        return positives, negatives

    def tucker_decode(self, feats, edges, mode, temperature=1.0, split_size=15000):
        positives = []
        negatives = []
        r = self.rel_embed[edges[:, 1]]
        core_weight = self.W.view(r.size(1), -1)
        for i in range(int(np.ceil(edges.size(0) / split_size))):
            first_idx = i * split_size
            last_idx = min((i + 1) * split_size, edges.size(0))
            temp_edges = edges[first_idx:last_idx]
            x = feats[temp_edges[:, 0]]
            x = self.bn0(x)
            x = self.input_dropout(x)
            x = x.view(-1, 1, feats.size(1))
            xr = r[first_idx:last_idx]
            xr = torch.mm(xr, core_weight)
            xr = xr.view(-1, feats.size(1), feats.size(1))
            x = torch.bmm(x, xr)
            x = x.view(-1, feats.size(1))
            sim_mat = torch.mm(x, feats.transpose(0, 1))
            if mode == 'test':
                return sim_mat
            sim_mat /= temperature
            temp_neg = torch.logsumexp(sim_mat, dim=1)
            negatives.append(temp_neg)
            temp_pos = torch.gather(sim_mat, 1, temp_edges[:, 2][:, None])
            positives.append(temp_pos)
        negatives = torch.cat(negatives)
        positives = torch.cat(positives).squeeze(1)
        return positives, negatives

    def forward(self, g_edges, h, norm):
        feats = self.embedding[h.squeeze()]
        for layer in self.tgcn:
            feats = layer(g_edges, feats, self.rel_embed[g_edges[:, 1]], norm)
        return feats

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.rel_embed.pow(2))

    def get_loss(self, embed, edges, temperature=1.0, split_size=15000):
        positives, negatives = self.cal_scores(
            embed,
            edges,
            mode='train',
            temperature=temperature,
            split_size=split_size,
        )
        loss_partial = negatives - positives
        loss = torch.mean(loss_partial)
        reg_loss = self.regularization_loss(embed)
        return loss + self.reg_param * reg_loss
    
    def predict(self, embed, edges):
        scores = self.cal_scores(
            embed,
            edges,
            mode='test'
        )
        scores = torch.sigmoid(scores)
        return scores
