# atteiton used in the paper

import torch
import torch.nn as nn
import numpy as np

# multihead atteition

class ScaledDotProductAttention(nn.Module):
    """
    uses attention pad for scores
    """
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k


    # @staticmethod
    def forward(self, qmat, kmat, vmat, attention_mask):

        # scores = QK^T / sqrt(d_k)
        scores = torch.matmul(qmat, kmat.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]

        # Basically removes elements of scores tensor were we have a masked token
        scores.masked_fill_(attention_mask, -1e9) 

        # Softmax on the scores matrix
        attention = nn.Softmax(dim=-1)(scores)

        # context = softmax(QK^T / sqrt(d_k))V
        context = torch.matmul(attention, vmat)

        return context, attention



class Attention(nn.Module):
    """
    head of model
    """
    def __init__(self, d_model, d_k, d_v, num_heads):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k 
        self.d_v = d_v 
        self.num_heads = num_heads 

        # Q, K, V for each head
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)

    def forward(self, qmat, kmat, vmat, attention_mask):
        # q: [batch_size x len_q x d_model] 
        # k: [batch_size x len_k x d_model]
        # v: [batch_size x len_k x d_model]
        residual, batch_size_head = qmat, qmat.size(0)

        # 1) Do all the linear projections in batch from d_model => d_k x num_heads
        # 2) .view(): [batch_size x len_q x d_model] -> [batch_size x len_q x n_heads x d_k]
        # 3) .transpose(): [batch_size x len_q x n_heads x d_k] -> [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(qmat).view(batch_size_head, -1, self.num_heads, self.d_k).transpose(1, 2)
        # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(kmat).view(batch_size_head, -1, self.num_heads, self.d_k).transpose(1, 2)
        # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(vmat).view(batch_size_head, -1, self.num_heads, self.d_v).transpose(1, 2)
        # v_s: [batch_size x n_heads x len_k x d_v]

        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        scaledAttention = ScaledDotProductAttention(self.d_k)
        context, attention = scaledAttention.forward(q_s, k_s, v_s, attention_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size_head, -1, self.num_heads * self.d_v)
        output = nn.Linear(self.num_heads * self.d_v, self.d_model)(context)
        return nn.LayerNorm(self.d_model)(output + residual), attention  # output: [batch_size x len_q x d_model]




