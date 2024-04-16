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
        scores = torch.matmul(qmat, kmat.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attention_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attention = nn.Softmax(dim=-1)(scores)
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
        self.W_Q = nn.Linear(d_model, d_q * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)

    def forward(self, qmat, kmat, vmat, attention_mask):
        residual, batch_size_head = qmat, qmat.size(0)

        q_s = self.W_Q(qmat).view(batch_size_head, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(kmat).view(batch_size_head, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(vmat).view(batch_size_head, -1, self.num_heads, self.d_v).transpose(1, 2)

        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        scaledAttention = ScaledDotProductAttention(self.d_k)
        context, attention = scaledAttention.forward(q_s, k_s, v_s, attention_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size_head, -1, self.num_heads * self.d_v)
        output = nn.Linear(self.num_heads * self.d_v, self.d_model)(context)
        return nn.LayerNorm(self.d_model)(output + residual), attention  # output: [batch_size x len_q x d_model]




