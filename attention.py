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
        # context = torch.matmul(attention, vmat)
        finalAttention = torch.matmul(attention, vmat)

        # return context, attention
        return finalAttention #, attention



class Attention(nn.Module):
    """
    head of model
    """
    def __init__(self, d_model, d_v, num_heads, attention_type):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.d_v = d_v 
        self.num_heads = num_heads 

        # Q, K, V for each head
        self.W_Q = nn.Linear(d_model, d_v * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.selectedAttention = SingleLinearAttentionLayer(self.d_k) if attention_type == "SLL" else ScaledDotProductAttention(self.d_k)

    def forward(self, qmat, vmat, attention_mask):
        # q: [batch_size x len_q x d_model] 
        # k: [batch_size x len_k x d_model]
        # v: [batch_size x len_k x d_model]

        # 1) get the size of the batch
        residual, batch_size_head = qmat, qmat.size(0)
        
        # 2) get the key, query, and value matracies in each dimension

        # a) Do all the linear projections in batch from d_model => d_k x num_heads
        # b) .view(): [batch_size x len_q x d_model] -> [batch_size x len_q x n_heads x d_k]
        # c) .transpose(): [batch_size x len_q x n_heads x d_k] -> [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(qmat).view(batch_size_head, -1, self.num_heads, self.d_v).transpose(1, 2)
        # q_s: [batch_size x n_heads x len_q x d_k]
        v_s = self.W_V(vmat).view(batch_size_head, -1, self.num_heads, self.d_v).transpose(1, 2)
        # v_s: [batch_size x n_heads x len_k x d_v]

        # 3) get the the attention mask 

        # a) og attention mask = [batch_size x seq_len]
        # b) attention mask.unsqueeze(1) = [batch_size x 1 x seq_len]
        # c) attention mask.unsqueeze(1).repeat(1, self.num_heads, 1) = [batch_size x n_heads x seq_len]
        #    Repeats the mask for each head
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)


        # 4) pass attention through each head

        # attention = [batch_size x n_heads x seq_len x d_k]
        
        # pick one:
        # scaledAttention = ScaledDotProductAttention(self.d_k)
        scaledAttention = SingleLinearAttentionLayer(self.d_v)

        multi_head_self_attention = scaledAttention.forward(q_s, v_s, attention_mask)
        # output: for each batch, for each attention head, for each token in the sentence,
        # the attention value

        # 5) combine attention from heads

        # convert the multi_head_self_attention 
        # 1) .transpose(1, 2) -> [batch_size x len_q x n_heads x d_v]
        context = multi_head_self_attention.transpose(1, 2).contiguous().view(batch_size_head, -1, self.num_heads * self.d_v)
        # for each batch, for each token in the sentence, for each attention head,
        # the attention value
        # converts to ---> 
        # batch_size_head x attention 
        # think the attentions tensors are concatenated together from different heads
        # attention = (num heads * d_v)

        # 6) convert concatenated attention to model size

        output = nn.Linear(self.num_heads * self.d_v, self.d_model)(context)
        # [batch size x sequence length x concatenated attention] 
        # converts to ---> 
        # [batch size x sequence length]

        # 7) add new outputs with old query valeus and normalize

        # output: [batch_size x len_q x d_model]
        return nn.LayerNorm(self.d_model)(output + residual)  


class SingleLinearAttentionLayer(nn.Module):
    """
    uses attention pad for scores
    """
    def __init__(self, d_v):
        super(SingleLinearAttentionLayer, self).__init__()
        self.d_v = d_v

        self.linearLayer = nn.Linear(d_v * 2, d_v)


    # @staticmethod
    def forward(self, qmat, vmat, attention_mask):

        # q: [batch_size x num_heads x d_model x d_k] 
        # k: [batch_size x num_heads x d_model x d_k]
        # v: [batch_size x num_heads x d_model x d_k]

        # 1) get the size of the batch
        # residual, batch_size_head = qmat, qmat.size(0)
        
        combined = torch.cat((qmat, vmat), 3)

        output = self.linearLayer(combined)
        return output




        # # scores = QK^T / sqrt(d_k)
        # scores = torch.matmul(qmat, kmat.transpose(-1, -2)) / np.sqrt(
        #     self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]

        # # Basically removes elements of scores tensor were we have a masked token
        # scores.masked_fill_(attention_mask, -1e9) 

        # # Softmax on the scores matrix
        # attention = nn.Softmax(dim=-1)(scores)

        # # context = softmax(QK^T / sqrt(d_k))V
        # # context = torch.matmul(attention, vmat)
        # finalAttention = torch.matmul(attention, vmat)

        # return context, attention
        # return finalAttention #, attention

