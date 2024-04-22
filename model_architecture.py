import torch
import torch.nn as nn
import math
import numpy as np
# import format_text
import preprocess
from attention import Attention, ScaledDotProductAttention
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"


vocabulary_size, sentences_length = preprocess.get_model_sizes()

sen_length = 32  # set sentence length to this for all so equal sentence length
batch_size = sentences_length
num_layers = 6  # number of Encoder Layer
num_heads = 12  # number of heads in Multi-Head Attention
d_model = 768  # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
dropout = 0.1

def get_attention_pad_mask(seq_q, seq_k):
    """
    :return: padded attention mask
    """
    batch_size_mask, len_q = seq_q.size()
    batch_size_mask, len_k = seq_k.size()
    pad_attention_mask = seq_k.data.eq(0).unsqueeze(
        1)  # batch_size x 1 x len_k(=len_q), one is masking. eq(0) is PAD token
    return pad_attention_mask.expand(batch_size_mask, len_q, len_k) # batch_size x len_q x len_k


def tanh(x):
    """
    activation functions
    """
    return (2.0 * torch.special.expit(2.0 * x) - math.sqrt(
        2.0 / math.pi))


def gelu(x):
    """
    activation functions
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0 / math.pi)) * (x + 0.044715 * (x ** 3.0)))


class Embedding(nn.Module):
    """
    does the actual embedding
    """
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocabulary_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(sen_length, d_model)  # position embedding
        self.seg_embed = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to(device) # (seq_len,) -> (batch_size, seq_len)

        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)

        return self.norm(embedding)


class PoswiseFeedForwardNet(nn.Module):
    """
    forward model
    """
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """
    encoding model
    """
    def __init__(self, type):
        super(EncoderLayer, self).__init__()
        # simple replace of the Attentino head
        self.enc_self_attention = Attention(d_model, d_k, d_v, num_heads, type)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attention_mask):
        enc_outputs = self.enc_self_attention(enc_inputs, enc_inputs, enc_inputs,
                                                         enc_self_attention_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs


class BERT(nn.Module):
    """
    BERT model put together
    """
    def __init__(self, type):
        super(BERT, self).__init__()

        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer(type) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = tanh
        self.linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        embed_weight = self.embedding.tok_embed.weight
        num_vocab, num_dim = embed_weight.size()
        self.decoder = nn.Linear(num_dim, num_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(num_vocab))

    def forward(self, modelinput_ids, modelsegment_ids, modelmasked_pos):
        output = self.embedding(modelinput_ids, modelsegment_ids)
        enc_self_attention_mask = get_attention_pad_mask(modelinput_ids, modelinput_ids)

        for layer in self.layers:
            output = layer(output, enc_self_attention_mask)
        # output : [batch_size, len, d_model], attention : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)

        modelmasked_pos = modelmasked_pos[:, :, None].expand(-1, -1, output.size(-1))  # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, modelmasked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ1(self.norm(self.activ2(self.linear(h_masked))))
        logits_lm_model = self.decoder(h_masked) + self.decoder_bias  # [batch_size, max_pred, n_vocab]

        return logits_lm_model


def get_model(type):
    return BERT(type) 
