import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 

def make_batch(src_vocab,tgt_vocab,sentences):
     input_batch=[[src_vocab[i] for i in sentences[0].split()]]
     output_batch=[[tgt_vocab[i] for i in sentences[1].split()]]
     target_batch=[[tgt_vocab[i] for i in sentences[2].split()]]
     return torch.LongTensor(input_batch),torch.LongTensor(output_batch),torch.LongTensor(target_batch)

#去除最后一个终止字符
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask1 = seq_k.data.eq(0)  # batch_size x 1 x len_k(=len_q), one is masking
    pad_attn_mask2 = pad_attn_mask1.unsqueeze(1)
    pad_attn_mask3=pad_attn_mask2.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k
    return pad_attn_mask3

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)