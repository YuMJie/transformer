import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import utils




class selfattention(nn.Module):
    def __init__(self) -> None:
        super(selfattention,self).__init__()

    def forward(self,q,k,v,mask,d_k=64):
        score=torch.matmul(q,k.transpose(-1,-2))/np.sqrt(d_k)
        score=torch.masked_fill(input=score,mask=mask,value=-1e9)
        attention=nn.Softmax(-1)(score)
        results=torch.matmul(score,v)
        return results,attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,d_k,d_q,d_v,d_model) -> None:
        super().__init__()
        self.n_head=n_head
        self.d_k=d_k
        self.d_q=d_q
        self.d_v=d_v
        self.d_model=d_model
        self.W_k=nn.Linear(self.d_model,self.d_k*self.n_head)
        self.W_q=nn.Linear(self.d_model,self.d_q*self.n_head)
        self.W_v=nn.Linear(self.d_model,self.d_v*self.n_head)
        self.attention=selfattention()
        self.W_Last=nn.Linear(self.d_v*self.n_head,self.d_model)
        self.LayerNorm=nn.LayerNorm(d_model)

    def forward(self,k,q,v,mask,mask_mask=None):
        residual = q #考虑残差

        k=self.W_k(k).view(batch, -1, n_heads, d_k).transpose(1,2)
        q=self.W_q(q).view(batch, -1, n_heads, d_q).transpose(1,2)
        v=self.W_v(v).view(batch, -1, n_heads, d_v).transpose(1,2)
        mask=mask.repeat(batch, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        output,attention=self.attention(k=k,q=q,v=v,mask=mask)
        output=output.transpose(1, 2).contiguous().view(batch,src_len,self.n_head*self.d_q)#transpose之后需要contiguous
        #transpose完之后再view，要不出来是错的
        output=self.W_Last(output)
        output=self.LayerNorm(output+residual)
        return output,attention
        
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,kernel_size) -> None:
        super().__init__()
        self.conv1=nn.Conv1d(in_channels=d_model,out_channels=d_ff,kernel_size=kernel_size)
        self.conv2=nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=kernel_size)
        self.Relu=nn.ReLU()
        self.LayerNorm=nn.LayerNorm(d_model)

    def forward(self,x):
        residual = x #考虑残差
        
        embedding=self.conv1(x.transpose(2,1))#因为是对句子的不同单词做卷积，所以sentences长度是后一个
        embedding=self.conv2(embedding)
        embedding=self.Relu(embedding).transpose(1,2)
        embedding=self.LayerNorm(embedding+residual)
        return embedding
    
class EncoderLayer(nn.Module):
    def __init__(self,n_head,d_k,d_q,d_v,d_model,d_ff,kernel_size) -> None:
        super().__init__()
        self.n_head=n_head
        self.d_k=d_k
        self.d_q=d_q
        self.d_v=d_v
        self.d_model=d_model
        self.d_ff=d_ff
        self.kernel_size=kernel_size
        self.Attention=MultiHeadAttention(n_head=self.n_head,d_k=self.d_k,d_q=self.d_q,d_v=self.d_v,d_model=self.d_model)
        self.FeedForward=FeedForward(d_model=self.d_model,d_ff=d_ff,kernel_size=self.kernel_size)

    def forward(self,x,mask):
        feature,attention=self.Attention(k=x,q=x,v=x,mask=mask)
        output=self.FeedForward(x=feature)
        return output,attention

class Encoder(nn.Module):
    def __init__(self,n_head,d_k,d_q,d_v,d_model,d_ff,kernel_size,vocal_size,LayerNum=6,src_len=5) -> None:
        super(Encoder,self).__init__()  
        self.Layer_Num=Layer_Num 
        self.src_emb=nn.Embedding(num_embeddings=vocal_size,embedding_dim=d_model)
        self.pos_emb = nn.Embedding.from_pretrained(utils.get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)

        self.layers=nn.ModuleList([EncoderLayer(n_head=n_heads,d_k=d_k,d_q=d_q,d_v=d_v,d_model=d_model,d_ff=d_ff,kernel_size=1) for i in range(LayerNum)])  

    def forward(self,x):
        embeddingX=self.src_emb(x)+self.pos_emb(torch.tensor([1,2,3,4,0]))
        mask = utils.get_attn_pad_mask(x,x)
        attention_lists=[]
        for i in range(Layer_Num):
            embeddingX,attention=self.layers[i](embeddingX,mask)
            attention_lists.append(attention)
        return embeddingX,attention_lists

class DecoderLayer(nn.Module):
    def __init__(self,n_head,d_k,d_q,d_v,d_model,d_ff,kernel_size) -> None:
        super(DecoderLayer,self).__init__()
        self.n_head=n_head
        self.d_k=d_k
        self.d_q=d_q
        self.d_v=d_v
        self.d_model=d_model
        self.d_ff=d_ff
        self.kernel_size=kernel_size
        self.MaskAttention=MultiHeadAttention(n_head=self.n_head,d_k=self.d_k,d_q=self.d_q,d_v=self.d_v,d_model=self.d_model)
        self.CrossAttention=MultiHeadAttention(n_head=self.n_head,d_k=self.d_k,d_q=self.d_q,d_v=self.d_v,d_model=self.d_model)
        self.FeedForward=FeedForward(d_model=self.d_model,d_ff=d_ff,kernel_size=self.kernel_size)

    def forward(self,decoder_input,encoder_output,mask,mask_mask):
        feature1,attention1=self.MaskAttention(k=decoder_input,q=decoder_input,v=decoder_input,mask=mask)
        feature2,attention2=self.CrossAttention(k=encoder_output,q=feature1,v=encoder_output,mask=mask_mask)
        output=self.FeedForward(x=feature2)
        return output,attention1,attention2

class Decoder(nn.Module):
    def __init__(self,n_head,d_k,d_q,d_v,d_model,d_ff,kernel_size,vocal_size,LayerNum=6,src_len=5) -> None:
        super(Decoder,self).__init__()
        self.LayerNum=Layer_Num
        self.n_head=n_head
        self.d_k=d_k
        self.d_q=d_q
        self.d_v=d_v
        self.d_model=d_model
        self.d_ff=d_ff
        self.kernel_size=kernel_size
        self.src_emb=nn.Embedding(embedding_dim=self.d_model,num_embeddings=vocal_size+1)
        self.pos_emb=nn.Embedding.from_pretrained(utils.get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        self.Layers=nn.ModuleList(DecoderLayer(n_head=n_heads,d_k=d_k,d_q=d_q,d_v=d_v,d_model=d_model,d_ff=d_ff,kernel_size=1) for i in range(Layer_Num))

    def forward(self,encoder_output,decoder_input):
        embeddingDecoderOuput=self.src_emb(decoder_input)+self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        mask=utils.get_attn_pad_mask(decoder_input,decoder_input)
        mask_mask=utils.get_attn_subsequent_mask(decoder_input)
        dec_self_attns, dec_enc_attns = [], []
        for i in range(self.LayerNum):
            embeddingDecoderOuput, dec_self_attn, dec_enc_attn = self.Layers[i](decoder_input=embeddingDecoderOuput, encoder_output=encoder_output, mask=mask, mask_mask=mask_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)           
        return embeddingDecoderOuput,dec_self_attns,dec_enc_attns

class Transformer(nn.Module):
    def __init__(self,n_head,d_k,d_q,d_v,d_model,d_ff,kernel_size,vocal_size,LayerNum=6,src_len=5) -> None:
        super(Transformer,self).__init__()
        self.LayerNum=Layer_Num
        self.n_head=n_head
        self.d_k=d_k
        self.d_q=d_q
        self.d_v=d_v
        self.d_model=d_model
        self.d_ff=d_ff
        self.kernel_size=kernel_size
        self.encoder=Encoder(n_head=n_heads,d_k=d_k,d_q=d_q,d_v=d_v,d_model=d_model,d_ff=d_ff,kernel_size=1,vocal_size=src_vocab_size,LayerNum=Layer_Num)
        self.decoder=Decoder(n_head=n_heads,d_k=d_k,d_q=d_q,d_v=d_v,d_model=d_model,d_ff=d_ff,kernel_size=1,vocal_size=src_vocab_size,LayerNum=Layer_Num)
        self.project=nn.Linear(in_features=d_model,out_features=tgt_vocab_size)
    
    def forward(self,encoder_input,decoder_input):
        encoder_output,EncoderAttention=self.encoder(encoder_input)
        decoder_output,DeSelfAttention,DeEncAttention=self.decoder(decoder_input=decoder_input,encoder_output=encoder_output)
        result=self.project(decoder_output)
        return result.view(-1, result.size(-1)), EncoderAttention,DeSelfAttention,DeEncAttention


if __name__=='__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)
    batch=1#后面要改
    src_len = 5 # length of source
    tgt_len = 5 # length of target
    d_model = 512  # Embedding Size
    Layer_Num=6
    n_heads = 8
    d_k = d_v = 64
    d_ff = 2048  # FeedForward dimension
    d_q=d_k
    kernel_size=1
    model=Transformer(n_head=n_heads,d_k=d_k,d_q=d_q,d_v=d_v,d_model=d_model,d_ff=d_ff,kernel_size=1,vocal_size=src_vocab_size,LayerNum=Layer_Num)
    #x=torch.normal(0, 1, (batch, src_len))  
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.01)
    enc_inputs, dec_inputs, target_batch = utils.make_batch(src_vocab=src_vocab,tgt_vocab=tgt_vocab,sentences=sentences)
    for epoch in range(2000):
        optimizer.zero_grad()
        output,_,_,_=model(encoder_input=enc_inputs,decoder_input=dec_inputs)

        loss=criterion(output,target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    

