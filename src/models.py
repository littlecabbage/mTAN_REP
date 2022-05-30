#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NF import MAF, RealNVP

class create_classifier(nn.Module):
 
    def __init__(self, latent_dim, nhidden=16, N=2):
        super(create_classifier, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, N))
       
    def forward(self, z):
        _, out = self.gru_rnn(z)
        return self.classifier(out.squeeze(0))
    

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    def attention(self, query, key, value, mask=None, dropout=None):
        """
        > The function takes in a query, key, and value, and returns the weighted sum of the values, and the
        attention weights
        
        :param query: the query from the decoder's first attention layer, with shape [batch_size,
        max_length, d_model]
        :param key: the key matrix
        :param value: the input to the attention layer
        :param mask: mask to prevent seeing future words (for decoder)
        :param dropout: The dropout probability
        :return: The attention vector and the attention weights.
        """
        
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        
        p_attn = F.softmax(scores, dim = -2)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        
        x, _ = self.attention(query, key, value, mask, dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)

        return self.linears[-1](x)
    
    
class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out
    
    
class dec_mtan_rnn(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)    
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps).to(self.device)
            key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, out)
        out = self.z0_to_obs(out)
        return out        
   
    
class enc_mtan_classif(nn.Module):
 
    def __init__(self, input_dim, query, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(enc_mtan_classif, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        # self.classifier = nn.Sequential(
        #     nn.Linear(nhidden, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 2))
        self.enc = nn.GRU(nhidden, nhidden)
        self.nf = MAF(n_blocks = 1, input_size = 1, hidden_size = 32, n_hidden = 1, cond_label_size=32, batch_norm=True,activation='tanh')

        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        data = x[:, :, :self.dim]
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
            
        out = self.att(query, key, x, mask) # 输出RT， RS # input x[128, 60, 88] // output out[128,128,128]

        out = out.permute(1, 0, 2) # 注释掉
        _, out = self.enc(out) # 注释掉
        # return self.classifier(out.squeeze(0)) # 修改

        data = data[:, -1, :]
        data =  data.reshape(-1,1)
        out = out.reshape(-1, 32)
        print(f"out {out.shape} data {data.shape}")
        log_prob = self.nf.log_prob(data,out)

        print(log_prob, log_prob.shape)
        return log_prob.mean()




class enc_mtan_classif_activity(nn.Module):
 
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(enc_mtan_classif_activity, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        # self.gru = nn.GRU(nhidden, nhidden, batch_first=True)
        # self.linear = nn.Linear(nhidden, self.dim * 2)
        
        self.nf = MAF(n_blocks = 1, input_size = 1, hidden_size = 64, n_hidden = 1, cond_label_size=4, batch_norm=False,activation='tanh')
        
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        batch = x.size(0)
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
        
        out = self.att(key, key, x, mask)

        data = x[:,:,:self.dim]

        out = out.reshape([-1,4])

        data = data.reshape([-1,1])

        log_prob = self.nf.log_prob(data,out).reshape([batch, -1])
        
        log_prob = log_prob.reshape([batch,-1,self.dim])
        
        log_prob = log_prob.mean(dim = 1)
       
        return log_prob
    
    
    
class enc_interp(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device='cuda'):
        super(enc_interp, self).__init__()
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.cross = nn.Linear(2*input_dim, 2*input_dim)
        self.bandwidth = nn.Linear(1, 2*input_dim, bias=False)
        self.gru_rnn = nn.GRU(2*input_dim, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))

    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        query, key = query.to(self.device), key.to(self.device)
        batch, seq_len, dim = value.size()
        scores = -(query.unsqueeze(-1) - key.unsqueeze(-2))**2
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        bandwidth = torch.log(1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1).to(self.device))))
        scores = scores * bandwidth
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
        
    def forward(self, x, time_steps):
        #time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        out, _ = self.attention(self.query.unsqueeze(0), time_steps, x, mask)
        out = self.cross(out)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out
    
    
class dec_interp(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device='cuda'):
        super(dec_interp, self).__init__()
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.bandwidth = nn.Linear(1, 2*nhidden, bias=False)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        query, key = query.to(self.device), key.to(self.device)
        batch, seq_len, dim = value.size()
        scores = -(query.unsqueeze(-1) - key.unsqueeze(-2))**2
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        bandwidth = torch.log(1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1).to(self.device))))
        scores = scores * bandwidth
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
        
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        out, _ = self.attention(time_steps, self.query.unsqueeze(0), out)
        out = self.z0_to_obs(out)
        return out        

    
class enc_rnn3(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, use_classif=False, learn_emb=False, device='cuda'):
        super(enc_rnn3, self).__init__()
        self.use_classif = use_classif 
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.cross = nn.Linear(2*input_dim, nhidden)
        if use_classif:
            self.gru_rnn = nn.GRU(nhidden, nhidden, batch_first=True)
        else:
            self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time) for _ in range(2)])
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        if use_classif:
            self.classifier = nn.Sequential(
                nn.Linear(nhidden, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, 2))
        else:
            self.hiddens_to_z0 = nn.Sequential(
                nn.Linear(2*nhidden, 50),
                nn.ReLU(),
                nn.Linear(50, latent_dim * 2))
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        d_k = query.size(-1)
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device) 
        out, _ = self.attention(query, key, x, mask)
        out = self.cross(out)
        if not self.use_classif:
            out, _ = self.gru_rnn(out)
            out = self.hiddens_to_z0(out)
        else:
            _, h = self.gru_rnn(out)
            out = self.classifier(h.squeeze(0))
        return out
    
    
class dec_rnn3(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16,learn_emb=False, device='cuda'):
        super(dec_rnn3, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                     nn.Linear(embed_time, embed_time),
                                     nn.Linear(2*nhidden, 2*nhidden)])
      
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        d_k = query.size(-1)
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps).to(self.device)
            key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out, _ = self.attention(query, key, out)
        out = self.z0_to_obs(out)
        return out        
    
if __name__ == "__main__":
    test = torch.ones([1,60,45])
    mask = torch.ones([1,60,45])

    tp = torch.linspace(1,60, steps=60)
    tp = tp.reshape([1, 60])

    # print(tp)
    print(test.shape, mask.shape, tp.shape)
    # def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
    model = enc_mtan_classif_activity(
        input_dim = 45,
        nhidden=512,
        embed_time = 128,
        num_heads = 1,
        learn_emb=False,
        freq=10,
        device='cpu'
    )

    print(model(torch.cat([test, mask],2), tp))

