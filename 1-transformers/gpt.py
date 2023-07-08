import torch 
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig
import math

class PositionEncoding(nn.Module):
    def __init__(self,max_len, n_embd):
        self.pe = nn.Embedding(max_len, n_embd)
    
    def forward(self,x, device):
        """
        inputs:
        - x : 3D input [batch size, sequence length, embedding dimension]
        
        output:
        - position embedding for the sequence length 
        """
        _, seq_len, _ = x.size()
        pos = torch.arange(0,seq_len, dtype=torch.long, device=device)
        return x + self.ps(pos)

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # head size is an integer
        self.config = config

        # key query value projection
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=config.bias)
        # final projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # model structure
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # TODO: flashattention
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size, config.block_size).view(1,1,config.block_size,config.block_size)))
    
    def forward(self, x):
        batch, seq_len, dimension = x.size()
        q,k,v = self.c_attn(x).chunk(3,dim=2)
        k = k.view(batch, seq_len, self.n_head, dimension // self.n_head).transpose(1,2) # batch, head, seq_len, dimension per head
        q = q.view(batch, seq_len, self.n_head, dimension // self.n_head).transpose(1,2) # batch, head, seq_len, dimension per head
        v = v.view(batch, seq_len, self.n_head, dimension // self.n_head).transpose(1,2) # batch, head, seq_len, dimension per head

        # self-attention: batch, seq_len, head, dimension per head @ batch, seq_len, head, dimension per head = batch,head,seq_len,seq_len
        # TODO: flash attention
        att = (q@k.transpose(-2,-1)) * (1.0 / math.sqrt(q.size(-1)))
        att = att.masked_fill(self.bias[:,:,:self.n_head,self.n_head]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(batch, seq_len, dimension)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    fully connected -> activation (GeLu) -> fully connected -> dropout"""
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd,bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.fc(x))))

class Block(nn.Module):
    """Transformer block with residual connection"""
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadedAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


