import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from emb_model import *
from conformer import *
from conformer.encoder import *
from conformer.activation import GLU

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=3000, return_vec=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.return_vec = return_vec
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        if not self.return_vec: 
            # x: (batch_size*num_windows, window_size, input_dim)
            x = x[:] + self.pe.squeeze()

            return self.dropout(x)
        else:
            return self.pe.squeeze()

class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)

class cross_attn(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, dropout=0.1):
        super(cross_attn, self).__init__()
        
        self.nhead = nhead
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, nhead * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, nhead * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, nhead * d_v, bias=False)
        self.fc = nn.Linear(nhead * d_v, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, q, k, v, mask=None):
        d_k, d_v, nhead = self.d_k, self.d_v, self.nhead
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, nhead, d_k)
        k = self.w_ks(k).view(sz_b, len_k, nhead, d_k)
        v = self.w_vs(v).view(sz_b, len_v, nhead, d_k)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
            
        attn = torch.matmul(q / d_k**0.5, k.transpose(-2, -1))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        output = self.dropout(self.fc(output))
        output += residual
        
        output = self.layer_norm(output)
        
        return output

class cross_attn_layer(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, conformer_class, d_ffn):
        super(cross_attn_layer, self).__init__()

        self.cross_attn = cross_attn(nhead=nhead, d_k=d_k, d_v=d_v, d_model=conformer_class)
        self.ffn = nn.Sequential(nn.Linear(conformer_class, d_ffn),
                                    nn.ReLU(),
                                    nn.Linear(d_ffn, conformer_class),
                                )   
        self.layer_norm = nn.LayerNorm(conformer_class, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v):
        out_attn = self.cross_attn(q, k, v)
            
        out = self.layer_norm(self.ffn(out_attn) + out_attn)

        return self.dropout(out)

class SingleP_Conformer(nn.Module):
    def __init__(self, conformer_class, d_model, d_ffn, n_head, enc_layers, dec_layers, norm_type='max', l=10, decoder_type='crossattn', encoder_type='conformer'):
        super(SingleP_Conformer, self).__init__()
        
        assert encoder_type in ['conformer', 'transformer'], "encoder_type must be one of ['conformer', 'transformer']"
        assert decoder_type in ['upsample', 'crossattn', 'unet', 'MGAN'], "encoder_type must be one of ['upsample', 'crossattn', 'unet', 'MGAN']"

        self.encoder_type = encoder_type
        if encoder_type == 'conformer':
            self.conformer = Conformer(num_classes=conformer_class, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=n_head, num_encoder_layers=enc_layers)

        self.fc = nn.Linear(conformer_class, 1)
        self.sigmoid = nn.Sigmoid()

        self.decoder_type = decoder_type
        self.dec_layers = dec_layers
       
        if decoder_type == 'crossattn':
            self.crossAttnLayer = nn.ModuleList([cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn)
                                                for _ in range(dec_layers)]
                                                )
            self.pos_emb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
        
    def forward(self, wave, input_lengths=3000):
        wave = wave.permute(0,2,1)
        
        if self.encoder_type == 'conformer':
            out, _ = self.conformer(wave, input_lengths)

        if self.decoder_type == 'crossattn':
            pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
            
            for i in range(self.dec_layers):
                if i == 0:
                    dec_out = self.crossAttnLayer[i](pos_emb, out, out)
                else:
                    dec_out = self.crossAttnLayer[i](dec_out, out, out)
                
            out = self.sigmoid(self.fc(dec_out))

        return out
