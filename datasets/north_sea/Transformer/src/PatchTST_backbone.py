__all__ = ['PatchTST_backbone']

#inspired from https://github.com/yuqinie98/PatchTST

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
#from collections import OrderedDict
from .PatchTST_layers import *
from .RevIN import RevIN

from fastai.layers import Module

# Cell
class PatchTST_backbone(Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, n_markers:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, rope=False, alibi=False, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        self.key_padding_mask=key_padding_mask


        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, n_markers=n_markers, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, rope=False, alibi=False, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z, output_attentions: bool = False):                                                                   # z: [bs x nvars x seq_len]
         
        #print("initial z = ,",z.cpu().detach().numpy())
        # norm
        ##print("first batch z = " , z[0].cpu().detach().numpy())
        ###print("z after revin = ,",z.cpu().detach().numpy())
        ###print(z.shape)

        # do patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(z)
        ###print(z.shape)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        x = x.permute(0,1,3,2)      
        ###print("just before backbone call")                                                        # z: [bs x nvars x patch_len x patch_num]
        ###print(z.shape)

        #masking patches
        if self.key_padding_mask=="auto":
            x,key_padding_mask_tensor = self.key_padding_masking(x)
        

        z = torch.nan_to_num(z)
        if self.revin: #todo implement revin for inputs with nan values
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
        #print("z after revin = ,",z.cpu().detach().numpy())
        ###print(z.shape)

        # add padding to last patch if necessary
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        ###print(z.shape)
        z= z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2) 

        ###print("key padding masking in patchtst_backbone forward = ",key_padding_mask_tensor)
        # model
        z = self.backbone(z,key_padding_mask_tensor=key_padding_mask_tensor,output_attentions=output_attentions)                # z: [bs x nvars x d_model x patch_num]
        
        ###print("z after backbone : ",z.shape)
        ##print("from patchtst backbone, z = ",z)
        if output_attentions:
            z,attn = z
        #print("z after backbone = ,",z.cpu().detach().numpy())
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        ####print("z after head : ",z.shape)
        z = z [:,0,:,:]  # z: [bs x nvars x target_window] 
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        if output_attentions:
            return z,attn
        else:
            return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )
    def key_padding_masking(self, x): 
            # x shape: [bs, n_vars, patch_len, patch_num]
            bs, n_vars, patch_len, patch_num = x.shape

            # Determine which patches are purely Padding (All NaNs)
            # Check along patch_len (dim=2). Result: [bs, n_vars, patch_num]
            patch_is_nan = torch.all(torch.isnan(x), dim=2)

            # Create the Master Mask (Shared across variables)
            # Result: [bs, patch_num]
            mask = torch.all(patch_is_nan, dim=1) 

            # This turns all NaNs (even partial ones) into 0.0, which allows the model to run.
            x = torch.nan_to_num(x, nan=0.0)

            # Explicitly zero out fully padded patches
            mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand(bs, n_vars, patch_len, patch_num)
            x = x.masked_fill(mask_expanded, 0.0)

            # 5. Return
            if mask.any():
                # Return mask: [bs, patch_num]
                # Note: If your backbone later flattens Batch*Vars, you might need  to repeat this mask. 
                # But usually, broadcasting handles it. The backbone flattens [Batch, Vars] -> [Batch*Vars]

                # [bs, patch_num] -> [bs, n_vars, patch_num]
                mask_expanded = mask.unsqueeze(1).repeat(1, n_vars, 1)

                # Flatten: [bs * n_vars, patch_num]
                mask_flat = mask_expanded.reshape(-1, patch_num)

                return x, mask_flat
            else:
                return x, None


class Flatten_Head(Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
class Lambda(nn.Module):
        def __init__(self, func):
            super(Lambda, self).__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)
        
    
    
class TSTiEncoder(Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, n_markers, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, rope=False, alibi=False, verbose=False, **kwargs):
        
        
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len

        # Initialize RoPE embeddings
        #self.rope_embeddings = RoPEEmbeddings(d_model)
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        self.key_padding_mask=key_padding_mask

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, rope=False, alibi=False, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        self.classification_head = nn.Sequential(
                                nn.Linear(in_features=d_model, out_features=n_markers+2), Lambda(lambda x: x.squeeze(1))
                            )
    def forward(self, x, key_padding_mask_tensor, output_attentions:bool = False) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        ##print("tsti encoder, is input nan ? : ",x.isnan().any())
        ###print(x.shape)
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]
        #print("z after droput = ,",u.cpu().detach().numpy())
        # Encoder
        ###print("key_padding_mask_tensor = ",key_padding_mask_tensor)
        
        z = self.encoder((u,key_padding_mask_tensor,output_attentions))          # z: [bs * nvars x patch_num x d_model]
        
        if output_attentions:
            z,attn=z
        #print("z after encoder = ,",z.cpu().detach().numpy())
        z = self.classification_head(z) #classification
        ##print("z = ",z)
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        if output_attentions:
            #print("z = ",z)
            #print("attn = ",attn)
            return z,attn
        else:
            return z    

# Cell
class TSTEncoder(Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu', rope=False, alibi=False,
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, rope=False, alibi=False, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src, attn_mask:Optional[Tensor]=None):
        
        if len(src)==3:
            src, key_padding_mask_tensor,output_attentions = src
            # #print("src = ",src)
            # #print("key padding mask tensor = ",key_padding_mask_tensor)
            # #print("output attention = ",output_attentions)
        else:
            src,key_padding_mask_tensor=src
            output_attentions = False
        ##print("tst encoder, is input nan ? : ",src.isnan().any())
        ###print("tstencodere here")
        output = src
        scores = None
        ##print("self res attention = ",self.res_attention)
        if self.res_attention or output_attentions:
            for mod in self.layers: 
                output, scores = mod(output, prev=scores, key_padding_mask_tensor=key_padding_mask_tensor, attn_mask=attn_mask,output_attentions=output_attentions)
                #print("z after mod = ,",output.cpu().detach().numpy())
            if output_attentions:
                return output,scores
            else:
                return output
                
        else:
            for mod in self.layers: 
                output = mod(output, key_padding_mask_tensor=key_padding_mask_tensor, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", rope=False, alibi=False, res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, rope=False, alibi=False,res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, key_padding_mask_tensor:Optional[Tensor]=None, prev:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, output_attentions: bool = False) -> Tensor:
        ##print("tstencoder layer , is input nan ? : ",src.isnan().any())
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention or output_attentions:
            ##print("src any nan ",src.isnan().any())
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask_tensor=key_padding_mask_tensor, attn_mask=attn_mask,output_attentions=output_attentions)
            ##print("attn any nan ", attn.isnan().any())
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask_tensor=key_padding_mask_tensor, attn_mask=attn_mask)

        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        #print("z after self_attn = ,",src2.cpu().detach().numpy())
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention or output_attentions:
            return src, scores
        else:
            return src



class _MultiheadAttention(Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, rope=False, alibi=False, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.rope = rope  # Store the rope parameter
        self.alibi = alibi  #store the alibi parameter
        
        # Initialize RoPE embeddings
        if self.rope: 
            print('Initialize RoPE embeddings')
            self.rope_embeddings = RoPEEmbeddings(d_k)
            
        
        # Initialising ALiBi: precompute head-wise slopes once (buffer = not trainable)
        if self.alibi:
            slopes = alibi_slopes(n_heads)  # shape: (H,)
            self.register_buffer("alibi_slopes", slopes, persistent=False)


        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Project output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask_tensor: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, output_attentions: bool = False):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)   # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply RoPE embeddings to Q and K
        if self.rope:
            q_s = self.rope_embeddings(q_s)
            k_s = self.rope_embeddings(k_s)
            
        
        # apply ALiBi with attn_mask (additive) ---
        if self.alibi:
            L_q = q_s.size(-2)                # length of queries
            L_k = k_s.size(-1)                # length of keys
            alibi_bias = build_alibi_bias(
                L_q=L_q, L_k=L_k, slopes=self.alibi_slopes,
                device=q_s.device, dtype=q_s.dtype
            )                                 # shape: (1, H, L_q, L_k)

            # Ensure attn_mask is additive (float) and broadcastable
            attn_mask = _as_additive_mask(attn_mask, ref_dtype=q_s.dtype)
            if attn_mask is None:
                attn_mask = alibi_bias
            else:
                attn_mask = attn_mask + alibi_bias  # relies on broadcasting


        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention or output_attentions:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask_tensor=key_padding_mask_tensor, attn_mask=attn_mask, output_attentions=output_attentions)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask_tensor=key_padding_mask_tensor, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # Back to the original input dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention or output_attentions: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights

        



class _ScaledDotProductAttention(Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask_tensor:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,output_attentions : bool = False):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
        ##print("scaled dot product attention, is input nan ? : ",q.isnan().any())
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # if attn_scores.isnan().any():
        #     assert True==False

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: 
            ##print("prev is not None")
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # if attn_scores.isnan().any():
        #     assert True==False

        ###print("new attn_scores contains nan= ",attn_scores.isnan().any())
        # Key padding mask (optional)
        if key_padding_mask_tensor is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            ##print("testing masking mechanism from pytorch")
            #key_padding_mask_tensor = key_padding_mask_tensor.view(key_padding_mask_tensor.shape[0], 1, 1, key_padding_mask_tensor.shape[1]).expand(-1, self.n_heads, -1, -1)
            key_padding_mask_tensor = key_padding_mask_tensor.unsqueeze(1).unsqueeze(2).repeat(1,1,key_padding_mask_tensor.shape[1],1)
            #print("key_padding_mask_tensor = ,",key_padding_mask_tensor.cpu().detach().numpy())
            attn_scores.masked_fill_(key_padding_mask_tensor, -np.inf)
        ##print("after masking attn_scores contains nan= ",attn_scores.isnan().any())
        ##print("before softmax, attn_scores = ",attn_scores.cpu().detach().numpy())
        # normalize the attention weights
        #print("attn_scores before softmax = ,",attn_scores.cpu().detach().numpy())
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        ##print("after softmax contains nan= ",attn_weights.isnan().any())
        #print("attn_scores after softmax = ,",attn_scores.cpu().detach().numpy())
        attn_weights = self.attn_dropout(attn_weights)
        ##plot attention map
        # import matplotlib.pyplot as plt
        # att_map = attn_scores[0]
        # att_map_norm = torch.mean(att_map,dim=0).cpu().detach().numpy()
        # plt.imshow(att_map_norm, interpolation='none')
        # plt.show()
        #assert True==False
        ##print("after dropout attn_weights contains nan= ",attn_weights.isnan().any())
        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention or output_attentions: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights
        





