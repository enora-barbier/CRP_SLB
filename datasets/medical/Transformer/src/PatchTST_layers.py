__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding', 'SinCosPosEncoding', 
           'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding', 'RoPEEmbeddings', 
           'alibi_slopes', 'build_alibi_bias', '_as_additive_mask' ]           

#inspired from https://github.com/yuqinie98/PatchTST
import torch
from torch import nn
from torch import Tensor
import math

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

    
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 
    
    
# decomposition

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
    
    
# pos_encoding

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)



class RoPEEmbeddings(nn.Module):
    def __init__(self, dim):
        super(RoPEEmbeddings, self).__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        # print(f"x shape: {x.shape}")
        seq_len = x.size(-2)  # Adjusted to match the sequence length dimension
        pos = torch.arange(seq_len, dtype=torch.float, device=x.device)
        sinusoid_inp = torch.einsum('i,j->ij', pos, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        # print(f"pos_emb shape before adjustment: {pos_emb.shape}")

        # Adjust pos_emb to match the dimensions of x
        if x.size(-1) == pos_emb.size(-1):
            pos_emb = pos_emb.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1, -1)
        else:
            
            seq_len = x.size(-1)  # Adjusted to match the sequence length dimension
            pos = torch.arange(seq_len, dtype=torch.float, device=x.device)
            sinusoid_inp = torch.einsum('i,j->ij', pos, self.inv_freq)
            pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
            # pos_emb = pos_emb.reshape()
            # print(f"regenerated positional embedding: {pos_emb.shape}")
            pos_emb = pos_emb.T.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), x.size(2),-1)
        
        # print(f"pos_emb shape after adjustment: {pos_emb.shape}")
        return self.apply_rotary_pos_emb(x, pos_emb)

    def apply_rotary_pos_emb(self, x, pos_emb):
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        pos_emb1, pos_emb2 = pos_emb[..., :self.dim // 2], pos_emb[..., self.dim // 2:]
        # print(f"x1 shape: {x1.shape}")
        # print(f"x2 shape: {x2.shape}")
        # print(f"pos_emb1 shape: {pos_emb1.shape}")
        # print(f"pos_emb2 shape: {pos_emb2.shape}")
        
        return torch.cat([x1 * pos_emb1.cos() - x1 * pos_emb1.sin(),
                          x2 * pos_emb2.sin() + x2 * pos_emb2.cos()], dim=-1)
    

#Alibi position encoding
def alibi_slopes(n_heads: int, min_log2: float = -8.0, max_log2: float = 0.0):

    """
    Returns n_heads positive slopes in a geometric progression in [2^min_log2, 2^max_log2] (inclusive).
    You will multiply them by -distance to get a *negative* bias on distant tokens.
    This mirrors the common ALiBi practice of having a spectrum of slopes across heads.
    """
    # Geometric progression in base 2 for numerical stability/interpretability
    # Example: for n_heads=8, slopes ~ [1.0, 0.66, 0.44, ..., 2^-8]
    if n_heads == 1:
        return torch.tensor([1.0])
    exponents = torch.linspace(max_log2, min_log2, steps=n_heads)  # from ~1.0 down to ~2^-8
    return (2.0 ** exponents)  # positive slopes; we'll apply a negative sign when building the bias


def build_alibi_bias(L_q: int, L_k: int, slopes: torch.Tensor, device=None, dtype=None):
    """
    Build an ALiBi bias tensor of shape (1, n_heads, L_q, L_k).
    Bias = - slope[h] * (i - j), where i (query pos) >= j (key pos) in causal attention.
    For non-causal attention, we still compute distances but you may want to clamp negatives to 0.
    """
    device = device if device is not None else slopes.device
    dtype = dtype if dtype is not None else torch.float32

    # Positions
    q_pos = torch.arange(L_q, device=device, dtype=torch.long)[:, None]  # (L_q, 1)
    k_pos = torch.arange(L_k, device=device, dtype=torch.long)[None, :]  # (1, L_k)

    # Distance from key to query (non-negative for j<=i when causal)
    distances = (q_pos - k_pos).clamp(min=0).to(dtype)  # (L_q, L_k)

    # Broadcast to heads
    # slopes: (n_heads,) -> (n_heads, 1, 1)
    slopes = slopes.to(device=device, dtype=dtype)
    slopes = slopes.view(-1, 1, 1)
    bias = -(slopes * distances)  # (n_heads, L_q, L_k)
    return bias.unsqueeze(0)  # (1, n_heads, L_q, L_k)


def _as_additive_mask(mask: Tensor, ref_dtype: torch.dtype) -> Tensor:
    """
    Convert boolean/byte masks to additive (-inf/0) masks. If already float, return as-is.
    Expected shape will broadcast to (B|1, H|1, L_q, L_k).
    """
    if mask is None:
        return None
    if mask.dtype == torch.bool or mask.dtype == torch.uint8:
        # True means masked -> add -inf
        return mask.masked_fill(mask, float('-inf')).to(ref_dtype)
    if not mask.is_floating_point():
        return mask.to(ref_dtype)
    return mask