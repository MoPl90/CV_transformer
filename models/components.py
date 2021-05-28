
import torch
from einops import rearrange, repeat
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Conv2DBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm2d(c_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.skip = nn.Sequential(nn.Conv2d(in_channels=c_in, out_channels = c_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                                  nn.UpsamplingBilinear2d(scale_factor=2)
                                 )

    def forward(self, x):
        return self.net(x) + self.skip(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_shape, out_shape, c_in, c_out, depth, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            c_tmp = min(c_in * 2, 128)
            self.layers.append(nn.ModuleList([
                Conv2DBlock(c_in=c_in, c_out=c_tmp, dropout=dropout)
            ]))
            c_in = c_tmp

        self.out_head = nn.Sequential(nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),
                                      nn.Softmax2d()
                                     )

    def forward(self, x):
        for l in self.layers:
            x = l[0](x)

        x = self.out_head(x)
        
        return x