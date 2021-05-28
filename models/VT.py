from .components import Transformer, Decoder
import torch
from torch import nn
from einops import rearrange, repeat
import numpy as np

MIN_NUM_PATCHES = 15

class VT(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.image_size % args.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (args.image_size // args.patch_size) ** 2
        self.patch_dim = args.channels * args.patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert args.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = args.patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, args.dim))
        self.patch_to_embedding = nn.Linear(self.patch_dim, args.dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.dim))
        self.dropout = nn.Dropout(args.emb_dropout)

        self.transformer = Transformer(args.dim, args.depth, args.heads, args.dim_head, args.mlp_dim, args.dropout)

        self.pool = args.pool
        self.to_latent = nn.Identity()

        self.segmentation = args.segmentation
        self.output_shape = (args.classes, args.image_size, args.image_size) if args. segmentation else args.classes

        if self.segmentation:
            self.decoder = Decoder(in_shape=(args.patch_size, args.patch_size), 
                                   out_shape=self.output_shape[1:], 
                                   c_in=args.dim // args.patch_size**2, 
                                   c_out=args.classes, 
                                   depth=int(np.log2(np.sqrt(num_patches))), 
                                   dropout=args.dropout)
        else:
            self.mlp_head = nn.Sequential(
                                            nn.LayerNorm(args.dim),
                                            nn.Linear(args.dim, args.classes)
                                         )

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        if self.segmentation:
            x = rearrange(x, 'b (c h w) -> b c h w', h=self.patch_size, w=self.patch_size)
            x = self.decoder(x)

        else:
            x = self.mlp_head(x)
        return x

