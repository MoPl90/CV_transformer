from .components import Transformer
import torch
from torch import nn
from einops import rearrange, repeat
import numpy as np

MIN_NUM_PATCHES = 15

class VT(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.image_size % args.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (args.image_size // args.patch_size) ** 2 if args.data_dim == 2 else \
                      (args.image_size // args.patch_size) ** 3
        self.patch_dim = args.channels * args.patch_size ** 2 if args.data_dim == 2 else \
                         args.channels * args.patch_size ** 3
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
            if args.data_dim == 2:
                self.mlp_head = nn.Sequential(
                                                nn.LayerNorm(args.dim),
                                                nn.Linear(args.dim, args.classes * args.image_size**2)
                                            )
            else:
                self.mlp_head = nn.Sequential(
                                                nn.LayerNorm(args.dim),
                                                nn.Linear(args.dim, args.classes * args.image_size**3)
                                            )
        else:
            self.mlp_head = nn.Sequential(
                                            nn.LayerNorm(args.dim),
                                            nn.Linear(args.dim, args.classes)
                                         )

    def forward(self, img):
        p = self.patch_size

        if len(img.shape) == 4: 
            x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        elif len(img.shape) == 5:
            x = rearrange(img, 'b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)', p1 = p, p2 = p, p3=p)

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        if self.segmentation:
            x = rearrange(x, 'b (c h w d) -> b c h w d', c=self.output_shape[0], h=self.output_shape[1], w=self.output_shape[2])
        return x
