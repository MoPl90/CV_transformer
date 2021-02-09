import torch
import torch.nn.functional as F
from einops import rearrange

def dice(prediction, target, train=True):
    
    if train:
        prediction = torch.softmax(prediction, dim=1)
    else:
        prediction = torch.argmax(prediction, dim=1)
        prediction = F.one_hot(prediction)
        prediction = rearrange(prediction, 'b h w c-> b c h w')
    if target.shape[1] == 1:
        target = F.one_hot(target.squeeze(1))
        target = rearrange(target, 'b h w c-> b c h w')

    intersection = torch.sum(prediction * target, axis=(2,3))   
    total_area   = torch.sum(prediction + target, axis=(2,3))
    
    return torch.mean( 2 * intersection / total_area, axis=0)
