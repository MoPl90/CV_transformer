from skimage.transform import resize, rescale, AffineTransform
import numpy as np


class Resize(object):
    
    def __init__(self, new_size):
        self.new_size = new_size


    def __call__(self, sample):
        image, mask = sample
        image = image.squeeze()
        mask = mask.squeeze()

        if isinstance(self.new_size, (int, float)):
            self.new_size = len(image.shape)*[self.new_size]
        
        if self.new_size == list(image.shape):
            
            return image[np.newaxis,...], mask[np.newaxis,...]
        
        image = resize(image,
                       self.new_size, 
                       preserve_range=True,
                       mode='constant', 
                       anti_aliasing=False)
        mask = resize(mask, 
                      self.new_size, 
                      preserve_range=True,
                      mode='constant', 
                      order=0, 
                      anti_aliasing=False)

        return image[np.newaxis,...], mask[np.newaxis,...]


class RandomZoom(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample
        image = image.T
        mask = mask.T

        img_size = image.shape[:-1]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image, 
            scale,
            preserve_range=True,
            multichannel=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            scale,
            order=0,
            preserve_range=True,
            multichannel=True,
            mode="constant",
            anti_aliasing=False,
        )
        if scale < 1.0:
            diff = (img_size[0] - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * len(img_size) + ((0,0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size[0]) // 2
            x_max = x_min + img_size[0]
            image = image[x_min:x_max, x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, x_min:x_max, ...]

        return image.T, mask.T

class RandomHorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample
        image = image.squeeze()
        mask = mask.squeeze()

        if np.random.rand() > self.flip_prob:
            return image[np.newaxis,...], mask[np.newaxis,...]

        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()

        return image[np.newaxis,...], mask[np.newaxis,...]