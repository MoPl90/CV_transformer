import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from pathlib import Path


class OasisDataset(VisionDataset):
	def __init__(self, train, root_dir, transform=None):
		super().__init__(root_dir)
		if train:
			self.ext_im = 'im_train'
			self.ext_lb = 'lb_train'
		else:
			self.ext_im = 'im_val'
			self.ext_lb = 'lb_val'
	
		self.fnames = os.listdir(os.path.join(self.root, self.ext_im))
		self.numfiles = len(self.fnames)
		self.transform = transform
		
		assert len(os.listdir(os.path.join(self.root, self.ext_lb))) == len(self.fnames), "Corrupt image/label folders"

	def __len__(self):
		return self.numfiles

	def __getitem__(self, idx):
		im = Image.fromarray(np.load(os.path.join(self.root + '/' + self.ext_im, self.fnames[idx])))
		lb = Image.fromarray(np.load(os.path.join(self.root + '/' + self.ext_lb, self.fnames[idx])))


		seed = np.random.randint(2147483647) # make a seed with numpy generator 
		random.seed(seed) # apply this seed to img tranfsorms
		torch.manual_seed(seed) # needed for torchvision 0.7
		if self.transform is not None:
			im = self.transform(np.array(im))

			#reset the seed	
			random.seed(seed)
			torch.manual_seed(seed)
			lb = self.transform(np.array(lb))
		
		sample = (im, lb)#{'image': im, 'label': lb}
		return sample

def build_oasis(train=True, root='./', transform=None):
        
	if transform is None:
		transform = Compose([ToTensor()])

	return OasisDataset(train, root, transform)
