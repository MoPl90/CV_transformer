import torch
import os
import numpy as np
from PIL import Image
import nibabel as nib
import random
from torchvision.datasets import VisionDataset
from pathlib import Path
from sklearn.model_selection import train_test_split


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
		im = np.load(os.path.join(self.root + '/' + self.ext_im, self.fnames[idx])).astype(np.uint8)
		lb = np.load(os.path.join(self.root + '/' + self.ext_lb, self.fnames[idx])).astype(np.float32)

		  
		seed = np.random.randint(2147483647) # make a seed with numpy generator 
		random.seed(seed) # apply this seed to img tranfsorms
		torch.manual_seed(seed) # needed for torchvision 0.7
		if self.transform is not None:
			im = self.transform(im)
			#reset the seed	
			random.seed(seed)
			torch.manual_seed(seed)
			lb = self.transform(lb)
		
		sample = (im, lb)#{'image': im, 'label': lb}
		return sample

def build_oasis(train=True, root='./', transform=None):
        
	return OasisDataset(train, root, transform)


class MPRAGEDataset(VisionDataset):
	
	def __init__(self, root_dir, train, train_size=0.8, transform=None):
		super().__init__(root_dir)
		self.ext_im = 'MPRAGE'
		self.ext_lb = 'MPRAGE_seg_lowres'

		fnames = os.listdir(os.path.join(self.root, self.ext_im))
		if train:
			self.fnames, _ = train_test_split(fnames, train_size = train_size)
		else:
			_, self.fnames = train_test_split(fnames, train_size = train_size)
			
		self.numfiles = len(self.fnames)
		self.transform = transform
		
		assert len(os.listdir(os.path.join(self.root, self.ext_lb))) == len(fnames), "Corrupt image/label folders"

	def __len__(self):
		return self.numfiles
		
		
	def __getitem__(self, idx):
		im = self._load_nifti(os.path.join(self.root + '/' + self.ext_im, self.fnames[idx])).astype(np.float)
		lb = self._load_nifti(os.path.join(self.root + '/' + self.ext_lb, self.fnames[idx]), label=True).astype(np.int)

		  
		seed = np.random.randint(2147483647) # make a seed with numpy generator 
		random.seed(seed) # apply this seed to img tranfsorms
		torch.manual_seed(seed) # needed for torchvision 0.7
		if self.transform is not None:
			im,lb = self.transform((im, lb))
		
		sample = (im, lb)#{'image': im, 'label': lb}
		return sample

	@staticmethod
	def _load_nifti(path, clipping_percentile=[0., 99.5], label=False):
		nii = nib.load(path)
		im = nib.as_closest_canonical(nii).get_fdata()
		p = np.percentile(im, clipping_percentile[1])

		if len(im.shape) < 4:
			im = im[np.newaxis,:, :, :]

		
		if not label:
			im = np.clip(im, clipping_percentile[0], p)
			im /= p
			
		return im


def build_oasis(train=True, root='./', transform=None):
        
	return OasisDataset(train, root, transform)

def build_mprage(root='./', train=True, train_size=0.8, transform=None):
        
	return MPRAGEDataset(root, train, train_size, transform)
