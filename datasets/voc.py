import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
from collections import Counter
from PIL import Image


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    # color map for each category
    cmap = voc_cmap()

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        
        self.image_set = image_set
        base_dir = "VOCdevkit/VOC2012"
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Please check your data download.')
        
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.

            What you should do
            1. read the image (jpg) as an PIL image in RGB format.
            2. read the mask (png) as a single-channel PIL image.
            3. perform the necessary transforms on image & mask.
        """
        img_path = os.path.join(self.image_dir, f"img_{index}.jpg") 
        image = Image.open(img_path).convert('RGB')

        mask_path = os.path.join(self.mask_dir, f"mask_{index}.png")  
        mask = Image.open(mask_path).convert('L')  

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """Decode semantic mask to RGB image for visualization, using the color map"""
        mask = np.array(mask)
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        for label, color in enumerate(cls.cmap):
            mask_label = mask == label
            rgb_mask[mask_label] = color

        return Image.fromarray(rgb_mask)




