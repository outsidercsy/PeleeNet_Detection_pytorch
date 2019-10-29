from  .config import HOME#######from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pickle

import scipy.io as scio

SynthText_ROOT = osp.join(HOME, "data/SynthText")

SynthText_CLASSES = (  # always index 0
    'text')

class SynthTextAnnotationTransform(object):
    ####from quadrilateral to bbox    
    def __call__(self, target, width, height):
        x_tl = np.min(target[:,[0,2,4,6]],axis=1,keepdims=True) / width
        y_tl = np.min(target[:,[1,3,5,7]],axis=1,keepdims=True) / height
        x_br = np.max(target[:,[0,2,4,6]],axis=1,keepdims=True) / width
        y_br = np.max(target[:,[1,3,5,7]],axis=1,keepdims=True) / height
        label = np.expand_dims(target[:,-1], axis=1) - 1####pkl中text index = 1,这里需要text index = 0,在match函数中会把0转变为1
        res = np.concatenate((x_tl, y_tl, x_br, y_br, label),axis=1)
   
        return res  # array[[xmin, ymin, xmax, ymax, label_ind], ... ]

class SynthTextDetection(data.Dataset):
    def __init__(self, root,
                 image_sets=None,
                 transform=None, target_transform=SynthTextAnnotationTransform(),
                 dataset_name='SynthText'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        if image_sets == 'train':

            mat = scio.loadmat(osp.join(SynthText_ROOT, 'gt.mat'))
            self._image_file_names = mat['imnames'][0]     ####array 858750,
            ####assert len(self._image_file_names)
            self._detections = mat['wordBB'][0]            ####array 858750,  2x4xnumtext   possible 2x4

        else:
            raise Exception('synthtext only for training')




    def __getitem__(self, index):
        im, gt  = self.pull_item(index)

        return im, gt

    def __len__(self):
        return self._image_file_names.shape[0]

    def pull_item(self, index):
        image_file_name = self._image_file_names[index][0]
        
        if self._detections[index].shape == (2, 4):####process  exception 2x4
            target_without_label = self._detections[index][:,np.newaxis].swapaxes(0,2).reshape(-1,8)
        else:
            target_without_label = self._detections[index].swapaxes(0,2).reshape(-1,8)

        target = np.hstack((target_without_label, np.ones([target_without_label.shape[0],1],dtype=float)))



        img = cv2.imread(osp.join(SynthText_ROOT, image_file_name))


        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            # target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target
        

    def pull_image_file_name(self, index):  
        return self._image_file_names[index]

if __name__  == '__main__':
    dataset = SynthTextDetection('1', image_sets='train', transform=None)
    for i in range(10):
        im, gt= dataset[i*1000]
        cv2.imwrite('test2.jpg', np.array(im.permute(1,2,0)[:, :, (0, 1, 2)]) )
        import pdb
        pdb.set_trace()