from .config import HOME######## from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pickle

import os

import time
HOLOV2_ROOT = osp.join(HOME, "data/holov2")

HOLOV2_CLASSES = ('car', 'sign', 'human', 'cone', 'led', 'bike')

# category_id_map = {'40': 0, '43': 1, '44': 2, '45': 3, '46': 4, '47': 5, '49': 6, '50': 7, '51': 8, '54': 9, '57': 10, '59': 11, '60': 12, '61': 13}


class HOLOV2AnnotationTransform(object):####映射到[0,1]且label减1
    ####from quadrilateral to bbox    
    def __call__(self, target, width, height):
        x_tl = np.expand_dims(target[:,0], axis=1) / width
        y_tl = np.expand_dims(target[:,1], axis=1) / height
        x_br = np.expand_dims(target[:,2], axis=1) / width
        y_br = np.expand_dims(target[:,3], axis=1) / height
        label = np.expand_dims(target[:,-1], axis=1) ####不需要-1  
        res = np.concatenate((x_tl, y_tl, x_br, y_br, label),axis=1)
        # print('target[0]',target[0])#########
        # print('res[0]',res[0])#########
        # print('width height',width,height)##########

        # print('res shape ',res.shape)########
        return res  # array[[xmin, ymin, xmax, ymax, label_ind], ... ]

class HOLOV2Detection(data.Dataset):
    def __init__(self, root,
                 image_sets=None,
                 transform=None, target_transform=HOLOV2AnnotationTransform(),
                 dataset_name='HOLOV2'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        if image_sets == 'train':
            print('use  holov2 training set')
            self._imgpath = osp.join(HOLOV2_ROOT, 'images', '%s')
            with open(osp.join(HOLOV2_ROOT, 'holov2_annotations_train.pkl'), "rb") as f:
                self._detections, self._image_file_names = pickle.load(f)
        else:
            self._imgpath = osp.join(HOLOV2_ROOT, 'images', '%s')
            with open(osp.join(HOLOV2_ROOT, 'holov2_annotations_test.pkl'), "rb") as f:
                self._detections, self._image_file_names = pickle.load(f)





    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)


        im_ = im.permute(1, 2, 0)[:,:,(2,1,0)].numpy()
        im_[:,:,0] += 104
        im_[:,:,1] += 117
        im_[:,:,2] += 123
        # cv2.imwrite('visual/holov2_test_{}_after.jpeg'.format(index), im_)##########
       

        return im, gt

    def __len__(self):
        return len(self._image_file_names)

    def pull_item(self, index):
        image_file_name = self._image_file_names[index]
        target = self._detections[image_file_name]

     
        
        img = cv2.imread(self._imgpath % image_file_name)
        height, width, channels = img.shape

   

        # img = cv2.resize(img,(600,600))####for fast data aug
        

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
          
            # cv2.imwrite('visual/holov2_test_{}_before.jpeg'.format(index), img)#########
            # t0 = time.time()########

            img = cv2.resize(img,(800,500))####for fast data aug
            
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])


            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            # t1 = time.time()########

            # print('transform duration',t1-t0)########

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        

    def pull_image_file_name(self, index):  
        return self._image_file_names[index]


if __name__  == '__main__':
    import sys
    sys.path.append("..")
    from utils.augmentations import SSDAugmentation
    from data import *
    cfg = holov2
    dataset = HOLOV2Detection('1', image_sets='train', transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    for i in range(100):
        im, gt= dataset[i]
        print(i)