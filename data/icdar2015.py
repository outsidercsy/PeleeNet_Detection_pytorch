from .config import HOME######## from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pickle
ICDAR2015_ROOT = osp.join(HOME, "data/ICDAR2015_dataset")

ICDAR2015_CLASSES = (  # always index 0
    'text')

class ICDAR2015AnnotationTransform(object):###映射到[0,1]且label减1
    ####from quadrilateral to bbox    
    def __call__(self, target, width, height):
        x_tl = np.min(target[:,[0,2,4,6]],axis=1,keepdims=True) / width
        y_tl = np.min(target[:,[1,3,5,7]],axis=1,keepdims=True) / height
        x_br = np.max(target[:,[0,2,4,6]],axis=1,keepdims=True) / width 
        y_br = np.max(target[:,[1,3,5,7]],axis=1,keepdims=True) / height
        label = np.expand_dims(target[:,-1], axis=1) - 1####pkl中text index = 1,这里需要text index = 0,在match函数中会把0转变为1
        res = np.concatenate((x_tl, y_tl, x_br, y_br, label),axis=1)
        # print('target[0]',target[0])#########
        # print('res[0]',res[0])#########
        # print('width height',width,height)##########

        # print('res shape ',res.shape)########
        return res  # array[[xmin, ymin, xmax, ymax, label_ind], ... ]

class ICDAR2015Detection(data.Dataset):
    def __init__(self, root,
                 image_sets=None,
                 transform=None, target_transform=ICDAR2015AnnotationTransform(),
                 dataset_name='ICDAR2015'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        if image_sets == 'train':
            self._imgpath = osp.join(ICDAR2015_ROOT, 'ch4_training_images', '%s')
            with open(osp.join(ICDAR2015_ROOT, 'ICDAR2015_annotations_train_notcare.pkl'), "rb") as f:
                self._detections, self._image_file_names = pickle.load(f)
        else:
            self._imgpath = osp.join(ICDAR2015_ROOT, 'ch4_test_images', '%s')
            with open(osp.join(ICDAR2015_ROOT, 'ICDAR2015_annotations_test_notcare.pkl'), "rb") as f:
                self._detections, self._image_file_names = pickle.load(f)


 

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        
        # im_ = im.permute(1, 2, 0)[:,:,(2,1,0)].numpy()
        # im_[:,:,0] += 104
        # im_[:,:,1] += 117
        # im_[:,:,2] += 123
        # cv2.imwrite('visual/holo_test_{}_after.jpeg'.format(index), im_)##########
        # import pdb
        # pdb.set_trace()

        return im, gt

    def __len__(self):
        return len(self._image_file_names)

    def pull_item(self, index):
        image_file_name = self._image_file_names[index]
        target = self._detections[image_file_name]
    
        ####filter out not care when cal mean iou
        ############
        # target = target[target[:,-1] >= 0]###############
        # if target.shape[0] == 0:########
        #     print('all are not cared')##########
        #     target = np.array([[1]* 9],dtype=np.float)###########
        #     print('target is ',target)########
        ###########
 

        img = cv2.imread(self._imgpath % image_file_name)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:            
            # cv2.imwrite('visual/holo_test_{}_before.jpeg'.format(index), img)#########

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        

    def pull_image_file_name(self, index):  
        return self._image_file_names[index]


if __name__  == '__main__':
    dataset = ICDAR2015Detection('1', image_sets='train', transform=None)
    for i in range(10):
        im, gt= dataset[i]
        import pdb
        pdb.set_trace()