from .config import HOME######## from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pickle
ICDAR2013_ROOT = osp.join(HOME, "data/ICDAR2013_dataset")

ICDAR2013_CLASSES = (  # always index 0
    'text', )

class ICDAR2013AnnotationTransform(object):####映射到[0,1]且label减1
    ####from quadrilateral to bbox    
    def __call__(self, target, width, height):
        x_tl = np.expand_dims(target[:,0], axis=1) / width
        y_tl = np.expand_dims(target[:,1], axis=1) / height
        x_br = np.expand_dims(target[:,2], axis=1) / width
        y_br = np.expand_dims(target[:,3], axis=1) / height
        label = np.expand_dims(target[:,-1], axis=1) - 1####pkl中text index = 1,这里需要text index = 0,在match函数中会把0转变为1
        res = np.concatenate((x_tl, y_tl, x_br, y_br, label),axis=1)
        # print('target[0]',target[0])#########
        # print('res[0]',res[0])#########
        # print('width height',width,height)##########

        # print('res shape ',res.shape)########
        return res  # array[[xmin, ymin, xmax, ymax, label_ind], ... ]

class ICDAR2013Detection(data.Dataset):
    def __init__(self, root,
                 image_sets=None,
                 transform=None, target_transform=ICDAR2013AnnotationTransform(),
                 dataset_name='ICDAR2013'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        if image_sets == 'train':
            self._imgpath = osp.join(ICDAR2013_ROOT, 'Challenge2_Training_Task12_Images', '%s')
            with open(osp.join(ICDAR2013_ROOT, 'ICDAR2013_annotations_train.pkl'), "rb") as f:
                self._detections, self._image_file_names = pickle.load(f)
        else:
            self._imgpath = osp.join(ICDAR2013_ROOT, 'Challenge2_Test_Task12_Images', '%s')
            with open(osp.join(ICDAR2013_ROOT, 'ICDAR2013_annotations_test.pkl'), "rb") as f:
                self._detections, self._image_file_names = pickle.load(f)




    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self._image_file_names)

    def pull_item(self, index):
        image_file_name = self._image_file_names[index]
        target = self._detections[image_file_name]



        img = cv2.imread(self._imgpath % image_file_name)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            # target = np.array(target)
            # cv2.imwrite('visual/' + image_file_name + '_ori.jpg',img)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # cv2.imwrite('visual/' + image_file_name + '_after.jpg',img)

            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        

    def pull_image_file_name(self, index):  
        return self._image_file_names[index]


if __name__  == '__main__':
    dataset = ICDAR2013Detection('1', image_sets='train', transform=None)
    for i in range(10):
        im, gt= dataset[i]
        import pdb
        pdb.set_trace()