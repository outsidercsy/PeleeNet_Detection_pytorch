"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import ICDAR2013Detection, ICDAR2013AnnotationTransform, ICDAR2013_ROOT####
# from data import VOC_CLASSES as labelmap
import torch.utils.data as data

# from ssd import build_ssd
from net import PeleeNet####

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

from data import icdar2013

# if sys.version_info[0] == 2:
#     import xml.etree.cElementTree as ET
# else:
#     import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--icdar2013_root', default=ICDAR2013_ROOT,####
                    help='Location of ICDAR2013 root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

print('trained_model is', args.trained_model)####

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
# imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
# imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
#                           'Main', '{:s}.txt')
# YEAR = '2007'
# devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
# set_type = 'test'


# class Timer(object):
#     """A simple timer."""
#     def __init__(self):
#         self.total_time = 0.
#         self.calls = 0
#         self.start_time = 0.
#         self.diff = 0.
#         self.average_time = 0.

#     def tic(self):
#         # using time.time instead of time.clock because time time.clock
#         # does not normalize for multithreading
#         self.start_time = time.time()

#     def toc(self, average=True):
#         self.diff = time.time() - self.start_time
#         self.total_time += self.diff
#         self.calls += 1
#         self.average_time = self.total_time / self.calls
#         if average:
#             return self.average_time
#         else:
#             return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap, sorted_scores####


def test_net(save_folder, net, cuda, dataset, transform, top_k,####top_k is useless
             im_size=304, thresh=0.05):####
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    top_bboxes = {}####pred
    recs = {} ####gt

    # timers
    # _t = {'im_detect': Timer(), 'misc': Timer()}
    # output_dir = get_output_dir('ssd300_120000', set_type)
    # det_file = os.path.join(output_dir, 'detections.pkl')

    f_imagesetfile=open('imagesetfile','w')
    f_text_detection=open('text_detection','w')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        gt[:, 0] *= w
        gt[:, 2] *= w
        gt[:, 1] *= h
        gt[:, 3] *= h

        image_id = dataset.pull_image_file_name(i)####
        print('processing',image_id)########

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        # _t['im_detect'].tic()


        
        detections = net(x).data####detections is (num, self.num_classes, self.top_k, 5)
        # detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        dets = detections[0, 1, :]####j to 1
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.size(0) == 0:
            continue
        boxes = dets[:, 1:]
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        scores = dets[:, 0].cpu().numpy()
        # cls_dets = np.hstack((boxes.cpu().numpy(),
        #                         scores[:, np.newaxis])).astype(np.float32,
        #                                                         copy=False)
        cls_dets = np.hstack((  scores[:, np.newaxis],              ####first confidence then location
                    boxes.cpu().numpy()  )).astype(np.float32, copy=False)
        top_bboxes[image_id] = cls_dets


 
        ####debug
        if i < 50:
            debug = True
        else:
            debug = False
        if debug:
            imgpath = os.path.join('/workspace2/csy','data','ICDAR2013_dataset', 'Challenge2_Test_Task12_Images', '%s') 
            im_origin = cv2.imread(imgpath % image_id)

 
            keep_inds = (top_bboxes[image_id][:, 0] > 0.18)
            
            for score_and_bbox in top_bboxes[image_id][keep_inds]:
                score = score_and_bbox[0].astype(np.float32)
                bbox  = score_and_bbox[1:].astype(np.int32)
                cv2.rectangle(im_origin,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    [0,0,255], 1
                )
                cv2.putText(im_origin, str(score), 
                    (bbox[0], bbox[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1
                )
       
            debug_file = os.path.join(save_folder, image_id)
            cv2.imwrite(debug_file,im_origin) #########



########


        f_imagesetfile.write(image_id)  
        f_imagesetfile.write('\n')  

        box_num=top_bboxes[image_id].shape[0]

        for box_ind in range(box_num):

            f_text_detection.write(image_id)                        ####打印image_id
            f_text_detection.write(' ')

            f_text_detection.write(str(top_bboxes[image_id][box_ind,0]))####先打印confidence    
            f_text_detection.write(' ')
            for i in range(1,5):                                                 ####打印bbbox
                f_text_detection.write(str(top_bboxes[image_id][box_ind,i]))            
                f_text_detection.write(' ')
            f_text_detection.write('\n')
        

        objects=[]
        for i in range(gt.shape[0]):
            obj_struct = {}
            obj_struct['name'] = 'text'
            obj_struct['pose'] = ' '
            obj_struct['truncated'] = '0'
            obj_struct['difficult'] = '0'
            obj_struct['bbox']=list(gt[i][0:4])  
            objects.append(obj_struct)
        recs[image_id]=objects


    f_imagesetfile.close()
    f_text_detection.close()



    with open('annots.pkl','wb') as f:
        pickle.dump(recs, f)


    
    rec, prec, ap, sorted_scores =  voc_eval(detpath="{}_detection",####
             annopath='',
             imagesetfile='imagesetfile',
             classname='text',
             cachedir=os.getcwd(),
             ovthresh=0.5,
             use_07_metric=False)

    print('rec,prec,ap=',rec, prec, ap)

    F2_index = np.argmax(2*prec*rec/(prec+rec))

    F2 = np.max(2*prec*rec/(prec+rec))
    print('F2_corresponding score = ', sorted_scores[F2_index])
    print('F2 coresponding rec prec = ', rec[F2_index], prec[F2_index])
    print('F2=',F2)

  
if __name__ == '__main__':
    # load net
    # num_classes = 1+1                 # +1 for background
    net = PeleeNet('test', icdar2013)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    


    net.eval()

    print('Finished loading model!')
    # load data
    dataset = ICDAR2013Detection(args.icdar2013_root, 'test',
                           BaseTransform(304, dataset_mean),####
                           ICDAR2013AnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(304, dataset_mean), args.top_k, 304,####
             thresh=args.confidence_threshold)
