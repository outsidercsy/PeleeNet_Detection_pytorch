# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("/workspace2/csy")####origin is "~"

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)####bgr

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 150000, 200000),
    'max_iter': 200000,
    'feature_maps': [19, 19, 10, 5, 3, 1],
    'min_dim': 304,
    'steps': [16, 16, 30, 61, 101, 304],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
    'anchor_nums': [4, 6, 6, 6, 4, 4],####
    'flip': True,####

}

holo = {
    'num_classes': 15,
    'lr_steps': (30000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [19, 19, 10, 5, 3, 1],
    'min_dim': 304,
    'steps': [16, 16, 30, 61, 101, 304],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'HOLO',
    'anchor_nums': [4, 6, 6, 6, 4, 4],####
    'flip': True,####

}

holov2 = {
    'num_classes': 7,
    'lr_steps': (100000, 150000, 200000),
    'max_iter': 200000,
    'feature_maps': [19, 19, 10, 5, 3, 1],
    'min_dim': 304,
    'steps': [16, 16, 30, 61, 101, 304],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'HOLOV2',
    'anchor_nums': [4, 6, 6, 6, 4, 4],####
    'flip': True,####

}



coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [19, 19, 10, 5, 3, 1],
    'min_dim': 304,
    'steps': [16, 16, 30, 61, 101, 304],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
    'anchor_nums': [4, 6, 6, 6, 4, 4],####
    'flip': True,####

}

icdar2015 = {                     ####
    'num_classes': 2,
    'lr_steps': (30000, 70000, 150000),
    'max_iter': 150000,
    'feature_maps': [19, 19, 10, 5, 3, 1],
    'min_dim': 304,
    'steps': [16, 16, 30, 61, 101, 304], 
    # 'min_sizes': [30, 60, 111, 162, 213, 264],
    # 'max_sizes': [60, 111, 162, 213, 264, 315],
    # 'aspect_ratios': [[2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10]],

    ###0.34 ave_iou
    'min_sizes': [15, 30, 90, 162, 213, 264],
    'max_sizes': [30, 90, 162, 213, 264, 315],
    'aspect_ratios': [[2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5]],   

    ####add anchor   0.44 ave_iou
    # 'min_sizes': [30, 60, 111, 162, 213, 264],
    # 'max_sizes': [60, 111, 162, 213, 264, 315],   
    # 'aspect_ratios': [[2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5]],   

    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'ICDAR2015',
    'anchor_nums': [8, 8, 8, 8, 8, 8],####
    'flip': True,####

}

icdar2013 = {                     ####
    'num_classes': 2,
    'lr_steps': (0, 15000),
    'max_iter': 20010,
    'feature_maps': [19, 19, 10, 5, 3, 1],
    'min_dim': 304,
    'steps': [16, 16, 30, 61, 101, 304],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10]],
    # 'min_sizes': [15, 30, 90, 162, 213, 264],
    # 'max_sizes': [30, 90, 162, 213, 264, 315],
    # 'aspect_ratios': [[2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5]],      
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'ICDAR2013',
    'anchor_nums': [7, 7, 7, 7, 7, 7],####
    'flip': False,####
    # 'anchor_nums': [11, 11, 11, 11, 11, 11],####
    # 'flip': True,####

}

synthtext = {                     
    ####pretrain for icdar2013
    # 'num_classes': 2,
    # 'lr_steps': (30000, 60000),
    # 'max_iter': 50010,
    # 'feature_maps': [19, 19, 10, 5, 3, 1],
    # 'min_dim': 304,
    # 'steps': [16, 16, 30, 61, 101, 304],
    # # 'min_sizes': [15, 30, 90, 162, 213, 264],
    # # 'max_sizes': [30, 90, 162, 213, 264, 315],
    # # 'aspect_ratios': [[2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5]], 
    # 'min_sizes': [30, 60, 111, 162, 213, 264],
    # 'max_sizes': [60, 111, 162, 213, 264, 315],
    # 'aspect_ratios': [[2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10]],       
    # 'variance': [0.1, 0.2],
    # 'clip': True,
    # 'name': 'SynthText',
    # 'anchor_nums': [7, 7, 7, 7, 7, 7],####
    # 'flip': False,####  1/ar  needed or not


    ####pretrain for icdar2015
    'num_classes': 2,
    'lr_steps': (30000, 60000),
    'max_iter': 50010,
    'feature_maps': [19, 19, 10, 5, 3, 1],
    'min_dim': 304,
    'steps': [16, 16, 30, 61, 101, 304],
    'min_sizes': [15, 30, 90, 162, 213, 264],
    'max_sizes': [30, 90, 162, 213, 264, 315],
    'aspect_ratios': [[2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5], [2,3,5]],   
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'SynthText',
    'anchor_nums': [8, 8, 8, 8, 8, 8],####
    'flip': True,####

}