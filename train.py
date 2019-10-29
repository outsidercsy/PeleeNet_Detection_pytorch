from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
# from ssd import build_ssd
from net import PeleeNet####
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data 
import numpy as np
import argparse
import datetime


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO','ICDAR2015','ICDAR2013','SynthText', 'HOLO', 'HOLOV2'],####
                    type=str, help='VOC or COCO or ICDAR2015' or 'ICDAR2013' or 'SynthText' or 'HOLO' or 'HOLOV2')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='pretrained_model_generate_from_VOCDetection.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,####origin is 1e-3
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')

parser.add_argument('--snapshot', default=10000, type=int,
                    help='Iteration for  saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        # if args.dataset_root == VOC_ROOT:
            # if not os.path.exists(COCO_ROOT):
            #     parser.error('Must specify dataset_root if specifying dataset')
            # print("WARNING: Using default COCO dataset_root because " +
            #       "--dataset_root was not specified.")
            # args.dataset_root = COCO_ROOT
        args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        # if args.dataset_root == COCO_ROOT:
        #     parser.error('Must specify dataset if specifying dataset_root')
        args.dataset_root = VOC_ROOT
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
                                                         

    elif args.dataset == 'HOLO':
        # if args.dataset_root == COCO_ROOT:
        #     parser.error('Must specify dataset if specifying dataset_root')
        args.dataset_root = HOLO_ROOT
        cfg = holo
        dataset = HOLODetection(root=args.dataset_root, image_sets='train',
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'HOLOV2':
        # if args.dataset_root == COCO_ROOT:
        #     parser.error('Must specify dataset if specifying dataset_root')
        args.dataset_root = HOLOV2_ROOT
        cfg = holov2
        dataset = HOLOV2Detection(root=args.dataset_root, image_sets='train',
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    elif args.dataset == 'ICDAR2015':####
        args.dataset_root = ICDAR2015_ROOT
        cfg = icdar2015
        dataset = ICDAR2015Detection(root=args.dataset_root, image_sets='train',
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    elif args.dataset == 'ICDAR2013':####
        args.dataset_root = ICDAR2013_ROOT
        cfg = icdar2013
        dataset = ICDAR2013Detection(root=args.dataset_root, image_sets='train',
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'SynthText':####
        args.dataset_root = SynthText_ROOT
        cfg = synthtext
        dataset = SynthTextDetection(root=args.dataset_root, image_sets='train',
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter( log_dir='runs/Pelee_test')

    pelee_net = PeleeNet('train', cfg)####
    net = pelee_net


# ##########
    # pelee_net = PeleeNet('train', cfg)####

    # pelee_net = pelee_net.cpu()
    # import pdb
    # pdb.set_trace()
    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(pelee_net, (304, 304), as_strings=True, print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)
    # pdb.set_trace()


    # from thop import profile
    # flops, params = profile(pelee_net, input_size=(1, 3, 304,304))
    # print('flops',flops)
    # print('params',params)

    # exit()
###########

    if args.cuda:
        net = torch.nn.DataParallel(pelee_net)
        cudnn.benchmark = True


###################

    ####random init first 
    print('Initializing weights...')
    pelee_net.apply(weights_init)####
    #     # initialize newly added layers' weights with xavier method
    #     pelee_net.extras.apply(weights_init)
    #     pelee_net.loc.apply(weights_init)
    #     pelee_net.conf.apply(weights_init)


    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        pelee_net.load_weights(args.resume)
    else:
        pelee_weights = torch.load(args.save_folder + args.basenet)
        print('Loading network except conf layers...')
        pelee_net.load_state_dict(pelee_weights, strict=False)

    if args.cuda:
        net = net.cuda()


#############
  


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training Pelee on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0


    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)




    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1####already moved it before update_vis_plot

            # update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,    ####
            #                 'append', epoch_size)

            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # images, targets = next(batch_iterator)####
        try:####

            images, targets = next(batch_iterator)
        except StopIteration:

            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)


        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images =images
            targets = [ann for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
 
  
        loc_loss += loss_l.data####origin is loc_loss += loss_l.data[0]
        conf_loss += loss_c.data####origin is   conf_loss += loss_c.data[0]



        if iteration % 100 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')####loss.data[0] to loss.data
            print(' \n ')########
        if args.visdom:
            # update_vis_plot(iteration, loss_l.data, loss_c.data,  ####loss_l.data[0]  loss_c.data[0]
            #                 iter_plot, epoch_plot, 'append')
            tb_writer.add_scalar('runs/Pelee_test',float(loss_l.data + loss_c.data), iteration)



        if iteration != 0 and iteration % args.snapshot == 0:
            print('Saving state, iter:', iteration)
            torch.save(pelee_net.state_dict(), 'weights/Pelee_' +str(iteration) + '.pth')

            os.system('python  holov2_eval.py  --trained_model weights/Pelee_' +str(iteration) + '.pth')


        # t2 = time.time()

        # print('t012',t0,t1,t2)####

    torch.save(pelee_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')
    os.system('python  holov2_eval.py  --trained_model'  + ' ' + args.save_folder  + args.dataset + '.pth'  )


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.kaiming_normal_(param)####


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        # m.bias.data.zero_()






if __name__ == '__main__':
    train()
