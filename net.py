
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torch.autograd import Variable
from layers import *
from data import voc, coco, icdar2015, icdar2013

from torchsummary import summary

# class Bypass(nn.Module):
#     def __init__(self):
#         super(Bypass, self).__init__()

#         self.layer1 = conv_bn_relu(nin=3, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
#         self.layer2 = conv_bn_relu(nin=32, nout=64, kernel_size=3, stride=2, padding=1, bias=False)
#         self.layer3 = conv_bn_relu(nin=64, nout=128, kernel_size=3, stride=2, padding=1, bias=False)
#         self.layer4 = conv_bn_relu(nin=128, nout=256, kernel_size=3, stride=2, padding=1, bias=False)
#         self.layer5 = conv_bn_relu(nin=256, nout=512, kernel_size=1, stride=1, padding=0, bias=False)

#     def forward(self, x): 
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x

class ResBlock(nn.Module):
    def __init__(self, nin):
        super(ResBlock, self).__init__()
        self.left_conv1 = conv_bn_relu(nin=nin, nout=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.left_conv2 = conv_bn_relu(nin=128, nout=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.left_conv3 = conv_bn_relu(nin=128, nout=256, kernel_size=1, stride=1, padding=0, bias=False)

        self.right_conv = conv_bn_relu(nin=nin, nout=256, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        left = self.left_conv1(x)
        left = self.left_conv2(left)
        left = self.left_conv3(left)

        right = self.right_conv(x)

        out = left + right
        return out

class conv_bn_relu(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        return out


# class Transition_layer(nn.Sequential):
#     def __init__(self, nin, theta=1):    
#         super(Transition_layer, self).__init__()
        
#         self.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=int(nin*theta), kernel_size=1, stride=1, padding=0, bias=False))
#         self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))####ceil_mode=True


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        
        self.conv_3x3_first = conv_bn_relu(nin=3, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.conv_1x1_left = conv_bn_relu(nin=32, nout=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3x3_left = conv_bn_relu(nin=16, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.max_pool_right = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv_1x1_last = conv_bn_relu(nin=64, nout=32, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x=x.to('cpu')######

        out_first = self.conv_3x3_first(x)
        
        out_left = self.conv_1x1_left(out_first)
        out_left = self.conv_3x3_left(out_left)
        
        out_right = self.max_pool_right(out_first)
        
        out_middle = torch.cat((out_left, out_right), 1)
        
        out_last = self.conv_1x1_last(out_middle)
                
        return out_last


class dense_layer(nn.Module):
    def __init__(self, nin, growth_rate, drop_rate=0.2, bottleneck_width=4):    ####
        super(dense_layer, self).__init__()
        
        inter_channel = growth_rate * 2 * bottleneck_width // 4####

        self.dense_left_way = nn.Sequential()
        
        self.dense_left_way.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=inter_channel, kernel_size=1, stride=1, padding=0, bias=False))####
        self.dense_left_way.add_module('conv_3x3', conv_bn_relu(nin=inter_channel, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))####
                
        self.dense_right_way = nn.Sequential()
        
        self.dense_right_way.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=inter_channel, kernel_size=1, stride=1, padding=0, bias=False))####
        self.dense_right_way.add_module('conv_3x3_1', conv_bn_relu(nin=inter_channel, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))####
        self.dense_right_way.add_module('conv_3x3 2', conv_bn_relu(nin=growth_rate//2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.drop_rate = drop_rate
      
    def forward(self, x):
        left_output = self.dense_left_way(x)
        right_output = self.dense_right_way(x)

        if self.drop_rate > 0:
            left_output = F.dropout(left_output, p=self.drop_rate, training=self.training)
            right_output = F.dropout(right_output, p=self.drop_rate, training=self.training)
            
        dense_layer_output = torch.cat((x, left_output, right_output), 1)
                
        return dense_layer_output

class DenseBlock(nn.Sequential):
     def __init__(self, nin, num_dense_layers, growth_rate, drop_rate=0.0, bottleneck_width=4):####
        super(DenseBlock, self).__init__()
                            
        for i in range(num_dense_layers):
            nin_dense_layer = nin + growth_rate * i
            self.add_module('dense_layer_%d' % i, dense_layer(nin=nin_dense_layer, growth_rate=growth_rate, drop_rate=drop_rate, bottleneck_width=bottleneck_width))####


class PeleeNet(nn.Module):
    def __init__(self, phase, cfg):####
        super(PeleeNet, self).__init__()
        
          
        num_dense_layers=[3,4,8,6]
        growth_rate=32
        nin_transition_layer = 32
        theta=1
        drop_rate=0.0

        self.phase = phase

        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        # if num_classes == 21:
        #     self.cfg = voc
        # elif num_classes == 2:
        #     self.cfg = icdar2015
        # else:
        #     self.cfg = coco
        


        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)


        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)


        assert len(num_dense_layers) == 4
        
        self.StemBlock = StemBlock()

        # self.Bypass = Bypass()


        ####stage1
        self.DenseBlock_1 = DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[0], growth_rate=growth_rate, drop_rate=0.0, bottleneck_width=1)####
        nin_transition_layer +=  num_dense_layers[0] * growth_rate
        self.trans_conv_1 = conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer*theta), kernel_size=1, stride=1, padding=0, bias=False)
        self.trans_ave_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        ####stage2
        self.DenseBlock_2 = DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[1], growth_rate=growth_rate, drop_rate=0.0, bottleneck_width=2)####
        nin_transition_layer +=  num_dense_layers[1] * growth_rate
        self.trans_conv_2 = conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer*theta), kernel_size=1, stride=1, padding=0, bias=False)
        self.trans_ave_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        ####stage3
        self.DenseBlock_3 = DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[2], growth_rate=growth_rate, drop_rate=0.0, bottleneck_width=4)####
        nin_transition_layer +=  num_dense_layers[2] * growth_rate
        self.trans_conv_3 = conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer*theta), kernel_size=1, stride=1, padding=0, bias=False)
        self.trans_ave_3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        ####stage4
        self.DenseBlock_4 = DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[3], growth_rate=growth_rate, drop_rate=0.0, bottleneck_width=4)####
        nin_transition_layer +=  num_dense_layers[3] * growth_rate
        self.trans_conv_4 = conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer*theta), kernel_size=1, stride=1, padding=0, bias=False)

        

        self.fe1_1 = conv_bn_relu(nin=704, nout=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.fe1_2 = conv_bn_relu(nin=256, nout=256, kernel_size=3, stride=2, padding=1, bias=False)####output  w=5 h=5 
        self.fe2_1 = conv_bn_relu(nin=256, nout=128, kernel_size=1, stride=1, padding=0, bias=False)####output  w=5 h=5 
        self.fe2_2 = conv_bn_relu(nin=128, nout=256, kernel_size=3, stride=1, padding=0, bias=False)####output  w=3 h=3  
        self.fe3_1 = conv_bn_relu(nin=256, nout=128, kernel_size=1, stride=1, padding=0, bias=False)####output  w=3 h=3  
        self.fe3_2 = conv_bn_relu(nin=128, nout=256, kernel_size=3, stride=1, padding=0, bias=False)####output  w=1 h=1  


        # self.ResBlocks = nn.ModuleList([ResBlock(nin=512), ResBlock(nin=512), ResBlock(nin=704), 
        #                     ResBlock(nin=256), ResBlock(nin=256), ResBlock(nin=256) ])
        self.ResBlocks = nn.ModuleList([ResBlock(nin=512), ResBlock(nin=512), ResBlock(nin=704), 
                    ResBlock(nin=256), ResBlock(nin=256), ResBlock(nin=256) ])

        self.loc_layers = []
        self.conf_layers = []
        # anchor_nums = [8, 8, 8, 8, 8, 8]####
        anchor_nums = self.cfg['anchor_nums']#[4, 6, 6, 6, 4, 4]####
        print('anchor_nums', anchor_nums)##########
        for i in range(len(anchor_nums)):
            self.loc_layers += [nn.Conv2d(256, anchor_nums[i] * 4, kernel_size=1, padding=0, bias=False)]####kernel_size=3, padding=1
            self.conf_layers += [nn.Conv2d(256, anchor_nums[i] * self.num_classes, kernel_size=1, padding=0, bias=False)]####kernel_size=3, padding=1

        self.loc_layers = nn.ModuleList(self.loc_layers)
        self.conf_layers = nn.ModuleList(self.conf_layers)


        
    def forward(self, x):
        # bypass_feature = self.Bypass(x)
        x = self.StemBlock(x)

        sources = list()
        loc = list()
        conf = list()

        x = self.DenseBlock_1(x)
        x = self.trans_conv_1(x)
        x = self.trans_ave_1(x)

        x = self.DenseBlock_2(x)
        x = self.trans_conv_2(x)
        # sources.append(x)########use_38
        x = self.trans_ave_2(x)

        x = self.DenseBlock_3(x)
        x = self.trans_conv_3(x)

        # x = x + bypass_feature####

        sources.append(x)


        sources.append(x)
        x = self.trans_ave_3(x)

        x = self.DenseBlock_4(x)
        x = self.trans_conv_4(x)
        sources.append(x)

        x = self.fe1_1(x)
        x = self.fe1_2(x)
        sources.append(x)
        x = self.fe2_1(x)
        x = self.fe2_2(x)
        sources.append(x)
        x = self.fe3_1(x)
        x = self.fe3_2(x)
        sources.append(x)


        # y = self.ResBlocks[1](sources[1])
        # z = self.loc_layers[1](y)


        for (x, r, l, c) in zip(sources, self.ResBlocks, self.loc_layers, self.conf_layers):

            x = r(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
    

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),        # conf preds
                self.priors.type(type(x.data)) )                 # default boxes  both is torch.tensor,no need for type transform


        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output



    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
       
            self.load_state_dict(  torch.load(base_file,map_location=lambda storage, loc: storage) )
    

            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


    # def init_weights(self):



if __name__ == '__main__':
    p = PeleeNet(cfg, phase='train')
    input = torch.ones(2, 3, 304, 304)
    output = p(input)
    
    # p.to('cpu')
    # summary(p, ( 3, 304, 304))
