import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
from torch import autocast

from PTDataset import TreeStructuredDataset

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
import seaborn as sn
import pandas as pd
import numpy as np
import timm

from datetime import timedelta

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


import signal
def ctrl_c_handler(signum, frame):
    msg = "Stopping..."
    print(msg, end="", flush=True, file=sys.stderr)
    if not os.path.exists('.STOP'):
        with open('.STOP', "w") as f:
            f.write("\n")
            f.close()
signal.signal(signal.SIGINT, ctrl_c_handler)

starting_start = time.time()

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.append('efficient')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--structure', metavar='structure', default='bcnn',
                    choices=('bcnn', 'rbcnn', 'hmlc', 'bcnn2', 'rbcnn2', 'none', ''),
                    help='hierarchy wrapper: ' +
                    ' | '.join(('bcnn', 'hmlc', '')) +
                    ' (default: bcnn)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--train-labels', default='', type=str,
                    help='Force training order label1,label2,... (default: none)')
parser.add_argument('--test-labels',
                    default='O/O/OTHERS,A/1/View_A2C,A/1/View_A3C,A/2/View_A4C,A/2/View_A5C,'
                    'P/L/View_PLAX,P/L/View_PLAX_RV,P/S/View_PSAX_AV,P/S/View_PSAX_TV_RV', type=str,
                    help='Force test order label1,label2,...')
parser.add_argument('--loss-weights',
                    default='0:0.98,0.01,0.01/13:0.1,0.8,0.1/23:0.1,0.2,0.7/33:0.0,0.0,1.0',
                    type=str, help='Block based loss weights.')
parser.add_argument('--learning-weights',
                    default='0:1.0,1.0,1.0',
                    type=str, help='Block based learning weights')
parser.add_argument('--filters', default='', type=str,
                    help='Input label filter per epoch <starting_epoch>:<filtred_label>,... (default: none)')
parser.add_argument('--suffix', default='', type=str,
                    help='Suffix added to the arch on all output files (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='test model on testing set')
parser.add_argument('--residual', dest='residual', action='store_true',
                    help='Use parents labels as residuals')
parser.add_argument('--reverse-labels', dest='reverse', action='store_true',
                    help='Reverse labels order')
parser.add_argument('--masked-learning', dest='maskedLearning', action='store_true',
                    help='Use parents labels to mask children losses')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0
best_epoch = 0

class HCNN(nn.Module):
    def __init__(self, arch, pretrained, num_classes, children,
                 residual=False, reverseLabels=False, maskedLearning=False,
                 lossWeights={0: [0.98, 0.01, 0.01],
                              13: [0.1, 0.8, 0.1],
                              23: [0.1, 0.2, 0.7],
                              33: [0.0, 0.0, 1.0]},
                 learningWeights={0: [0.98, 1.0, 1.0],
                                  13: [0.1, 0.8, 1.0],
                                  23: [0.1, 0.2, 0.7],
                                  33: [0.0, 0.0, 1.0]},
                 structure='bcnn',
                 suffix=''):
        super(HCNN,self).__init__()

        self.structure = structure
        self.name = arch+suffix
        self.num_classes = num_classes
        self._children = children
        
        if structure != '':
            self.name = structure+'-'+self.name
        
        if residual and ( structure == 'bcnn' or structure == 'bcnn2' ):
            self.name = 'r'+self.name
        else:
            residual = False
        
        self.residual = residual
        self.maskedLearning = maskedLearning
        self.reverseLabels = reverseLabels
        
        self.labelSeq = [x for x in range(len(num_classes))]
        if self.reverseLabels:
            self.labelSeq = [x for x in reversed(self.labelSeq)]
        
        self.lossWeights = []
        lastWeights = lossWeights[0]
        for key in lossWeights:
            for x in range(len(self.lossWeights), key):
                self.lossWeights.append(lastWeights)
            lastWeights = lossWeights[key]
        self.lossWeights.append(lastWeights)
        
        self.learningWeights = []
        lastWeights = learningWeights[0]
        for key in learningWeights:
            for x in range(len(self.learningWeights), key):
                self.learningWeights.append(lastWeights)
            lastWeights = learningWeights[key]
        self.learningWeights.append(lastWeights)
        
        if arch == "efficient":
          backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained)
          
        elif pretrained:
          print("=> using pre-trained model '{}'".format(self.name), file=sys.stderr)
          backbone = models.__dict__[arch](pretrained=True)
        else:
          print("=> creating model '{}'".format(self.name), file=sys.stderr)
          backbone = models.__dict__[arch]()
        
        if structure == 'bcnn':
            if arch == 'vgg16':
                self.blockA = nn.Sequential(*list(backbone.features.children())[0:10])
                ## Level-1 classifier after second conv block
                self.branch_one = nn.Sequential(nn.Linear(128*56*56, 256),
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, 256), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, num_classes[self.labelSeq[0]]))
                self.blockB = nn.Sequential(*list(backbone.features.children())[10:17])
                ## Level-2 classifier after third conv block
                self.branch_two = nn.Sequential(nn.Linear(256*28*28
                                                          + (num_classes[self.labelSeq[0]] if self.residual else 0),
                                                          1024), 
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, 1024), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, num_classes[self.labelSeq[1]]))
                self.blockC = nn.Sequential(*list(backbone.features.children())[17:])
                ## Level-3 classifier after fifth conv block
                self.branch_three = nn.Sequential(nn.Linear(512*7*7
                                                            + (num_classes[self.labelSeq[1]] if self.residual else 0),
                                                            4096), 
                                                  nn.ReLU(),
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[2]]))
                
            elif arch == 'vgg19':
                self.blockA = nn.Sequential(*list(backbone.features.children())[0:10])
                self.branch_one = nn.Sequential(nn.Linear(128*56*56, 256),
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, 256), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, num_classes[self.labelSeq[0]]))
                self.blockB = nn.Sequential(*list(backbone.features.children())[10:19])
                self.branch_two = nn.Sequential(nn.Linear(256*28*28
                                                          + (num_classes[self.labelSeq[0]] if self.residual else 0),
                                                          1024), 
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, 1024), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, num_classes[self.labelSeq[1]]))
                self.blockC = nn.Sequential(*list(backbone.features.children())[19:])
                self.branch_three = nn.Sequential(nn.Linear(512*7*7
                                                            + (num_classes[self.labelSeq[1]] if self.residual else 0),
                                                            4096), 
                                                  nn.ReLU(), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[2]]))
                
            elif arch == 'densenet121':
                self.blockA = nn.Sequential(*list(backbone.features.children())[0:6])
                self.branch_one = nn.Sequential(nn.Linear(128*28*28, 256),
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, 256), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, num_classes[self.labelSeq[0]]))
                self.blockB = nn.Sequential(*list(backbone.features.children())[6:8])
                self.branch_two = nn.Sequential(nn.Linear(256*14*14
                                                          + (num_classes[self.labelSeq[0]] if self.residual else 0),
                                                          1024), 
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, 1024), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, num_classes[self.labelSeq[1]]))
                self.blockC = nn.Sequential(*list(backbone.features.children())[8:])
                self.branch_three = nn.Sequential(nn.Linear(1024*7*7
                                                            + (num_classes[self.labelSeq[1]] if self.residual else 0),
                                                            4096), 
                                                  nn.ReLU(), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[2]]))
              
            elif arch == 'mobilenet_v2':
                self.blockA = nn.Sequential(*list(backbone.features.children())[0:7])
                self.branch_one = nn.Sequential(nn.Linear(32*28*28, 256),
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, 256), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, num_classes[self.labelSeq[0]]))
                self.blockB = nn.Sequential(*list(backbone.features.children())[7:14])
                self.branch_two = nn.Sequential(nn.Linear(96*14*14
                                                          + (num_classes[self.labelSeq[0]] if self.residual else 0),
                                                          1024), 
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, 1024), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, num_classes[self.labelSeq[1]]))
                self.blockC = nn.Sequential(*list(backbone.features.children())[14:19])
                self.branch_three = nn.Sequential(nn.Linear(1280*7*7
                                                            + (num_classes[self.labelSeq[1]] if self.residual else 0),
                                                            4096), 
                                                  nn.ReLU(), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[2]]))
                
            elif arch == 'resnet18':
                self.blockA = nn.Sequential(*list(backbone.children())[0:6])
                self.branch_one = nn.Sequential(nn.Linear(128*28*28, 256),
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, 256), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, num_classes[self.labelSeq[0]]))
                self.blockB = nn.Sequential(*list(backbone.children())[6:7])
                self.branch_two = nn.Sequential(nn.Linear(256*14*14
                                                          + (num_classes[self.labelSeq[0]] if self.residual else 0),
                                                          1024), 
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, 1024), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, num_classes[self.labelSeq[1]]))
                self.blockC = nn.Sequential(*list(backbone.children())[7:9])
                self.branch_three = nn.Sequential(nn.Linear(512*1*1
                                                            + (num_classes[self.labelSeq[1]] if self.residual else 0),
                                                            4096), 
                                                  nn.ReLU(), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[2]]))
                
            elif arch == 'resnet50':
                self.blockA = nn.Sequential(*list(backbone.children())[0:6])
                self.branch_one = nn.Sequential(nn.Linear(512*28*28, 256),
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, 256), 
                                                nn.BatchNorm1d(256), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(256, num_classes[self.labelSeq[0]]))
                self.blockB = nn.Sequential(*list(backbone.children())[6:7])
                self.branch_two = nn.Sequential(nn.Linear(1024*14*14
                                                          + (num_classes[self.labelSeq[0]] if self.residual else 0),
                                                          1024), 
                                                nn.ReLU(), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, 1024), 
                                                nn.BatchNorm1d(1024), 
                                                nn.Dropout(0.5), 
                                                nn.Linear(1024, num_classes[self.labelSeq[1]]))
                self.blockC = nn.Sequential(*list(backbone.children())[7:9])
                self.branch_three = nn.Sequential(nn.Linear(2048*1*1
                                                            + (num_classes[self.labelSeq[1]] if self.residual else 0),
                                                            4096), 
                                                  nn.ReLU(), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[2]]))
                
            elif arch == 'efficient':
                raise NotImplementedError
                
            else:
                raise NotImplementedError
        elif structure == 'bcnn2':
            if arch == 'vgg16':
                self.block = backbone.features
                self.branch_one = nn.Sequential(nn.Linear(512*7*7, 4096), 
                                                  nn.ReLU(),
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[0]]))
                self.branch_two = nn.Sequential(nn.Linear(512*7*7
                                                          + (num_classes[self.labelSeq[0]] if self.residual else 0),
                                                          4096), 
                                                  nn.ReLU(),
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[1]]))
                self.branch_three = nn.Sequential(nn.Linear(512*7*7
                                                            + (num_classes[self.labelSeq[1]] if self.residual else 0),
                                                            4096), 
                                                  nn.ReLU(),
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, 4096), 
                                                  nn.BatchNorm1d(4096), 
                                                  nn.Dropout(0.5), 
                                                  nn.Linear(4096, num_classes[self.labelSeq[2]]))
            else:
                raise NotImplementedError
        else:
            outputFeature = sum(num_classes) if structure == 'hmlc' else num_classes[-1]
            if arch == 'vgg16' or arch == 'vgg19':
                backbone.classifier = torch.nn.Sequential(
                    torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(p=0.5, inplace=False),
                    torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(p=0.5, inplace=False),
                    torch.nn.Linear(in_features=4096, out_features=outputFeature, bias=True)
                )
            elif arch == 'densenet121':
                backbone.classifier = torch.nn.Linear(in_features=1024, out_features=outputFeature, bias=True)
            elif arch == 'mobilenet_v2':
                backbone.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(in_features=1280, out_features=outputFeature, bias=True)
                )
            elif arch == 'resnet18' or arch == 'resnet50':
                backbone.fc = torch.nn.Linear(in_features=2048, out_features=outputFeature, bias=True)
            elif arch == 'efficient':
                backbone.classifier = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(output_size=1),
                    torch.nn.Flatten(),
                    torch.nn.Dropout(p=0.2, inplace=False),
                    torch.nn.Linear(in_features=1280, out_features=outputFeature, bias=True)
                )
            else:
                raise NotImplementedError
            self.backbone = backbone
    
    def forward(self,x):
        if self.structure == 'bcnn':
            x = self.blockA(x)
            bch_one = x.view(x.size(0), -1)
            bch_one = self.branch_one(bch_one)
            x = self.blockB(x)
            bch_two = x.view(x.size(0), -1)
            if self.residual:
                bch_two = torch.cat((bch_two, bch_one), 1)
            bch_two = self.branch_two(bch_two)
            x = self.blockC(x)
            bch_three = x.view(x.size(0), -1)
            if self.residual:
                bch_three = torch.cat((bch_three, bch_two), 1)
            bch_three = self.branch_three(bch_three)
            if self.reverseLabels:
                return (bch_three, bch_two, bch_one)
            return (bch_one, bch_two, bch_three)
        if self.structure == 'bcnn2':
            x = self.block(x)
            bch_one = x.view(x.size(0), -1)
            bch_one = self.branch_one(bch_one)
            bch_two = x.view(x.size(0), -1)
            if self.residual:
                bch_two = torch.cat((bch_two, bch_one), 1)
            bch_two = self.branch_two(bch_two)
            bch_three = x.view(x.size(0), -1)
            if self.residual:
                bch_three = torch.cat((bch_three, bch_two), 1)
            bch_three = self.branch_three(bch_three)
            if self.reverseLabels:
                return (bch_three, bch_two, bch_one)
            return (bch_one, bch_two, bch_three)
        elif self.structure == 'hmlc':
            y = self.backbone(x)
            return (y[:, :self.num_classes[0]],
                    y[:, self.num_classes[0]:sum(self.num_classes[:2])],
                    y[:, sum(self.num_classes[:2]):sum(self.num_classes[:3])])
        else:
            return (torch.zeros((x.size(0), self.num_classes[0]), device=torch.device('cuda')),
                    torch.zeros((x.size(0), self.num_classes[1]), device=torch.device('cuda')),
                    self.backbone(x))
    
    def currentLossWeights(self, epoch):
        if epoch < len(self.lossWeights):
            return self.lossWeights[epoch]
        else:
            return self.lossWeights[len(self.lossWeights)-1]
    
    def currentLearningWeights(self, epoch):
        if epoch < len(self.learningWeights):
            return self.learningWeights[epoch]
        else:
            return self.learningWeights[len(self.learningWeights)-1]
    
    def loss(self, out, label, criterions, epoch):
        currentWeights = self.currentLossWeights(epoch)
        prediction = out
        if self.maskedLearning:
            for l in range(len(label)-1):
                for b in range(len(label[l])):
                    for i in range(len(label[l][b])):
                        if label[l][b][i] == 0:
                            for c in self._children[l][i]:
                                prediction[l+1][b][c] = 0.
        losses = []
        total_loss = 0
        for i in range(len(criterions)):
            losses.append(criterions[i](prediction[i], label[i]))
            total_loss += currentWeights[i] * losses[i]
        
        return (total_loss, losses)
    
    def initialParameters(self):
        if self.structure == 'bcnn':
            return [{'params': self.blockA.parameters()},
                    {'params': self.blockB.parameters()},
                    {'params': self.blockC.parameters()},
                    {'params': self.branch_one.parameters()},
                    {'params': self.branch_two.parameters()},
                    {'params': self.branch_three.parameters()}]
        elif self.structure == 'bcnn2':
            return [{'params': self.block.parameters()},
                    {'params': self.branch_one.parameters()},
                    {'params': self.branch_two.parameters()},
                    {'params': self.branch_three.parameters()}]
        else:
            return self.backbone.parameters()
                
    def adjust_learning_rate(self, optimizer, epoch, initial_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        global_lr = initial_lr * (0.1 ** (epoch // 30))
        lr_perblock = self.currentLearningWeights(epoch)
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = global_lr * lr_perblock[i%3]
        ## print
        print(epoch, "lrs", global_lr, end=": ", file=sys.stderr)
        for i in range(min(3, len(optimizer.param_groups))):
            print(lr_perblock[i], end=" ", file=sys.stderr)
        losses_ws = self.currentLossWeights(epoch)
        print("losses", end='', file=sys.stderr)
        for i in range(3):
            print('', losses_ws[i], end='', file=sys.stderr)
        print('', file=sys.stderr)
    
    def accuracy(self, out, label, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        _label = []
        for l in range(len(label)):
            _label.append(torch.argmax(label[l], axis=1))
        
        pred = []
        correct = []
        with torch.no_grad():
            maxk = max(topk)
            batch_size = _label[0].size(0)
            for i in range(len(out)):
                pred.append(out[i].topk(min(maxk, len(out[i][0])), 1, True, True)[1].t())
                correct.append(pred[i].eq(_label[i].view(1, -1).expand_as(pred[i])))
            
            res = []
            for k in topk:
                correct_k = []
                for i in range(len(out)):
                    correct_k.append(correct[i][:min(k, len(out[i][0]))]\
                                     .contiguous().view(-1).float().sum(0, keepdim=True))
                res.append( [ c.mul_(100.0 / batch_size) for c in correct_k ] )
            return res

def countIndexes(indexes):
  unique, counts = np.unique(indexes, return_counts=True)
  return counts

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
        args.gpu=None

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_epoch
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {0} for {1}ing".format(args.gpu, args.action), file=sys.stderr)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loader
    traindir = os.path.join(args.data, 'training')
    valdir = os.path.join(args.data, 'validation')
    testdir = os.path.join(args.data, 'testing')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_size = 224
    
    train_dataset = TreeStructuredDataset(
        traindir,
        image_size,
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.GaussianBlur(3),
            transforms.RandomRotation((-10, 10)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
            transforms.RandomAffine(0, translate=None, scale=(1, 1.4), shear=2),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize]),
        classes = args.train_labels.split(',') if args.train_labels else []
    )
    
    valid_dataset = TreeStructuredDataset(
        valdir,
        image_size,
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize]),
        classes = train_dataset.classes
    )
    
    test_dataset = TreeStructuredDataset(
        testdir,
        image_size,
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize]),
        classes = train_dataset.classes
    )

    loss_weights = {}
    for l in args.loss_weights.split('/'):
        key, arr = l.split(':')
        key = int(key)
        loss_weights[key] = []
        for v in arr.split(','):
            loss_weights[key].append(float(v))
    learning_weights = {}
    for l in args.learning_weights.split('/'):
        key, arr = l.split(':')
        key = int(key)
        learning_weights[key] = []
        for v in arr.split(','):
            learning_weights[key].append(float(v))

    if args.structure == 'none':
        args.structure = ''
    if args.structure == '':
        loss_weights = {0: [0., 0., 1.]}
    
    if args.structure == 'rbcnn':
        args.residual = True
        args.structure = 'bcnn'
    if args.structure == 'rbcnn2':
        args.residual = True
        args.structure = 'bcnn2'
        
    
    # create model
    originalModel = HCNN(args.arch, args.pretrained, train_dataset.num_classes, train_dataset.children(),
                         residual = args.residual,
                         reverseLabels = args.reverse,
                         maskedLearning= args.maskedLearning,
                         lossWeights = loss_weights, learningWeights = learning_weights,
                         structure = args.structure,
                         suffix = args.suffix)
    args.arch = originalModel.name
    model = originalModel
    print(model, file=sys.stderr)
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
      model = torch.nn.DataParallel(model).cuda()

    traininglabelsC1 = train_dataset.labels[:, 0]
    traininglabelsC2 = train_dataset.labels[:, 1]
    traininglabelsC3 = train_dataset.labels[:, 2]
    
    positiveWeights = ( [ countIndexes(traininglabelsC1),
                          countIndexes(traininglabelsC2),
                          countIndexes(traininglabelsC3) ] )
    total = len(train_dataset.labels)
    for idx1 in range(len(positiveWeights)):
      for idx2 in range(len(positiveWeights[idx1])):
        positiveWeights[idx1][idx2] = (total - positiveWeights[idx1][idx2]) / positiveWeights[idx1][idx2]
      positiveWeights[idx1] = torch.tensor(positiveWeights[idx1]).to(args.gpu)
    
    criterions = ([nn.BCEWithLogitsLoss(pos_weight=positiveWeights[0]).cuda(args.gpu),
                   nn.BCEWithLogitsLoss(pos_weight=positiveWeights[1]).cuda(args.gpu),
                   nn.BCEWithLogitsLoss(pos_weight=positiveWeights[2]).cuda(args.gpu)])
    
    optimizer = torch.optim.SGD(model.module.initialParameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    train_loss = []
    train_losses = []
    train_accs = []
    valid_loss = []
    valid_losses = []
    valid_accs = []
    
    # Resume from a checkpoint
    if args.resume or args.test or args.evaluate:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume), file=sys.stderr)
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_epoch = checkpoint['epoch']
            model = model.cpu()
            model.load_state_dict(checkpoint['state_dict'])
                
            model = model.cuda()
            
            history = {}
            if 'history' in checkpoint:
                history = checkpoint['history']
                train_loss = history['train_loss']
                valid_loss = history['valid_loss']
                
            elif os.path.exists(args.data+'_'+args.arch+'_history.npy'):
                history = np.load(args.data+'_'+args.arch+'_history.npy', allow_pickle=True).item()
                train_loss = history['train_loss']
                valid_loss = history['valid_loss']
            
            if 'train_loss1' in history:
                train_losses.append(history['train_loss1'])
                train_losses.append(history['train_loss2'])
                train_losses.append(history['train_loss3'])
                valid_losses.append(history['valid_loss1'])
                valid_losses.append(history['valid_loss2'])
                valid_losses.append(history['valid_loss3'])
                train_accs.append(history['train_acc1'])
                train_accs.append(history['train_acc2'])
                train_accs.append(history['train_acc3'])
                valid_accs.append(history['valid_acc1'])
                valid_accs.append(history['valid_acc2'])
                valid_accs.append(history['valid_acc3'])
            else:
                train_losses = history['train_losses']
                valid_losses = history['valid_losses']
                train_accs = history['train_accs']
                valid_accs = history['valid_accs']
            
            if 'labels' in checkpoint:
                train_dataset.sortLabels(checkpoint['labels'].split(','))
                valid_dataset.sortLabels(checkpoint['labels'].split(','))
                test_dataset.sortLabels(checkpoint['labels'].split(','))
            elif os.path.exists(args.data+'_'+args.arch+'_labels.npy'):
                train_dataset.sortLabels(np.load(args.data+'_'+args.arch+'_labels.npy',
                                                allow_pickle=True).split(','))
                valid_dataset.sortLabels(np.load(args.data+'_'+args.arch+'_labels.npy',
                                                allow_pickle=True).split(','))
                test_dataset.sortLabels(np.load(args.data+'_'+args.arch+'_labels.npy',
                                                allow_pickle=True).split(','))
                
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']), file=sys.stderr)
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), file=sys.stderr)
            return

    cudnn.benchmark = True
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler
      )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    if args.evaluate:
        args.epochs = 1
        train(valid_loader, 'val', model, originalModel,
              criterions, optimizer, 0, args)
        return

    if args.test:
        args.epochs = 1
        train_dirs = args.train_labels.split(',') if args.train_labels else train_dataset.classes
        test_dirs = args.test_labels.split(',') if args.test_labels else train_dirs
        if not train_dirs:
            train_dirs = train_dataset.classes
        if not test_dirs:
            test_dirs = train_dataset.classes
        
        in_maps = []
        for dir in train_dirs:
            fields = dir.split('/')
            for idx in range(len(fields)):
                if len(in_maps) <= idx:
                    in_maps.append({})
                if not fields[idx] in in_maps[idx]:
                    in_maps[idx][fields[idx]] = len(in_maps[idx])
        
        out_maps = []
        for dir in test_dirs:
            fields = dir.split('/')
            for idx in range(len(fields)):
                if len(out_maps) <= idx:
                    out_maps.append({})
                if not fields[idx] in out_maps[idx]:
                    out_maps[idx][fields[idx]] = len(out_maps[idx])
        
        conversion_tab = np.zeros((len(in_maps), len(train_dirs)), dtype=int)
        for idx in range(len(in_maps)):
            for key in in_maps[idx]:
                conversion_tab[idx][in_maps[idx][key]] = out_maps[idx][key]
        
        train(test_loader, 'test', model, originalModel,
              criterions, optimizer, 0, args, label_conversion_table = conversion_tab)
        return

    filters = {}
    if args.filters:
        for filter in args.filters.split(','):
            e, l = filter.split(':')
            filters[int(e)] = l
    
    if os.path.exists('.STOP'):
        print("Aborted", file=sys.stderr)
        exit(1)
        
    for epoch in range(args.start_epoch, args.epochs):
        
        if epoch in filters:
            train_dataset.filter(filters[epoch])
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        originalModel.adjust_learning_rate(optimizer, epoch, args.lr)
        
        # train for one epoch
        accs, loss, losses = train(train_loader, 'train', model, originalModel,
                                   criterions, optimizer, epoch, args)
        train_loss.append(loss)
        for x in range(len(losses)):
            if len(train_losses) == x:
                train_losses.append([])
                train_accs.append([])
            train_losses[x].append(losses[x])
            train_accs[x].append(accs[x].data.cpu())
        
        # evaluate on validation set
        acc, loss, losses = train(valid_loader, 'val', model, originalModel,
                                  criterions, optimizer, epoch, args)
        
        valid_loss.append(loss)
        for x in range(len(losses)):
            if len(valid_losses) == x:
                valid_losses.append([])
                valid_accs.append([])
            valid_losses[x].append(losses[x])
            valid_accs[x].append(accs[x].data.cpu())
        
        # remember best acc@1 and save checkpoint
        is_best = False
        if accs[len(accs)-1] > best_acc1:
            is_best = True
            best_acc1 = accs[len(accs)-1]
            best_epoch = epoch
        elif epoch > 65 and epoch - best_epoch > 10:
            args.epochs = epoch
                    
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint( {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'labels': ','.join(train_dataset.classes),
                'data': args.data,
                'history': { 'train_loss': train_loss,
                             'train_losses': train_losses,
                             'valid_loss': valid_loss,
                             'valid_losses': valid_losses,
                             'train_accs': train_accs,
                             'valid_accs': valid_accs }
            }, is_best )
        
        if os.path.exists('.STOP'):
            os.remove('.STOP')
            print("Aborted", file=sys.stderr)
            args.epochs = epoch
    
    
    np.save(args.data+'_'+args.arch+'_history.npy',
            { 'train_loss': train_loss,
              'train_losses': train_losses,
              'valid_loss': valid_loss,
              'valid_losses': valid_losses,
              'train_accs': train_accs,
              'valid_accs': valid_accs })

from colorama import Fore, Back, Style
from scipy import stats
import statistics
from collections import Counter

def train(dataloader, phase,  model, originalModel, criterions, optimizer, epoch, args,
          label_conversion_table = ()):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sublosses = [AverageMeter() for _ in range(len(criterions))]
    top1 = [AverageMeter() for _ in range(len(criterions))]
    top5 = [AverageMeter() for _ in range(len(criterions))]

    # switch to train mode
    if phase == 'train':
      model.train()
    else:
      model.eval()

    testIO = {}
    y_pred = np.empty((len(criterions), len(dataloader) if phase == 'test' else 0))
    y_true = np.empty((len(criterions), len(dataloader) if phase == 'test' else 0))
    
    epoch_start = time.time()
    start = time.time()
    
    if phase == 'test' and os.path.exists(args.data+'_'+args.arch+'_results.csv'):
        os.remove(args.data+'_'+args.arch+'_results.csv')
            
    for i, (input, label) in enumerate(dataloader):
        data_time.add(time.time() - start)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        
        if isinstance(label, list):
            for l in range(len(label)):
                label[l] = label[l].cuda(args.gpu, non_blocking=True)
        else:
            label = label.cuda(args.gpu, non_blocking=True)
        
        # compute output
        out = model(input)
        loss = originalModel.loss(out, label, criterions, epoch)
        
        # measure accuracy and record loss
        acc1, acc5 = originalModel.accuracy(out, label, topk=(1, 5))

        if phase == 'test':
            exportCSV(args.data+'_'+args.arch+'_results.csv', out, label,
                      dataloader.dataset.filenames, i, label_conversion_table)
            preds = []
            labels = []
            for l in range(len(out)):
                preds.append(torch.argmax(out[l], axis=1))
                labels.append(torch.argmax(label[l], axis=1))

            ios = [[], []]
            for l in range(len(out)):
                y_pred[l][i]=label_conversion_table[l][torch.max(torch.exp(out[l]), 1)[1].data.cpu().numpy()]
                y_true[l][i]=label_conversion_table[l][labels[l].data.cpu().numpy()]
                ios[0].append(y_true[l][i])
                ios[1].append(y_pred[l][i])

            testIO[dataloader.dataset.filenames[i]] = ( [item for field in ios for item in field] )
        
        if isinstance(loss, tuple):
            losses.add(loss[0].item(), input.size(0))
            for subloss in range(len(loss[1])):
                sublosses[subloss].add(loss[1][subloss].item(), input.size(0))
                top1[subloss].add(acc1[subloss][0], input.size(0))
                top5[subloss].add(acc5[subloss][0], input.size(0))
        else:
            losses.add(loss.item(), input.size(0))
            sublosses.add(loss.item(), input.size(0))
            top1.add(acc1[0], input.size(0))
            top5.add(acc5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        if phase == 'train':
            if isinstance(loss, tuple):
                loss[0].backward()
            else:
                loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.add(time.time() - start)
        start = time.time()
        
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        print(end=LINE_CLEAR)
        
        if i > 0 and i % args.print_freq == 0:
            print('{0} {1} [{2}/{3}] [{4}/{5}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'L ' # {loss.val:.3f}:{sublossVal} '
                  '({loss.avg:.3f}:{sublossAvg})\t'
                  'A ' #{top1Val}:{top5.val:.3f} '
                  '({top1Avg}:{top5.avg:.3f})'\
                  .format( originalModel.name, phase, epoch+1, args.epochs, i,
                           len(dataloader),
                           batch_time=batch_time,
                           data_time=data_time,
                           loss=losses,
                           sublossVal='-'.join(["{:.3f}".format(sl.val) for sl in sublosses]),
                           sublossAvg='-'.join(["{:.3f}".format(sl.avg) for sl in sublosses]),
                           top1Val='-'.join(["{:.3f}".format(top.val) for top in top1]),
                           top1Avg='-'.join(["{:.3f}".format(top.avg) for top in top1]),
                           top5=top5[len(top5)-1] ), end='\r')
        
        if i == len(dataloader)-1:
          epoch_time = time.time() - epoch_start
          eta  = ( (time.time() - starting_start) / ((epoch+1) - args.start_epoch)
                   * (args.epochs - (epoch+1)) )
          if phase == 'train':
            print(Fore.GREEN, end='')
          else:
            print(Fore.BLUE, end='')
          print('{0} {1} [{2}/{3}] \t'
                'Time {epoch_time:.3f}\t'
                'L ({loss.avg:.4f}:{sublossAvg})\t'
                'A ({top1Avg}:{top5.avg:.4f})\t'
                'ETA {eta}'\
                .format( originalModel.name, phase, epoch+1, args.epochs,
                         epoch_time=epoch_time,
                         eta=str(timedelta(seconds=round(eta))),
                         data_time=data_time, loss=losses,
                         sublossAvg='-'.join(["{:.4f}".format(sl.avg) for sl in sublosses]),
                         top1Avg='-'.join(["{:.4f}".format(top.avg) for top in top1]),
                         top5=top5[len(top5)-1] ), end='\n')
          
          print(Style.RESET_ALL, end='')
          
    
    if phase == 'test':
        # constant for classes
        classes = dataloader.dataset.classes_names(label_conversion_table)
    
        for idx1 in range(len(classes)):
            for idx2 in range(len(classes[idx1])):
                if classes[idx1][idx2][:5] == 'View_':
                    classes[idx1][idx2] = classes[idx1][idx2][5:]
                
        # Build confusion matrix
        drawCM(y_true, y_pred, classes, args, '')
        
        # Vote classification
        vote_classif_10F = {}
        for filename in dataloader.dataset.filenames:
            vote_classif_10F[filename[:-8]] = np.array([])
        
        shuffled = dataloader.dataset.filenames.copy()
        random.shuffle(shuffled)
        for filename in shuffled:
            if len(vote_classif_10F[filename[:-8]]) == 0:
                vote_classif_10F[filename[:-8]] = np.array([testIO[filename]])
            elif len(vote_classif_10F[filename[:-8]]) < 10:
                vote_classif_10F[filename[:-8]] = np.concatenate((vote_classif_10F[filename[:-8]],
                                                                  [testIO[filename]]), axis=0)
        
        vote_classif_fl = {}
        for filename in dataloader.dataset.filenames:
            vote_classif_fl[filename[:-8]] = np.array([])
        
        rev_label_conversion_table = label_conversion_table.copy()
        for i1 in range(len(label_conversion_table)):
            for i2 in reversed(range(len(label_conversion_table[i1]))):
                rev_label_conversion_table[i1][label_conversion_table[i1][i2]] = i2
        
        for filename in shuffled:
            if dataloader.dataset.isAValidLabel(([rev_label_conversion_table[i%len(label_conversion_table)][int(x)]
                                                 for i, x in enumerate(testIO[filename])])):
                if len(vote_classif_fl[filename[:-8]]) == 0:
                    vote_classif_fl[filename[:-8]] = np.array([testIO[filename]])
                else:
                    vote_classif_fl[filename[:-8]] = np.concatenate((vote_classif_fl[filename[:-8]],
                                                                     [testIO[filename]]), axis=0)
        
        for i, vote in enumerate([vote_classif_10F, vote_classif_fl]):
            
            vote_y_true = np.empty((3, len(vote)), dtype=int)
            vote_y_pred = np.empty((3, len(vote)), dtype=int)

            idx = 0
            for key in vote:
                t = vote[key].transpose()
                if len(t) == 0:
                    t = vote_classif_10F[key].transpose()
                vote_y_true[0][idx] = int(t[0][0])
                vote_y_true[1][idx] = int(t[1][0])
                vote_y_true[2][idx] = int(t[2][0])
                c1 = Counter(t[3])
                c2 = Counter(t[4])
                c3 = Counter(t[5])
                vote_y_pred[0][idx] = int(max(c1, key=c1.get))
                vote_y_pred[1][idx] = int(max(c2, key=c2.get))
                vote_y_pred[2][idx] = int(max(c3, key=c3.get))
                idx += 1
            
            drawCM(vote_y_true, vote_y_pred, classes, args, ['_10F', '_FL'][i])
        
        return
    
    # Writing to log file
    try:
        with open(args.data+'_'+args.arch+'_'+phase+'_results.txt', 'a') as file:
            file.write( '[{0}/{1}]\t'
                        'Time {batch_time.avg:.3f}\t'
                        'Data {data_time.avg:.3f}\t'
                        'Loss {loss.avg:.4f}\t'
                        'L ({losses})\t'
                        'Acc@1 ({top1})\t'
                        'Acc@5 ({top5.avg:.3f})\n'\
                        .format( epoch+1, args.epochs,
                                 batch_time=batch_time,
                                 data_time=data_time,
                                 loss=losses,
                                 losses='-'.join(["{:.4f}".format(sl.avg) for sl in sublosses]),
                                 top1='-'.join(["{:.4f}".format(top.avg) for top in top1]),
                                 top5=top5[len(top5)-1] ) )
        
    except Exception as err:
        print(err, file=sys.stderr)

    writer.add_scalar("Loss/train", losses.avg, epoch)
    for idx in range(len(sublosses)):
        writer.add_scalar("Loss{}/train".format(idx), sublosses[idx].avg, epoch)
    for idx in range(len(sublosses)):
        writer.add_scalar("Acc{}/train".format(idx), top1[idx].avg, epoch)
    
    return [top.avg for top in top1], losses.avg, [sl.avg for sl in sublosses]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, state['data']+'_'+state['arch']+'_'+filename)
    if is_best:
        shutil.copyfile(state['data']+'_'+state['arch']+'_'+filename,
                        state['data']+'_'+state['arch']+'_model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from torch.nn.functional import normalize
import csv


def drawCM(pred, true, classes, args, tag):
    for cl in range(len(true)):
        cf_matrix = confusion_matrix(true[cl], pred[cl])
        # print(classes[cl])
        df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis],
                             index = [i for i in classes[cl]],
                             columns = [i for i in classes[cl]])
        # df_cm = pd.DataFrame(cf_matrix/numpy.sum(cf_matrix) * len(classes[cl]),
        #                      index = [i for i in classes[cl]],
        #                      columns = [i for i in classes[cl]])
        pyplot.figure(figsize = (12, 10))
        sn.heatmap(df_cm, annot=True)
        pyplot.savefig(args.data+'_'+args.arch+tag+'_'+str(cl)+'_confusion_matrix.png')
        
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes[cl]],
                            columns = [i for i in classes[cl]])
        pyplot.figure(figsize = (12,10))
        sn.heatmap(df_cm, annot=True, fmt='g')
        pyplot.savefig(args.data+'_'+args.arch+tag+'_'+str(cl)+'_confusion_matrix_card.png')

def exportCSV(filename, output, label, samples, batchId, label_conversion_table):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print('GT')
    # print(label.shape)
    # print('Prediction')
    # print(output.shape)
    _label = []
    for l in range(len(label)):
        _label.append(torch.argmax(label[l], axis=1))
    rev_label_conversion_table = label_conversion_table.copy()
    for i1 in range(len(label_conversion_table)):
        for i2 in reversed(range(len(label_conversion_table[i1]))):
            rev_label_conversion_table[i1][label_conversion_table[i1][i2]] = i2
    with open(filename, 'a', newline='\n') as file:
        writer = csv.writer(file, delimiter=' ', lineterminator='\n')
        for id in range(_label[0].shape[0]):
            row = []
            sample = samples[(batchId * _label[0].shape[0]) + id].split('.dcm')
            row.append(sample[0] + '.dcm') # DICOM
            row.append(sample[1]) # FRAME
            for i in range(len(_label)): # LABEL0, LABEL1, LABE3 
                row.append(label_conversion_table[i][_label[i][id].data.cpu().item()])
            for i in range(len(output)): # PRED0, PRED1, PRED3 
                row.append(label_conversion_table[i][torch.argmax(output[i][id].data.cpu()).item()])
            for i in range(len(_label)):
                tens = output[i][id].data.cpu()
                tens -= torch.min(tens)
                tens /= torch.sum(tens)
                # tens = normalize(, dim=0)
                for cls in range(output[i].shape[1]): # PRED_W_L1_0, PRED_W_L1_1, ... PRED_W_L3_8
                    row.append(tens[rev_label_conversion_table[i][cls]].item())
            
            writer.writerow(row)
        
        file.close()

if __name__ == '__main__':
    main()
