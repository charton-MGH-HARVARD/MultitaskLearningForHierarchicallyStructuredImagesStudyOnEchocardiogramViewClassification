import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import os
import os.path as osp
import glob
import numpy as np
import random

def OH_encode(arr, num_classes):
  returnArr = np.zeros( (len(arr), num_classes) )
  for idx, a in enumerate(arr):
    a = int(a)
    returnArr[idx, a] = 1.0
  return returnArr

def encodeLabels(label, num_classes):
  result = []
  for c, cla in enumerate(num_classes):
    oh = np.zeros( (num_classes[c]), dtype=np.float32 )
    oh[int(label[c])] = 1.0
    result.append(oh)
  return result

# O/O/OTHERS
# A/1/View_A2C
# A/1/View_A3C
# A/2/View_A4C
# A/2/View_A5C
# P/L/View_PLAX
# P/L/View_PLAX_RV
# P/S/View_PSAX_AV
# P/S/View_PSAX_TV_RV

#       O--'OTHERS'--'OTHERS'
#      /
#     /   ---'View_A2C'
#    /  /o ---'View_A3C'
#   /  A
#  / /  \o ---'View_A4C'
# o       ---'View_A5C'
#  \     
#   \     /--'View_PLAX'
#    \  .L---'View_PLAX_RV'
#     \P
#       \ 
#        S --'View_PSAX_AV'
#         \--'View_PSAX_TV_RV'

class TreeStructuredDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""
    
    def __init__(self, directory, img_size=224, transform=None, classes = []):
        self.directory = directory
        self.chains = set()
        self.classes, self.maps = self.__readClasses(directory, classes)
        self.num_classes = ([len(self.maps[i]) for i in range(len(self.maps))])
        self.all_filenames = self.__readFiles(directory)
        self.all_labels = self.__labelsFromPaths(self.all_filenames)
        self.filenames = self.all_filenames
        self.labels = self.all_labels
        self.__filter = []
        self.img_dim = (img_size, img_size)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.filenames[idx]
        _labels = encodeLabels( self.labels[idx], self.num_classes )
        
        if self.transform:
         img_tensor = self.transform(Image.open(img_path))
        else:
          img = cv2.imread(img_path)
          img = cv2.resize(img, self.img_dim)
          img_tensor = torch.from_numpy(img / 255.0)
          img_tensor = img_tensor.permute(2, 0, 1)
        
        return img_tensor, [torch.tensor(_labels[x]) for x in range(len(_labels))]
        
    def __readClasses(self, directory, dirs = []):
      if not dirs:
        for dirpath, dirnames, filenames in os.walk(directory, topdown=True, onerror=None, followlinks=True):
          if not dirnames and dirpath:
            dirs.append(dirpath[len(directory)+1:])
        dirs = sorted(dirs)
      
      maps = []
      for dir in dirs:
        fields = dir.split('/')
        for idx in range(len(fields)):
          if len(maps) <= idx:
            maps.append({})
          if not fields[idx] in maps[idx]:
            maps[idx][fields[idx]] = len(maps[idx])
        t = tuple([maps[x][fields[x]] for x in range(len(fields))])
        self.chains.add(t)

      return dirs, maps
      
    def __readFiles(self, directory):
      files = list()
      for dirpath, dirnames, filenames in os.walk(directory, topdown=True, onerror=None, followlinks=True):
        if filenames:
          for file in filenames:
            if file[-4:] == '.png':
              files.append(osp.join(dirpath, file))
      return files
    
    def __labelsFromPaths(self, filenames):
      labels = np.empty((len(filenames), 3), dtype=int)
      for i, filename in enumerate(filenames):
        fields = filename[len(self.directory)+1:].split('/')[0:-1]
        for x in range(len(labels[i])):
          labels[i][x] = self.maps[x][fields[x]]
      return labels
      
    def __applyFilter(self):
      if self.__filter:
        self.filenames = []
        self.labels = []
        for filename, label in zip(self.all_filenames, self.all_labels):
          if label[self.__filter[0]] == self.__filter[1]:
            self.filenames.append(filename)
            self.labels.append(label)
      else:
        self.filenames = self.all_filenames
        self.labels = self.all_labels

    def applyFilter(self, label=''):
      if label == '':
        self.__filter = []
      else:
        self.__filter = [len(label.split('/'))-1, -1]
        subclasses = []
        for c in self.classes:
          subclass = '/'.join(c.split('/')[:self.__filter[0]+1])
          if not subclasses or subclasses[-1] != subclass:
            subclasses.append(subclass)
        for i, c in enumerate(subclasses):
          if c[:len(label)] == label:
            self.__filter[1] = i
      self.__applyFilter()
    
    def classes_names(self, label_conversion_table):
      classes = np.empty((len(self.maps)), dtype=list)
      for i in range(len(self.maps)):
        classes[i] = ["" for _ in range(len(self.maps[i]))]
        for key in self.maps[i]:
          classes[i][label_conversion_table[i][self.maps[i][key]]] = key
      return classes
    
    def sortLabels(self, classes):
      self.chains = set()
      self.classes, self.maps = self.__readClasses(self.directory, classes)
      self.all_labels = self.__labelsFromPaths(self.all_filenames)
      self.filenames = self.all_filenames
      self.labels = self.all_labels
      self.labels = self.all_labels
      self.__applyFilter()
        
    def isAValidLabel(self, label):
      return tuple(label) in self.chains
    
    def children(self):
      children = []
      for t in self.chains:
        for i, e in enumerate(t[:-1]):
          if len(children) == i:
            children.append({})
          if e in children[i]:
            children[i][e].add(t[i+1])
          else:
            children[i][e] = set([t[i+1]])
      return children
