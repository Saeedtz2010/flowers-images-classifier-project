import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import numpy as np
import time
from torch import tensor
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse

#----------------

from load_model import load_model
from nn_model import nn_model
from pred import pred


parser = argparse.ArgumentParser(description='predict')
parser.add_argument('--dest', nargs='*', action="store", default="flowers/", help='path to flowers folder')
parser.add_argument('--input', default='flowers/test/19/image_06155.jpg', nargs='*', action="store", type = str)
parser.add_argument('--checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action="store",type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
args=parser.parse_args()

data_dir = args.dest
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Load the datasets with ImageFolder

train_transforms =transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32,shuffle=True) 
testloader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True)

model=load_model(path='checkpoint.pth')

model.class_to_idx =train_data.class_to_idx
probs,classes = pred(args.input, model,args.gpu)
print(probs)
print(classes)