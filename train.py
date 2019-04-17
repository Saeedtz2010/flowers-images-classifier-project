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
from torch import nn
from torch import tensor
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse

#-----------
from do_deep_learning import do_deep_learning
from nn_model import nn_model
from process_image import process_image

# first off: in the command line; cd aipnd-project

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dest', nargs='*', action="store", default="flowers/", help='path to flowers folder')
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu",help="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth",help='checkpoint')
parser.add_argument('-lr','--learning_rate', dest="learning_rate", action="store", default=0.001,help='learning rate')
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2,help='dropout')
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=2,help='epoches')
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str,help='model archicture')
parser.add_argument('--hidden_layer_in', type=int, dest="hidden_layer_in", action="store", default=400,help='hidden layer inputs')
parser.add_argument('--hidden_layer_out', type=int, dest="hidden_layer_out", action="store", default=200,help='hidden layer inputs')
args=parser.parse_args()

data_dir = args.dest
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

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


model= nn_model(args.arch,args.hidden_layer_in,args.hidden_layer_out, args.dropout)
deep_learning=do_deep_learning(model,trainloader,validloader,args.epochs,args.learning_rate,args.gpu)

model.class_to_idx = train_data.class_to_idx
if torch.cuda.is_available() and args.gpu == 'gpu':
    model.to('cuda')
torch.save({'model_name' :args.arch,
            'hidden_layer_in':args.hidden_layer_in,
            'hidden_layer_out':args.hidden_layer_out,
            'dropout':args.dropout,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            'checkpoint.pth')