# -*- coding: utf-8 -*-

import argparse
import time

from model import resnet

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from model.utils import load_filtered_state_dict, SaveBestModel, AverageMeter, accuracy
from PIL import Image
import glob
import os

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+', 
        default=[0], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
        default=32, type=int)
        
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='Directory path for validation data.',
        default='./data/test', type=str)
        
    parser.add_argument('--saved_model', help='Path of model snapshot for continue training.',
        default='./models/epoch_36.pkl', type=str)

    parser.add_argument('--num_classes', help='num of classes.', default=5, type=int)

    args = parser.parse_args()
    return args


def main(args):    
    
    model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], args.num_classes)
    saved_state_dict = torch.load(args.saved_model)    

    transformations = transforms.Compose([transforms.Resize(320),
        transforms.RandomCrop(299), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    if args.gpu[0] >=0:        
        cudnn.enabled = True 
        softmax = nn.Softmax().cuda()
        model.cuda()
    else:
        softmax = nn.Softmax()        

    load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=True)

    imgs_path = glob.glob(os.path.join(args.test_data_dir, '*.jpg'))
    for i in xrange((len(imgs_path) + args.batch_size - 1) / args.batch_size):
        if args.gpu[0] >=0:
            imgs = torch.FloatTensor(args.batch_size, 3, 224, 224).cuda()
        else:
            imgs = torch.FloatTensor(args.batch_size, 3, 224, 224)
        for j in xrange(min(args.batch_size, len(imgs_path))):
            img = Image.open(imgs_path[i*args.batch_size + j])
            img = img.convert("RGB")            
            imgs[j] = transformations(img)

        pred = model(imgs)
        pred = softmax(pred)
        _, pred = pred.topk(1, 1, True, True)

        for j in xrange(min(args.batch_size, len(imgs_path))):
            print("{} -- {}".format(imgs_path[i*args.batch_size + j], pred[j]))


if __name__ == '__main__':
    args = parse_args()

    main(args)