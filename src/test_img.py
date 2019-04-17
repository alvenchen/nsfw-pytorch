# -*- coding: utf-8 -*-

import argparse
import time

from model import resnet
from model.dpn import dpn92

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
import cv2
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+', 
        default=[0], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
        
    parser.add_argument('--image_size', dest='image_size', help='Image size.',
        default=299, type=int)
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='Directory path for validation data.',
        default='./data/test', type=str)
        
    parser.add_argument('--saved_model', help='Path of model snapshot for continue training.',
        default='./models/epoch_53.pkl', type=str)

    parser.add_argument('--num_classes', help='num of classes.', default=5, type=int)
    parser.add_argument('--save_path', help='path for save result.', default='', type=str)

    args = parser.parse_args()
    return args

default_class=['drawings', 'hentai', 'neutral', 'porn', 'sexy']

def main(args):    
    
    model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], args.num_classes)    
    #model = dpn92(num_classes=args.num_classes)

    transformations = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    
    if args.gpu[0] >=0:        
        cudnn.enabled = True 
        softmax = nn.Softmax().cuda()
        model.cuda()
        saved_state_dict = torch.load(args.saved_model)
    else:
        softmax = nn.Softmax()        
        saved_state_dict = torch.load(args.saved_model, map_location='cpu')

    load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=True, gpu=cudnn.enabled)

    imgs_path = glob.glob(os.path.join(args.test_data_dir, '*.jpg'))
    imgs_path += glob.glob(os.path.join(args.test_data_dir, '*.jpeg'))
    imgs_path += glob.glob(os.path.join(args.test_data_dir, '*.png'))

    for i in xrange((len(imgs_path) + args.batch_size - 1) / args.batch_size):
        if args.gpu[0] >=0:
            imgs = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size).cuda()
        else:
            imgs = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
        for j in xrange(min(args.batch_size, len(imgs_path))):
            img = Image.open(imgs_path[i*args.batch_size + j])
            img = img.convert("RGB")            
            imgs[j] = transformations(img)

        pred = model(imgs)
        pred = softmax(pred)
        _, pred_1 = pred.topk(1, 1, True, True)

        for j in xrange(min(args.batch_size, len(imgs_path))):            
            c = default_class[pred_1.cpu().numpy()[0][0]]
            print("{} -- {} {}".format(imgs_path[i*args.batch_size + j], pred_1, c))

            if args.save_path:
                img_numpy = imgs[j].cpu().numpy()
                img_numpy = img_numpy * 255
                # change to channel last
                img_numpy = np.transpose(img_numpy, (1,2,0)).astype(np.uint8)
                # rgb to bgr
                img_numpy = img_numpy[...,::-1].copy()

                cv2.putText(img_numpy, c, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

                save_path = os.path.join(args.save_path, os.path.basename(imgs_path[i*args.batch_size + j]))                
                cv2.imwrite(save_path, img_numpy)


if __name__ == '__main__':
    args = parse_args()

    main(args)