# -*- coding: utf-8 -*-

import argparse
import time

from model import resnet

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable

from model.utils import load_filtered_state_dict, SaveBestModel, AverageMeter, accuracy
from data_wrapper import get_dataset, DataWrapper
from tensorboardX import SummaryWriter

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use', nargs='+',
            default=[0, 1], type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=100, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=64, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.1, type=float)
    parser.add_argument('--trainning_data_dir', dest='trainning_data_dir', help='Directory path for trainning data.',
          default='./data/train', type=str)
    parser.add_argument('--validation_data_dir', dest='validation_data_dir', help='Directory path for validation data.',
          default='./data/test', type=str)
    
    parser.add_argument('--save_path', dest='save_path', help='Path of model snapshot for save.',
          default='./models', type=str)
    parser.add_argument('--saved_model', help='Path of model snapshot for continue training.',
          default='./models/resnet50-19c8e357.pth', type=str)

    args = parser.parse_args()
    return args



def evaluate(eval_loader, model, writer, step, Save_model, epoch):
        
    top_prec = AverageMeter()    
    softmax = nn.Softmax().cuda()

    for i, (images, labels, names) in enumerate(eval_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        label_pred = model(images)
        label_pred = softmax(label_pred)

        prec = accuracy(label_pred, labels, topk=(1,))
        
        top_prec.update(prec[0].item())
        

    print('evaluate * Prec@1 {top:.3f}'.format(top=top_prec.avg))

    writer.add_scalar('eval_prec', top_prec.avg, step)
        
    Save_model.save(model, top_prec.avg, epoch)

def train(train_loader, model, criterion, optimizer, writer, batch_size, epoch, step, n):
    last_time = time.time()

    for i, (images, labels, name) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
                    
        label_pred = model(images)        

        # Cross entropy loss
        loss = criterion(label_pred, labels)

        writer.add_scalar('loss', loss, step)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            writer.add_scalar('learning_rate', lr, step)
            break            
        
        optimizer.zero_grad()
        loss.backward()            
        optimizer.step()

        if i % 10 == 0:
            curr_time = time.time()
            sps = 10.0 / (curr_time - last_time) * batch_size
            print("Epoch [{}], Iter [{}/{}]  {} samples/sec, Losses: {}".format(epoch+1, 
                i+1, n//batch_size, sps, loss.item()))
            
            last_time = curr_time

        step += 1

    # evaluate
    softmax = nn.Softmax().cuda()
    label_pred = softmax(label_pred)
    prec = accuracy(label_pred, labels, topk=(1,))
    print('training * Prec@1 {top:.3f}'.format(top=prec[0].item()))
    writer.add_scalar('training_prec', prec[0].item(), step)

    return step

def main(args):
    cudnn.enabled = True    

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(320),
        transforms.RandomCrop(299), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_x, train_y, classes_names = get_dataset(args.trainning_data_dir)
    test_x, test_y, _ = get_dataset(args.validation_data_dir)
    num_classes = len(classes_names)

    trainning_dataset = DataWrapper(train_x, train_y, transformations)
    eval_dataset = DataWrapper(test_x, test_y, transformations)

    train_loader = torch.utils.data.DataLoader(dataset=trainning_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=32)
    
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=32)
    n = trainning_dataset.__len__()
    print(n)

    # ResNet50 structure
    model = resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes)
    if args.saved_model:
        print('Loading model.')
        saved_state_dict = torch.load(args.saved_model)

        # 'origin model from pytorch'
        if 'resnet' in args.saved_model:
            load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=False)
        else:
            load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=True)


    crossEntropyLoss = nn.CrossEntropyLoss().cuda()    
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60], gamma=0.1)    
    

    # multi-gpu
    model = nn.DataParallel(model, device_ids=args.gpu)
    model.cuda()

    Save_model = SaveBestModel(save_dir=args.save_path)
    Writer = SummaryWriter()
    step = 0
    for epoch in range(args.num_epochs):
        scheduler.step()
        evaluate(eval_loader, model, Writer, step, Save_model, epoch)
        step = train(train_loader, model, crossEntropyLoss, optimizer, Writer, args.batch_size, epoch, step, n)        


if __name__ == '__main__':
    args = parse_args()

    main(args)
    


