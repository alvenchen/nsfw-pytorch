import os
import torch

def load_filtered_state_dict(model, snapshot, ignore_layer=None, reverse=False, gpu=False):
    model_dict = model.state_dict()     
    if reverse:
        #  snapshot keys have prefix 'module.'
        new_snapshot = dict()
        for k, v in snapshot.items():
            name = k[7:] # remove `module.`
            new_snapshot[name] = v  
        snapshot = new_snapshot
    else:
        snapshot = {k: v for k, v in snapshot.items() if k in model_dict}

    if ignore_layer:
        for l in ignore_layer:   
            print("ignore_layer : {}".format(snapshot[l]))         
            del snapshot[l]

    model_dict.update(snapshot)
    print(len(snapshot), len(model_dict))

    model.load_state_dict(model_dict)


class SaveBestModel():
    def __init__(self, save_dir='models'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir;
        self._reset()
    def _reset(self):
        self.precision = 0
        self.last_epoch = 0
    def save(self, model, prec, epoch):
        if prec > self.precision or epoch - self.last_epoch >= 5:
            self.precision = prec
            self.last_epoch = epoch

            print('Taking snapshot...')
            torch.save(model.state_dict(), self.save_dir + '/epoch_'+ str(epoch+1) + '.pkl')



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""    
    
    batch_size = target.size(0)
    
    res = []

    # absolute_difference esitimate
    _, abs_esti  = output.topk(1, 1, True, True)
    for k in topk:    
        correct = (torch.abs(target.view(-1) - abs_esti.view(-1).long()) < k).float().sum()
        res.append(correct.mul_(100.0 / batch_size))
        
    return res