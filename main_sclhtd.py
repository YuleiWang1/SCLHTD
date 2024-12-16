import numpy as np
import torch
import os
import sys
import argparse
import math
import time
import torch.backends.cudnn as cudnn

from scipy import io as sio
from torchvision import transforms, datasets
from models import SelfSupConNet
from losses import SupConLoss
from contrastive_loss import InstanceLoss, ClusterLoss
from utils import AverageMeter, adjust_learning_rate, warmup_learning_rate
from utils import set_optimizer, save_model
from utils import loadData

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, Datapath1, Datapath2):
        self.DataList1 = Datapath1/1.0
        self.DataList2 = Datapath2/1.0
        
    def __getitem__(self, index):
        index = index
        Data = (self.DataList1[index])
        Data2 = (self.DataList2[index])
       
        return torch.FloatTensor(Data), torch.FloatTensor(Data2)
    def __len__(self):
        return len(self.DataList1)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='70,80,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='MUUFL',
                        choices=['GF5', 'MUUFL', 'Avon', 'Sandiego2', 'Sandiego100', 'HYDICE', 'Airport', 'Salinas', 'Airport2', 'Beach', 'Beach2', 'Urban1', 'Urban2', 'Segundo', 'Sandiego', 'Synthetic'], help='dataset')
    

    # method
    parser.add_argument('--model', type=str, default='ConResNet')
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    
    parser.add_argument('--trial', type=str, default='noSRCAM',
                        help='id for recording multiple runs用于记录多次运行的 id')

    opt = parser.parse_args()

    opt.model_path = './save/SelfCon/{}_models'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True

    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate


    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    model = SelfSupConNet()
    
    criterion_instance = InstanceLoss(opt.batch_size, opt.temp)
    criterion_cluster = ClusterLoss(2, opt.temp)


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion_instance = criterion_instance.cuda()
        criterion_cluster = criterion_cluster.cuda()
        cudnn.benchmark = True

    return model, criterion_instance, criterion_cluster

def train(train_loader, model, criterion_instance, criterion_cluster, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    bsz = opt.batch_size

    losses = AverageMeter()

    for idx, (x_i, x_j) in enumerate(train_loader):


        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')

        # z_i, z_j, c_i, c_j = model(x_i, x_j)
        # loss_instance = criterion_instance(z_i, z_j)
        # loss_cluster = criterion_cluster(c_i, c_j)
        # loss = loss_instance + 0*loss_cluster

        z_i, z_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss = loss_instance


        
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():

    opt = parse_option()

    # build data loader
    data, label , agumentation_one, agumentation_two = loadData(opt.dataset)
    label = label.reshape(-1)
    
    # without data augmentation
    H = agumentation_one.shape[0]
    W = agumentation_one.shape[1]
    B = agumentation_one.shape[2]
    agumentation_one = agumentation_one.reshape(H*W,B)
    agumentation_two = agumentation_two.reshape(H*W,B)
    # without data augmentation

    train_data = PairDataset(agumentation_one, agumentation_two)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    
    # build model and criterion
    model, criterion_instance, criterion_cluster = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    start = time.time()

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        loss = train(train_loader, model, criterion_instance, criterion_cluster, optimizer, epoch, opt)
       

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    end = time.time()
    print('training time:', end - start)
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
