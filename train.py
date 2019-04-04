from __future__ import print_function

import warnings

from utils.core import print_info, set_logger, init_net, set_optimizer, set_criterion, anchors, save_checkpoint, adjust_learning_rate, write_logger, print_train_log
from m2det.datasets.datasets import get_dataloader, DataSets

warnings.filterwarnings('ignore')

import time
import torch
import argparse
from net import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from m2det.datasets import detection_collate
from configs.CC import Config

# from utils.core import *

parser = argparse.ArgumentParser(description='M2Det Training')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg.py')
parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')
args = parser.parse_args()

print_info('----------------------------------------------------------------------\n'
           '|                       M2Det Training Program                       |\n'
           '----------------------------------------------------------------------',['yellow','bold'])

logger = set_logger(args.tensorboard)
global cfg
cfg = Config.fromfile(args.config)
net = build_net('train', 
                size = cfg.model.input_size, # Only 320, 512, 704 and 800 are supported
                config = cfg.model.m2det_config)
init_net(net, cfg, args.resume_net) # init the network with pretrained weights or resumed weights

if args.ngpu>1:
    net = torch.nn.DataParallel(net)
if cfg.train_cfg.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = set_optimizer(net, cfg)
criterion = set_criterion(cfg)
priorbox = PriorBox(anchors(cfg))

with torch.no_grad():
    priors = priorbox.forward()
    if cfg.train_cfg.cuda:
        priors = priors.cuda()

if __name__ == '__main__':
    net.train()
    epoch = args.resume_epoch
    print_info('===> Loading Dataset...',['yellow','bold'])
    # dataset = get_dataloader(cfg, args.dataset, 'train_sets')
    root_path = "/raid/projects/logo_detection/M2Det/datasets/toplogo10"
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    # dataset = DataSets(name=DataSets.NAME_TOPLOGO10, root_path=root_path, transforms=tf).train
    # dataset = DataSets(name=DataSets.NAME_COCO, root_path=root_path, transforms=tf, cfg=cfg, dataset=args.dataset, setname="train_sets")
    dataset = DataSets(name=DataSets.NAME_TOPLOGO10, root_path=root_path, transforms=tf, cfg=cfg, dataset=args.dataset, setname="train_sets").train

    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    max_iter = getattr(cfg.train_cfg.step_lr,args.dataset)[-1] * epoch_size
    stepvalues = [_*epoch_size for _ in getattr(cfg.train_cfg.step_lr, args.dataset)[:-1]]
    print_info('===> Training M2Det on ' + args.dataset, ['yellow','bold'])
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    dataloader = data.DataLoader(
        dataset,
        cfg.train_cfg.per_batch_size * args.ngpu,
        shuffle=True,
        num_workers=cfg.train_cfg.num_workers,
        collate_fn=detection_collate
    )
    iteration = 0
    N_epoch = 1000
    for epoch in range(N_epoch):
        if epoch % cfg.model.save_eposhs == 0:
            save_checkpoint(net, cfg, final=False, datasetname = args.dataset, epoch=epoch)
        for iter_idx, (images, targets) in enumerate(dataloader):

            # if iteration % epoch_size == 0:
                # batch_iterator = iter(data.DataLoader(dataset,
                #                                       cfg.train_cfg.per_batch_size * args.ngpu,
                #                                       shuffle=True,
                #                                       num_workers=cfg.train_cfg.num_workers,
                #                                       collate_fn=detection_collate))
                # if epoch % cfg.model.save_eposhs == 0:
                #     save_checkpoint(net, cfg, final=False, datasetname = args.dataset, epoch=epoch)
                # epoch += 1
            load_t0 = time.time()
            if iteration in stepvalues:
                step_index += 1
            lr = adjust_learning_rate(optimizer, cfg.train_cfg.gamma, epoch, step_index, iteration, epoch_size, cfg)
            # try:
            #     images, targets = next(batch_iterator)
            # except:
            #     print("for debugging...")
            if cfg.train_cfg.cuda:
                images = images.cuda()
                targets = [anno.cuda() for anno in targets]
            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets)
            loss = loss_l + loss_c
            write_logger({'loc_loss':loss_l.item(),
                          'conf_loss':loss_c.item(),
                          'loss':loss.item()},logger,iteration,status=args.tensorboard)
            loss.backward()
            optimizer.step()
            load_t1 = time.time()
            print_train_log(iteration, cfg.train_cfg.print_epochs,
                                [time.ctime(),epoch,iteration%epoch_size,epoch_size,iteration,loss_l.item(),loss_c.item(),load_t1-load_t0,lr])

            iteration += 1
    save_checkpoint(net, cfg, final=True, datasetname=args.dataset,epoch=-1)
