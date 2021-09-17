import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from losses import losses
from utils import utils
import argparse
import numpy as np
import os
from evaluation.evaluation import evaluate_verification
from networks import resnet
from datasets import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='/home/weifeng/Desktop/datasets/FingerVeinDatasets/FV-USM-processed',
                        help="dataset path")
    parser.add_argument('--dataset', type=str, default='FVUSM', help="name of the database: FVUSM or Palmvein")
    parser.add_argument('--network', type=str, default='resnet18')
    parser.add_argument('--loss', type=str, default='softmax')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--s', type=float, default=30.0, help='scaling factor')
    parser.add_argument('--m', type=float, default=0.2, help='margin')
    parser.add_argument('--p', type=int, default=8, help='randomly select p classes for tripletloss')
    parser.add_argument('--k', type=int, default=4, help='randomly select k samples per class for tripletloss')
    parser.add_argument('--hard_margin', type=float, default=0.2, help='hard_margin for triplet loss')
    parser.add_argument('--gamma_tri', type=float, default=4.0,
                        help='weight factor for triplet loss when using fusion loss')
    parser.add_argument('--gamma_cos', type=float, default=1.0,
                        help='weight factor for cosface loss when using fusion loss')
    parser.add_argument('--max_epoch', type=int, default=80, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--seed', type=int, default=1, help='random seed for repeating results')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument("--pretrained", action='store_true', help="pretrained on imageNet")
    parser.add_argument("--intra_aug", action='store_true', help="apply intra-class data augmentation")
    parser.add_argument('--inter_aug', type=str, default='', help="apply inter-class data augmentation by left-right flip (LR) or top-bottom flip (TB)")
    args = parser.parse_args()
    return args


def save_model(model, current_result, best_result, best_snapshot):
    eer = current_result['eer']
    epoch = current_result['epoch']
    prefix = 'seed=%d_dataset=%s_network=%s_loss=%s' % (args.seed, args.dataset, args.network, args.loss)
    # save the current best model
    if not os.path.exists('snapshots'):
        os.mkdir('snapshots')
    if best_result is None or eer <= best_result['eer']:
        best_result = current_result
        snapshot = {'model': model.state_dict(), 'epoch': epoch, 'args': args}
        if best_snapshot is not None:
            os.system('rm %s' % (best_snapshot))
        best_snapshot = './snapshots/%s_BestEER=%.2f_Epoch=%d.pth' % (prefix, eer * 100, epoch)
        torch.save(snapshot, best_snapshot)
    # always save the final model
    if epoch == args.max_epoch - 1:
        snapshot = {'model': model.state_dict(), 'epoch': epoch, 'args': args}
        last_snapshot = './snapshots/%s_FinalEER=%.2f_Epoch=%d.pth' % (prefix, eer * 100, epoch)
        torch.save(snapshot, last_snapshot)
    return best_result, best_snapshot


def train(model, trainloader, optimizer, criterion, epoch):
    model.train()
    loss_stats = utils.AverageMeter()
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.cuda(), labels.cuda()
        if args.loss == 'triplet':
            features, _ = model(data)
            loss, num_triplet = criterion(features, labels)
        elif args.loss == 'fusion':
            features, outputs = model(data)
            criterion_triplet = criterion['triplet']
            criterion_cosface = criterion['cosface']
            loss_triplet, num_triplet = criterion_triplet(features, labels)
            loss_cosface = criterion_cosface(outputs, labels)
            loss = args.gamma_cos*loss_cosface + args.gamma_tri*loss_triplet
        else:
            features, outputs = model(data)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_stats.update(loss.item())
        if batch_idx % args.log_interval == 0:
            print(utils.dt(), 'Epoch:[%d]-[%d/%d] batchLoss:%.4f averLoss:%.4f' %
                  (epoch, batch_idx, len(trainloader), loss_stats.val, loss_stats.avg))


def train_epochs_openset(model, trainloader, testloader, optimizer, lr_scheduler, loss_func):
    max_epoch = args.max_epoch
    best_result = None
    best_snapshot = None
    print(utils.dt(), 'Training started.')
    for epoch in range(max_epoch):
        train(model, trainloader, optimizer, loss_func, epoch)
        roc, _ = evaluate_verification(model, testloader)
        lr_scheduler.step()
        # save the current best model based on eer
        best_result, best_snapshot = \
            save_model(model, {'metrics': roc,  'eer': roc[0], 'epoch': epoch}, best_result, best_snapshot)

    print(utils.dt(), 'Training completed.')
    print(utils.dt(), '------------------Best Results---------------------')
    epoch, roc = best_result['epoch'], best_result['metrics']
    print(utils.dt(), 'EER: %.2f%%, FPR100:%.2f%%, FPR1000:%.2f%%, FPR10000:%.2f%%, FPR0:%.2f%%, Aver: %.2f%% @ epoch %d' %
          (roc[0]*100, roc[1]*100, roc[2]*100, roc[3]*100, roc[4]*100, np.mean(roc)*100, epoch))


def main():
    normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    transform_train = []
    if args.intra_aug:
        transform_train.append(transforms.RandomResizedCrop(size=(64, 144), scale=(0.5, 1.0), ratio=(2.25, 2.25)))
        transform_train.append(transforms.RandomRotation(degrees=3))
        transform_train.append(transforms.RandomPerspective(distortion_scale=0.3, p=0.9))
        transform_train.append(transforms.ColorJitter(brightness=0.7, contrast=0.7))
    transform_train.append(transforms.ToTensor())
    transform_train.append(normalize)
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    if args.dataset == 'FVUSM':
        sample_per_class = 12
    else:
        raise ValueError('Dataset %s not exists!' % (args.dataset))
    from datasets.datasets import ImageDataset
    trainset = ImageDataset(root=args.data, sample_per_class=sample_per_class, transforms=transform_train, mode='train', inter_aug=args.inter_aug)

    testset = ImageDataset(root=args.data, sample_per_class=sample_per_class, transforms=transform_test, mode='test', inter_aug="")

    if args.loss == 'triplet' or args.loss == 'fusion':
        train_batch_sampler = datasets.BalancedBatchSampler(trainset, n_classes=args.p, n_samples=args.k)
        trainloader = DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=4, pin_memory=True)
    else:
        trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if args.network == 'resnet18':
        model = resnet.resnet18(pretrained=args.pretrained, num_classes=trainset.class_num, loss=args.loss)
    elif args.network == 'resnet34':
        model = resnet.resnet34(pretrained=args.pretrained, num_classes=trainset.class_num, loss=args.loss)
    elif args.network == 'resnet50':
        model = resnet.resnet50(pretrained=args.pretrained, num_classes=trainset.class_num, loss=args.loss)
    else:
        raise ValueError('Network %s not supported!' % (args.network))

    if args.loss == 'softmax':
        loss_func = nn.CrossEntropyLoss()
    elif args.loss == 'cosface':
        loss_func = losses.CosFace(s=args.s, m=args.m)
    elif args.loss == 'triplet':
        loss_func = losses.OnlineTripletLoss(margin=args.hard_margin, is_distance=True)
    elif args.loss == 'fusion':
        tripletloss = losses.OnlineTripletLoss(margin=args.hard_margin, is_distance=True)
        cosface = losses.CosFace(s=args.s, m=args.m)
        loss_func = {'triplet': tripletloss, 'cosface': cosface}
    else:
        raise ValueError('Loss %s not supported!' % (args.loss))

    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=args.lr_decay)

    train_epochs_openset(model, trainloader, testloader, optimizer, lr_scheduler, loss_func)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    utils.set_seed(args.seed)
    main()
