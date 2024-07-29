import torch
from models.models import ResNets
from trainer import SupervisedTrainer
import numpy as np
import random
from data.dataset import VeinDataset, BalancedBatchSampler, get_transforms
from torch.utils.data.dataloader import DataLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FVUSM', help='name of the dataset')
    parser.add_argument('--trainset', type=str, help='train set path')
    parser.add_argument('--testset', type=str, help='test set path')
    parser.add_argument('--network', type=str, default='resnet18', help='name of the network: {resnet18, resnet34, resnet50}')
    parser.add_argument('--load_from', type=str, default=None, help='load pretrained model')
    parser.add_argument('--head_type', type=str, default='cls_norm', help='the type of head: {cls_norm, cls}')
    parser.add_argument('--loss', type=str, default='fusionloss', help="loss function: {softmax, cosface, tripletloss, fusionloss}")
    parser.add_argument("--intra_aug", action='store_true', help="apply intra-class data augmentation")
    parser.add_argument('--inter_aug', type=str, default=None, help="apply flipping-based inter-class data augmentation: {LR, TB}")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay factor')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for sgd optimizer')
    # parser.add_argument('--wd', type=float, default=4e-4, help='weight decay')
    parser.add_argument('--s', type=float, default=30.0, help='scaling factor in CosFace')
    parser.add_argument('--m', type=float, default=0.2, help='additive margin in CosFace')
    parser.add_argument('--p', type=int, default=8, help='randomly select p classes for a mini-batch')
    parser.add_argument('--k', type=int, default=4, help='randomly select k samples per class for a mini-batch')
    parser.add_argument('--hard_margin', type=float, default=0.2, help='hard_margin in triplet loss')
    parser.add_argument('--w_cls', type=float, default=1.0,
                        help='weight factor of large margin cosine loss (cosface) in fusion loss')
    parser.add_argument('--w_metric', type=float, default=4.0,
                        help='weight factor of triplet loss in fusion loss')
    parser.add_argument('--max_epoch', type=int, default=80, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=1, help='random seed for repeating results')
    parser.add_argument("--save_image", action='store_true', help="save the augmented images during training")
    parser.add_argument('--simple_eval', action='store_true', help="whether to use simplified evaluation protocol")
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    if args.dataset_name.lower() == "fvusm":
        sample_per_class = 12
        img_size = (64, 144)
    else:
        raise ValueError("Dataset %s not supported!" % args.dataset_name)

    transform_train, transform_test = get_transforms(args.dataset_name, img_size=img_size, data_aug=args.intra_aug)
    trainset = VeinDataset(root=args.trainset, sample_per_class=sample_per_class, transform=transform_train, inter_aug=args.inter_aug)
    if args.loss == 'tripletloss' or args.loss == 'fusionloss':
        train_batch_sampler = BalancedBatchSampler(trainset, n_classes=args.p, n_samples=args.k)
        trainloader = DataLoader(dataset=trainset, batch_sampler=train_batch_sampler, num_workers=4, pin_memory=True)
    else:
        trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    testset = VeinDataset(root=args.testset, sample_per_class=sample_per_class, transform=transform_test, inter_aug=None)
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    network = ResNets(backbone=args.network, head_type=args.head_type, num_classes=trainset.class_num).to(device)
    if args.load_from is not None and args.load_from != "null":
        pretrained_dict = torch.load(args.load_from)['model']
        current_dict = network.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in current_dict and 'head' not in k}
        current_dict.update(filtered_dict)
        network.load_state_dict(current_dict)
        print(f"loaded model keys: {filtered_dict.keys()}")

    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=args.lr_decay)

    params = {"max_epochs": args.max_epoch,
              "lr_scheduler": lr_scheduler,
              "batch_size": args.batch_size,
              "save_image": args.save_image,
              "loss": args.loss,
              "w_cls": args.w_cls,
              "w_metric": args.w_metric,
              "s": args.s,
              "m": args.m,
              "hard_margin": args.hard_margin,
              "args": args
              }
    trainer = SupervisedTrainer(network=network, optimizer=optimizer, device=device, **params)
    trainer.train(trainloader, testloader)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    set_seed(args.seed)
    main()
