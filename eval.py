"""
    Evaluate the verification performance of provided ckpt on the testing set of FVUSM
    Example: python3 -u ./eval.py --ckpt ${ckpt} --data ${data} --dataset ${dataset} --network ${network}
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from networks import resnet
from evaluation.evaluation import compute_roc, compute_roc_metrics
from datasets.datasets import ImageDataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help="checkpoint path")
    parser.add_argument('--data', type=str, help="dataset path")
    parser.add_argument('--dataset', type=str, default='FVUSM', help="name of the database")
    parser.add_argument('--network', type=str, default='resnet18')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.dataset == 'FVUSM':
        sample_per_class = 12
    else:
        raise ValueError('Dataset %s not exists!' % (args.dataset))

    normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    testset = ImageDataset(root=args.data, sample_per_class=sample_per_class, transforms=transform_test, mode='test')
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if args.network == 'resnet18':
        model = resnet.resnet18(pretrained=True, num_classes=testset.class_num)
    elif args.network == 'resnet34':
        model = resnet.resnet34(pretrained=True, num_classes=testset.class_num)
    elif args.network == 'resnet50':
        model = resnet.resnet50(pretrained=True, num_classes=testset.class_num)
    else:
        raise ValueError('Network %s not supported!' % (args.network))
    ckpt = torch.load(args.ckpt)['model']

    current_model = model.state_dict()
    ckpt_wo_fc = {k: v for k, v in ckpt.items() if k in current_model and 'fc' not in k}
    current_model.update(ckpt_wo_fc)
    model.load_state_dict(current_model)
    model = model.cuda()

    fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets = compute_roc(model, testloader)
    compute_roc_metrics(fpr, tpr, thresholds)


if __name__ == '__main__':
    main()
