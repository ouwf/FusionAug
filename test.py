"""
    Evaluate the verification performance of provided ckpt on the testing set of FVUSM
    Example: python3 -u ./test.py --ckpt ${ckpt} --data ${data} --dataset_name ${dataset} --network ${network}
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from models.models import ResNets
from eval.evaluation import evaluate_verification
from data.dataset import VeinDataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help="checkpoint path")
    parser.add_argument('--data', type=str, help="path of the testing dataset")
    parser.add_argument('--dataset_name', type=str, default='FVUSM', help="name of the dataset")
    parser.add_argument('--network', type=str, default='resnet18', help="name of the network")
    parser.add_argument('--simple_eval', action='store_true', help="whether to use simplified evaluation protocol")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset_name.lower() == 'fvusm':
        sample_per_class = 12
    else:
        raise ValueError('Dataset %s not exists!' % (args.dataset))
    # testing set
    normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    testset = VeinDataset(root=args.data, sample_per_class=sample_per_class, transform=transform_test)
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # we don't use the head's outputs during testing, so the head type has no effect on the test results.
    model = ResNets(backbone=args.network, head_type='cls').to(device)
    # load checkpoint
    ckpt = torch.load(args.ckpt)['model']
    current_dict = model.state_dict()
    filtered_dict = {k: v for k, v in ckpt.items() if k in current_dict and 'head' not in k}
    current_dict.update(filtered_dict)
    model.load_state_dict(current_dict)
    model = model.to(device)
    # evaluate biometric verification performance
    evaluate_verification(model, testloader, device, two_session=not args.simple_eval)


if __name__ == '__main__':
    main()
