import datetime
from ptflops import get_model_complexity_info
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def cal_flops():
    from networks import resnet
    import time
    input_size = [3, 64, 144]
    model = resnet.resnet18()
    flops, params = get_model_complexity_info(model, tuple(input_size), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops)
    print('params: ', params)
    aver_network_time(model, input_size=input_size, use_gpu=True)


def aver_network_time(model, input_size, use_gpu=False):
    import time
    import cv2
    if use_gpu:
        model = model.cuda()
    input_size.insert(0, 32)
    sum = 0
    count = 100
    for i in range(count):
        input = torch.rand(input_size)
        if use_gpu:
            input = input.cuda()
        t0 = time.time()
        output = model(input)
        sum += (time.time() - t0)

    print("aver feature extraction time: %.6f s)" % (sum / count))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')


if __name__ == '__main__':
    cal_flops()
