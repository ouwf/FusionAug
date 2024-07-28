import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from eval.evaluation import evaluate_verification
from torchvision.utils import save_image
import utils
import numpy as np
from loss.loss_functions import FusionLoss, CosFace, OnlineTripletLoss


class SupervisedTrainer:
    def __init__(self, network, loss, optimizer, device, **params):
        self.network = network
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = params['max_epochs']
        self.lr_scheduler = params["lr_scheduler"]
        self.batch_size = params['batch_size']
        self.save_image = params['save_image']
        if loss == 'softmax':
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == 'tripletloss':
            self.loss = OnlineTripletLoss(margin=params["hard_margin"])
        elif loss == 'cosface':
            self.loss = CosFace(s=params['s'], m=params['m'])
        elif loss == 'fusionloss':
            tripletloss = OnlineTripletLoss(margin=params['hard_margin'])
            cosface = CosFace(s=params['s'], m=params['m'])
            self.loss = FusionLoss(cls_loss=cosface, metric_loss=tripletloss, w_cls=params['w_cls'], w_metric=params['w_metric'])
        else:
            raise ValueError('Loss %s not supported!' % loss)
        self.args = params['args']

    def train(self, trainloader, testloader):
        loss_stats = utils.AverageMeter()
        best_result, best_snapshot = None, None
        for epoch in range(self.max_epochs):
            for batch_idx, (data, labels) in enumerate(trainloader):
                data, labels = data.to(self.device), labels.to(self.device)
                if self.save_image:
                    os.makedirs("./augmented_images", exist_ok=True)
                    save_image(data[:16], "augmented_images/%d.bmp" % (batch_idx % 50), nrow=4,
                               normalize=True, range=(-1.0, 1.0))
                loss = self.update(data, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_stats.update(loss.item())
                print(utils.dt(), 'Epoch:[%d]-[%d/%d] batchLoss:%.4f averLoss:%.4f' %
                      (epoch, batch_idx, len(trainloader), loss_stats.val, loss_stats.avg))

            roc, aver, auc = evaluate_verification(self.network, testloader, self.device, not self.args.simple_eval)
            self.lr_scheduler.step()
            # save the current best model based on eer
            best_result, best_snapshot = \
                save_model(self.network, {'metrics': roc, 'eer': roc[0], 'epoch': epoch}, best_result, best_snapshot, self.args)
            print("End of epoch {}".format(epoch))

        print(utils.dt(), 'Training completed.')
        print(utils.dt(), '------------------Best Results---------------------')
        epoch, roc = best_result['epoch'], best_result['metrics']
        print(utils.dt(),
              'EER: %.2f%%, FPR100:%.2f%%, FPR1000:%.2f%%, FPR10000:%.2f%%, FPR0:%.2f%%, Aver: %.2f%% @ epoch %d' %
              (roc[0] * 100, roc[1] * 100, roc[2] * 100, roc[3] * 100, roc[4] * 100, np.mean(roc) * 100, epoch))

    def update(self, data, labels):
        features, logits = self.network(data)
        if isinstance(self.loss, OnlineTripletLoss):
            loss = self.loss(features, labels)
        elif isinstance(self.loss, (CosFace, torch.nn.CrossEntropyLoss)):
            loss = self.loss(logits, labels)
        elif isinstance(self.loss, FusionLoss):
            loss = self.loss((features, logits), labels)
        return loss


def save_model(model, current_result, best_result, best_snapshot, args):
    eer = current_result['eer']
    epoch = current_result['epoch']
    prefix = 'seed=%d_dataset=%s_network=%s_loss=%s' % (args.seed, args.dataset_name, args.network, args.loss)
    os.makedirs("snapshots", exist_ok=True)
    # save the current best model
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