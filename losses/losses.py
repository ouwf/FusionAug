import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.triplet_selector import RandomNegativeTripletSelector


class CosFace(nn.Module):
    def __init__(self, s=30.0, m=0.2):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, input, labels):
        # input: size = B x num_class
        cos = input
        one_hot = torch.zeros(cos.size()).cuda()
        one_hot = one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = self.s * (cos - one_hot * self.m)

        softmax_output = F.log_softmax(output, dim=1)
        loss = -1 * softmax_output.gather(1, labels.view(-1, 1))
        loss = loss.mean()

        return loss


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin=0.2, is_distance=True):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = RandomNegativeTripletSelector(margin=margin, is_distance=is_distance)
        self.is_distance = is_distance

    def forward(self, embeddings, target):
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        triplets = self.triplet_selector.get_triplets(embeddings_normalized, target)

        if embeddings_normalized.is_cuda:
            triplets = triplets.cuda()
        if self.is_distance:
            ap_distances = (embeddings_normalized[triplets[:, 0]] - embeddings_normalized[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embeddings_normalized[triplets[:, 0]] - embeddings_normalized[triplets[:, 2]]).pow(2).sum(1)
            losses = F.relu(ap_distances - an_distances + self.margin)

        else:
            ap_distances = (embeddings_normalized[triplets[:, 0]] * embeddings_normalized[triplets[:, 1]]).sum(1)
            an_distances = (embeddings_normalized[triplets[:, 0]] * embeddings_normalized[triplets[:, 2]]).sum(1)
            losses = 2*F.relu(an_distances - ap_distances + self.margin)
        return losses.mean(), len(triplets)
