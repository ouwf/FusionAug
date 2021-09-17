from itertools import combinations
import numpy as np
import torch


def pdist(vectors, is_distance=True):
    if is_distance:
        # the embeddings are already l2-normalized, thus has norm 1
        distance_matrix = - 2 * np.matmul(vectors, vectors.T) + 2.0
    else:
        # cosine similarity
        distance_matrix = np.matmul(vectors, vectors.T)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, is_distance=True, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.is_distance = is_distance

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        distance_matrix = pdist(embeddings, self.is_distance)

        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            negative_indices = np.where(np.logical_not(label_mask))[0]
            if len(label_indices) < 2:
                continue
            # anchor_positives = np.array(list(combinations(label_indices, 2)))  # All anchor-positive pairs
            anchor_positives = np.array(list(combinations(label_indices, 2)) + list(combinations(label_indices[::-1], 2)))  # All anchor-positive pairs
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                if self.is_distance:
                    loss_values = ap_distance - distance_matrix[anchor_positive[0], negative_indices] + self.margin
                    # an_distance = distance_matrix[anchor_positive[0], negative_indices]
                    # pn_distance = distance_matrix[anchor_positive[1], negative_indices]
                    # loss_values = ap_distance - np.array([an_distance, pn_distance]).min(0) + self.margin
                else:
                    loss_values = distance_matrix[anchor_positive[0], negative_indices] - ap_distance + self.margin
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
        triplets = torch.LongTensor(triplets)
        return triplets


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None
    # return hard_negative


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None
    # return np.argsort(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def HardestNegativeTripletSelector(margin, is_distance=True, cpu=True): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 is_distance=is_distance,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, is_distance=True, cpu=True): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                is_distance=is_distance,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, is_distance=True, cpu=True): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  is_distance=is_distance,
                                                                                  cpu=cpu)
