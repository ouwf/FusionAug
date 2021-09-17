import torch
from sklearn import preprocessing, metrics
from itertools import combinations
from utils import utils
import numpy as np


def get_embeddings(model, testloader):
    model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for data, target in testloader:
            data = data.cuda()
            f = model(data)
            f = f[0] if isinstance(f, tuple) else f
            embeddings.append(f.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())
    embeddings = np.vstack(embeddings)
    embeddings = preprocessing.normalize(embeddings)
    targets = np.concatenate(targets)
    model.train()
    return embeddings, targets


def compute_roc(model, testloader):
    embeddings, targets = get_embeddings(model, testloader)
    emb_num = len(embeddings)
    # Cosine similarity between any two pairs, note that all embeddings are l2-normalized
    scores = np.matmul(embeddings, embeddings.T)
    class_num = testloader.dataset.class_num
    samples_per_class = emb_num // class_num
    # define matching pairs
    intra_class_combinations = np.array(list(combinations(range(samples_per_class), 2)))
    match_pairs = [i*samples_per_class + intra_class_combinations for i in range(class_num)]
    match_pairs = np.concatenate(match_pairs, axis=0)
    scores_match = scores[match_pairs[:, 0], match_pairs[:, 1]]
    labels_match = np.ones(len(match_pairs))

    # define imposter pairs
    inter_class_combinations = np.array(list(combinations(range(class_num), 2)))
    imposter_pairs = [np.expand_dims(i*samples_per_class, axis=0) for i in inter_class_combinations]
    imposter_pairs = np.concatenate(imposter_pairs, axis=0)
    scores_imposter = scores[imposter_pairs[:, 0], imposter_pairs[:, 1]]
    labels_imposter = np.zeros(len(imposter_pairs))

    # merge matching pairs and imposter pairs and assign labels
    all_scores = np.concatenate((scores_match, scores_imposter))
    all_labels = np.concatenate((labels_match, labels_imposter))
    # compute roc, auc and eer
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    return fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets


def compute_roc_metrics(fpr, tpr, thresholds):
    fnr = 1 - tpr
    # find indices where EER, fpr100, fpr1000, fpr0, best acc occur
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    fpr100_idx = sum(fpr <= 0.01) - 1
    fpr1000_idx = sum(fpr <= 0.001) - 1
    fpr10000_idx = sum(fpr <= 0.0001) - 1
    fpr0_idx = sum(fpr <= 0.0) - 1

    # compute EER, FRR@FAR=0.01, FRR@FAR=0.001, FRR@FAR=0
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    fpr100 = fnr[fpr100_idx]
    fpr1000 = fnr[fpr1000_idx]
    fpr10000 = fnr[fpr10000_idx]
    fpr0 = fnr[fpr0_idx]

    metrics = (eer, fpr100, fpr1000, fpr10000, fpr0)
    metrics_thred = (thresholds[eer_idx], thresholds[fpr100_idx], thresholds[fpr1000_idx], thresholds[fpr10000_idx], thresholds[fpr0_idx])
    print(utils.dt(), 'Performance evaluation...')
    print('EER:%.2f%%, FRR@FAR=0.01: %.2f%%, FRR@FAR=0.001: %.2f%%, FRR@FAR=0.0001: %.2f%%, FRR@FAR=0: %.2f%%, Aver: %.2f%%' %
          (eer * 100, fpr100 * 100, fpr1000 * 100, fpr10000 * 100, fpr0 * 100, np.mean(metrics) * 100))
    return metrics, metrics_thred


def evaluate_verification(model, testloader):
    fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets = compute_roc(model, testloader)
    roc_metrics, metrics_threds = compute_roc_metrics(fpr, tpr, thresholds)
    return roc_metrics, np.mean(roc_metrics)

