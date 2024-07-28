import torch
from sklearn import preprocessing, metrics
from itertools import combinations
import numpy as np
import sklearn
import datetime


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')


def get_embeddings(model, testloader, device):
    model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            f = model(data)
            f = f[0] if isinstance(f, tuple) else f
            embeddings.append(f.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())
    embeddings = np.vstack(embeddings)
    embeddings = preprocessing.normalize(embeddings)
    targets = np.concatenate(targets)
    model.train()
    return embeddings, targets


def compute_roc(embeddings, class_num):
    emb_num = len(embeddings)
    # Cosine similarity between any two pairs, note that all embeddings are l2-normalized
    scores = np.matmul(embeddings, embeddings.T)
    samples_per_class = emb_num // class_num
    # define matching pairs
    intra_class_combinations = np.array(list(combinations(range(samples_per_class), 2)))
    genuine_pairs = [i*samples_per_class + intra_class_combinations for i in range(class_num)]
    genuine_pairs = np.concatenate(genuine_pairs, axis=0)
    genuine_scores = scores[genuine_pairs[:, 0], genuine_pairs[:, 1]]
    labels_genuine = np.ones(len(genuine_pairs))

    # define imposter pairs
    inter_class_combinations = np.array(list(combinations(range(class_num), 2)))
    imposter_pairs = [np.expand_dims(i*samples_per_class, axis=0) for i in inter_class_combinations]
    imposter_pairs = np.concatenate(imposter_pairs, axis=0)
    imposter_scores = scores[imposter_pairs[:, 0], imposter_pairs[:, 1]]
    labels_imposter = np.zeros(len(imposter_pairs))
    print(f"Number of genunie pairs: {len(genuine_scores)}")
    print(f"Number of imposter pairs: {len(imposter_scores)}")
    # merge matching pairs and imposter pairs and assign labels
    all_scores = np.concatenate((genuine_scores, imposter_scores))
    all_labels = np.concatenate((labels_genuine, labels_imposter))
    # compute roc, auc and eer
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    return fpr, tpr, thresholds, genuine_scores, imposter_scores


def compute_roc_two_session(embeddings, class_num):
    emb_num = len(embeddings)
    # Cosine similarity between any two pairs, note that all embeddings are l2-normalized
    scores = np.matmul(embeddings, embeddings.T)
    samples_per_class = emb_num // class_num

    ind_all = np.arange(0, emb_num)
    ind_session_1 = []
    ind_session_2 = []
    for i in range(0, emb_num, samples_per_class):
        ind_session_1.append(ind_all[i: i + samples_per_class // 2])
        ind_session_2.append(ind_all[i + samples_per_class // 2: i + samples_per_class])
    # define genuine pairs
    genuine_pairs = []
    for i in range(0, class_num):
        s1 = ind_session_1[i]
        s2 = ind_session_2[i]
        # genuine_pairs.extend([[x, y] for x in s1 for y in s2])
        genuine_pairs.append(np.array(np.meshgrid(s1, s2)).T.reshape(-1, 2))
    genuine_pairs = np.concatenate(genuine_pairs, 0)
    # genuine_pairs = np.array(genuine_pairs)
    genuine_scores = scores[genuine_pairs[:, 0], genuine_pairs[:, 1]]
    labels_genuine = np.ones(len(genuine_scores))
    # define imposter pairs
    imposter_pairs = []
    for i in range(0, class_num):
        s1 = ind_session_1[i]
        ind_session_2_copy = ind_session_2.copy()
        ind_session_2_copy.pop(i)
        s2 = np.concatenate(ind_session_2_copy, 0)
        # imposter_pairs.extend([[x, y] for x in s1 for y in s2])
        imposter_pairs.append(np.array(np.meshgrid(s1, s2)).T.reshape(-1, 2))
    imposter_pairs = np.concatenate(imposter_pairs, 0)
    # imposter_pairs = np.array(imposter_pairs)
    imposter_scores = scores[imposter_pairs[:, 0], imposter_pairs[:, 1]]
    labels_imposter = np.zeros(len(imposter_pairs))
    print(f"Number of genunie pairs: {len(genuine_scores)}")
    print(f"Number of imposter pairs: {len(imposter_scores)}")
    # merge matching pairs and imposter pairs and assign labels
    all_scores = np.concatenate((genuine_scores, imposter_scores))
    all_labels = np.concatenate((labels_genuine, labels_imposter))
    # compute roc, auc and eer
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    return fpr, tpr, thresholds, genuine_scores, imposter_scores


def compute_roc_metrics(fpr, tpr, thresholds):
    fnr = 1 - tpr
    # find indices where EER, fpr100, fpr1000, fpr10000, fpr0 occur
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    fpr100_idx = sum(fpr <= 0.01) - 1
    fpr1000_idx = sum(fpr <= 0.001) - 1
    fpr10000_idx = sum(fpr <= 0.0001) - 1
    fpr0_idx = sum(fpr <= 0.0) - 1

    # compute EER, FRR@FAR=0.01, FRR@FAR=0.001, FRR@FAR=0.0001, FRR@FAR=0
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    fpr100 = fnr[fpr100_idx]
    fpr1000 = fnr[fpr1000_idx]
    fpr10000 = fnr[fpr10000_idx]
    fpr0 = fnr[fpr0_idx]

    metrics = (eer, fpr100, fpr1000, fpr10000, fpr0)
    metrics_thred = (thresholds[eer_idx], thresholds[fpr100_idx], thresholds[fpr1000_idx], thresholds[fpr10000_idx], thresholds[fpr0_idx])
    AUC = sklearn.metrics.auc(fpr, tpr)
    print(dt(), 'Performance evaluation...')
    print('EER:%.2f%%, FRR@FAR=0.01: %.2f%%, FRR@FAR=0.001: %.2f%%, FRR@FAR=0.0001: %.2f%%, FRR@FAR=0: %.2f%%, Aver: %.2f%%, AUC:%.2f%%' %
          (eer * 100, fpr100 * 100, fpr1000 * 100, fpr10000 * 100, fpr0 * 100, np.mean(metrics) * 100, AUC * 100))
    return metrics, metrics_thred, AUC


def evaluate_verification(model, testloader, device, two_session=True):
    embeddings, targets = get_embeddings(model, testloader, device)
    if two_session:
        fpr, tpr, thresholds, scores_match, scores_imposter = compute_roc_two_session(embeddings, testloader.dataset.class_num)
    else:
        fpr, tpr, thresholds, scores_match, scores_imposter = compute_roc(embeddings, testloader.dataset.class_num)
    roc_metrics, metrics_threds, AUC = compute_roc_metrics(fpr, tpr, thresholds)
    return roc_metrics, np.mean(roc_metrics), AUC
