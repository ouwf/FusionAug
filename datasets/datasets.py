import torch
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import numpy as np
import glob


class ImageDataset(Dataset):
    def __init__(self, root, transforms, sample_per_class, mode='train', inter_aug=''):
        self.transform = transforms
        self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))
        self.class_num = len(self.files) // sample_per_class // 2
        self.labels = np.arange(self.class_num).repeat(sample_per_class)
        self.img_data = []
        if mode == 'train':
            for i in np.arange(0, len(self.files) // 2):
                with open(os.path.join(root, self.files[i]), 'rb') as f:
                    img = Image.open(f)
                    self.img_data.append(img.copy())
                    img.close()
            # self.img_data = [Image.open(os.path.join(root, self.files[i])) for i in np.arange(0, len(self.files) // 2)]
        elif mode == 'test':
            for i in np.arange(len(self.files) // 2, len(self.files)):
                with open(os.path.join(root, self.files[i]), 'rb') as f:
                    img = Image.open(f)
                    self.img_data.append(img.copy())
                    img.close()
            # self.img_data = [Image.open(os.path.join(root, self.files[i])) for i in np.arange(len(self.files) // 2, len(self.files))]

        if inter_aug == 'LF':  # left-right flip
            self.img_data.extend([self.img_data[i].transpose(Image.FLIP_LEFT_RIGHT) for i in np.arange(0, len(self.img_data))])
            aug_classes = np.arange(self.class_num, self.class_num * 2).repeat(sample_per_class)
            self.labels = np.concatenate([self.labels, aug_classes])
            self.class_num = self.class_num * 2
        elif inter_aug == 'TB':  # top-bottom flip
            self.img_data.extend([self.img_data[i].transpose(Image.FLIP_TOP_BOTTOM) for i in np.arange(0, len(self.img_data))])
            aug_classes = np.arange(self.class_num, self.class_num * 2).repeat(sample_per_class)
            self.labels = np.concatenate([self.labels, aug_classes])
            self.class_num = self.class_num * 2
        print(len(self.labels))
        print(len(self.img_data))
        pass

    def __getitem__(self, index):
        image = self.img_data[index]
        image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.img_data)


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        self.labels = np.array(dataset.labels)

        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(np.random.choice(self.label_to_indices[class_], self.n_samples, replace=False))
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
