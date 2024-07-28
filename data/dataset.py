import torch
import os
import cv2
from PIL import Image
import numpy as np
import glob
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader


class VeinDataset(data.Dataset):
    """ Vein Image Dataset
    Args:
        root: dataset root
        sample_per_class: number of samples per class
        num_samples: number of samples to be used in the dataset
        transform: transform to be applied on a sample
        inter_aug: inter-class data augmentation, valid options: {'LR', 'TB'}
    """
    def __init__(self, root, transform, sample_per_class, num_samples=None, inter_aug=None):
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))
        if num_samples is not None and num_samples < len(self.files) and num_samples > 0:
            self.files = self.files[:num_samples]
        self.class_num = len(self.files) // sample_per_class
        self.labels = np.arange(self.class_num).repeat(sample_per_class)
        self.img_data = [cv2.imread(os.path.join(root, self.files[i])) for i in np.arange(0, len(self.files))]
        if inter_aug is not None:
            if inter_aug == 'LR':  # left-right (horizontal) flipping for inter-class data augmentation
                # self.img_data.extend([self.img_data[i].transpose(Image.FLIP_LEFT_RIGHT) for i in np.arange(0, len(self.img_data))])
                self.img_data.extend([self.img_data[i][:, ::-1, :] for i in np.arange(0, len(self.img_data))])
            elif inter_aug == 'TB':  # top-bottom (vertical) flipping for inter-class data augmentation
                # self.img_data.extend([self.img_data[i].transpose(Image.FLIP_TOP_BOTTOM) for i in np.arange(0, len(self.img_data))])
                self.img_data.extend([self.img_data[i][::-1, :, :] for i in np.arange(0, len(self.img_data))])
            else:
                ValueError(f"{inter_aug} is not a valid option.")
            aug_classes = np.arange(self.class_num, self.class_num * 2).repeat(sample_per_class)
            self.labels = np.concatenate([self.labels, aug_classes])
            self.class_num = self.class_num * 2
        print(f"{len(self.img_data)} images loaded.")

    def __getitem__(self, index):
        image = Image.fromarray(self.img_data[index])
        if self.transform is not None:
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


def get_transforms(dataset, data_aug=True):
    # normalize to [-1, 1]
    normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    transform_train = []
    if data_aug:
        if dataset.lower() == 'fvusm':
            transform_train.append(transforms.RandomResizedCrop(size=(64, 144), scale=(0.5, 1.0), ratio=(2.25, 2.25)))
            transform_train.append(transforms.RandomRotation(degrees=3))
            transform_train.append(transforms.RandomPerspective(distortion_scale=0.3, p=0.9))
            transform_train.append(transforms.ColorJitter(brightness=0.7, contrast=0.7))
        else:
            ValueError(f"Dataset {dataset} not supported!")
    transform_train.append(transforms.ToTensor())
    transform_train.append(normalize)
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    return transform_train, transform_test
