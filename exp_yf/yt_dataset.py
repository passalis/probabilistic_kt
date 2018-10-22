import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class YT_Dataset(Dataset):
    def __init__(self, path='../data/youtube_faces.h5', split='train',
                 feature_type='transfer', seed=0):

        h5file = h5py.File(path, 'r')
        self.images = h5file['images']
        self.features = h5file['features']
        self.labels = np.int64(h5file['labels'])

        # Find which images/frames to use
        frequent_ids = np.where(np.bincount(self.labels) > 500)[0]

        idx = []
        for i in range(len(self.labels)):
            if self.labels[i] in frequent_ids:
                idx.append(i)

        np.random.seed(seed + 1)
        np.random.shuffle(idx)

        train_idx = idx[:200000]
        test_idx = idx[200000:200000 + 1000]
        if split == 'train':
            self.idx = train_idx
        else:
            self.idx = test_idx

        self.feature_type = feature_type

    def __getitem__(self, idx):
        idx = self.idx[idx]
        features = self.features[idx]
        label = self.labels[idx]

        # convert to torch tensors
        features = torch.from_numpy(np.float32(features))
        label = torch.from_numpy(np.int64([label]))

        if not self.feature_type == 'lbp':
            img = self.images[idx]
            # normalize image
            img = np.float32(img) / 255.0
            img = np.float32(img - 0.5).transpose((2, 0, 1))
            img = torch.from_numpy(img)

        if self.feature_type == 'transfer':
            return img, features, label
        elif self.feature_type == 'lbp':
            return features, label
        elif self.feature_type == 'image':
            return img, label

    def __len__(self):
        return len(self.idx)


def get_yt_loaders(batch_size=128, feature_type='transfer', seed=1):
    train_data = YT_Dataset(split='train', seed=seed, feature_type=feature_type)
    train_data_original = YT_Dataset(split='train', seed=seed, feature_type=feature_type)
    test_data = YT_Dataset(split='test', seed=seed, feature_type=feature_type)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=1, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=1, pin_memory=False)
    train_loader_original = torch.utils.data.DataLoader(train_data_original, batch_size=batch_size, shuffle=False,
                                                        num_workers=1, pin_memory=False)

    return train_loader, test_loader, train_loader_original
