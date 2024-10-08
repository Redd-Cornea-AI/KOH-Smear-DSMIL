import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte

np.random.seed(0)

class Dataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list.iloc[idx, 0]
        img = Image.open(temp_path)
        img = transforms.functional.to_tensor(img)
        if self.transform:
            sample = self.transform(img)
        return sample

class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img 

class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        train_dataset = Dataset(csv_file='all_patches.csv', transform=SimCLRDataTransform(data_augment))
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.5 * self.s, 0.5 * self.s, 0.5 * self.s, 0.1 * self.s)
        data_transforms = transforms.Compose([ToPIL(),
                                              transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomRotation(degrees=20),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              # gaussian blurring doesn't work well on almost monotonous images like here.
                                              transforms.RandomApply([GaussianBlur(kernel_size=int(0.015 * self.input_shape[0]))], p=0.2),
                                              transforms.ToTensor(),
                                              ])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices) # turning off shuffling to keep multiple patient scan together

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        
        # print(train_idx[0])
        # print(train_idx[-1])
        # print(valid_idx[0]) # check if val set is representative with or without shuffling
        # print(valid_idx[-1])

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
