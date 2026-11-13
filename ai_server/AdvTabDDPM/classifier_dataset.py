import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset

class TracesDataset(Dataset):
    def __init__(self, dataset_pkls, key, normalize=True, path='data/pickle/', norm_par_path="", sanitize=True, transform=None, target_transform=None):
        self.obs_input, self.obs_labels = None, None
        self.path = path
        self.norm_params = None

        if isinstance(dataset_pkls, str):
            dataset_pkls = [dataset_pkls]
        
        if norm_par_path != "":
            self.norm_params = pickle.load(open(os.path.join(path, norm_par_path), 'rb'))

        for p in dataset_pkls:
            dataset = pickle.load(open(os.path.join(self.path, p), 'rb'))
            # let's retrieve the relative normalization parameters for this specific dataset
            # ds_filename = os.path.basename(p)
            # ds_dir = os.path.dirname(p)
            # trials_str, filemarker = ds_filename.split('__')[2:]
            # normp_filename = "cols_maxmin__" + trials_str + "__" + filemarker
            # norm_params = pickle.load(open(os.path.join(self.path, ds_dir, normp_filename), 'rb'))

            if normalize:
                norm_key = 'norm'
            else:
                norm_key = 'no_norm'
            if (self.obs_input is None) and (self.obs_labels is None):
                self.obs_input, self.obs_labels = dataset[key]['samples'][norm_key], dataset[key]['labels']
            else:
                self.obs_input = torch.cat((self.obs_input, dataset[key]['samples'][norm_key]), dim=0)
                self.obs_labels = torch.cat((self.obs_labels, dataset[key]['labels']), dim=0)

        # sanitize columns: remove columns with no variation of data (after relabeling)
        if sanitize:
            obs_std = np.std(self.obs_input.numpy(), axis=(0, 1))
            features_to_remove = np.where(obs_std == 0)[0]
            # assert np.all(features_to_remove == self.norm_params['info']['exclude_cols_ix']), "Double check that the column to exclude correpsonds to the ones in the common paramters" # modify
            all_feats = torch.arange(0, self.obs_input.shape[-1])
            indexes_to_keep = [i for i in range(len(all_feats)) if i not in self.norm_params['info']['exclude_cols_ix']]
            self.obs_input = self.obs_input[:, :, indexes_to_keep]


        self.sanitized = sanitize

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.obs_input)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.obs_labels.iloc[idx, 0])
        #image = read_image(img_path)       # we load at run time instead that in __init__
        #label = self.obs_labels.iloc[idx, 1]
        obs = self.obs_input[idx, :, :]
        label = self.obs_labels[idx]

        if self.transform:
            obs = self.transform(obs)
        if self.target_transform:
            label = self.target_transform(label)
        return obs, label
    
    def info(self):
        unique_labels, unique_count = np.unique(np.array(self.obs_labels), return_counts=True)
        ds_info = {
            'numfeats': self.obs_input.shape[2],
            'slice_len': self.obs_input.shape[1],
            'numsamps': self.obs_input.shape[0],
            'nclasses': len(unique_labels),
            'samples_per_class': unique_count
        }
        return ds_info