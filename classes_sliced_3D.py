from tensorflow.keras.utils import Sequence

import numpy as np
import h5py
import warnings

import albumentations as A


class SliceDataGenerator(Sequence):
    def __init__(self, vol_labels, hdf5_fname, hdf5_groups, to_fit=True, n_slices=3, output_rep=1, batch_size=5, shuffle=True, augment=True):
        self.vol_labels = vol_labels
        self.hdf5_fname = hdf5_fname
        self.hdf5_groups = hdf5_groups
        self.to_fit = to_fit
        self.n_slices = n_slices
        self.output_rep = output_rep
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.cache = {}

        self.indexes = None

        self.augment_transform = A.GridDistortion(num_steps=5, p=0.5)

        self.hdf5_fid = h5py.File(self.hdf5_fname, 'r')

        self.data_list = self._create_data_list()
        
        print('#All patches: {}'.format(len(self.data_list)))

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.data_list) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _create_data_list(self):

        data_list = list()
        
        for vol_idx in self.vol_labels:
            volume=np.array(self.hdf5_fid[self.hdf5_groups[1]][vol_idx])
            H, W, D = volume.shape[:3]

            H_adj = H-self.n_slices
            W_adj = W-self.n_slices
            D_adj = D-self.n_slices

            #data_list += [(vol_idx, 'H', n) for n in range(H_adj)]
            #data_list += [(vol_idx, 'W', n) for n in range(W_adj)]
            #data_list += [(vol_idx, 'D', n) for n in range(D_adj)]

            for n in range(H_adj):
                if np.any(volume[n:n+self.n_slices, :, :, :]):
                    data_list.append((vol_idx, 'H', n))
            for n in range(W_adj):
                if np.any(volume[:, n:n+self.n_slices, :, :]):
                    data_list.append((vol_idx, 'W', n))
            for n in range(D_adj):
                if np.any(volume[:, :, n:n+self.n_slices, :]):
                    data_list.append((vol_idx, 'D', n))


       
        return data_list

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        def load_slices(inds, group):
            slice_list = []
            for i in inds:
                vol_idx, slice_dir, n = self.data_list[i]

                cache_key = vol_idx, slice_dir, n, group
                if  cache_key in self.cache:
                    vol_slice = self.cache[vol_idx, slice_dir, n, group]
                else:
                    volume=np.array(self.hdf5_fid[group][vol_idx])

                    if volume.ndim == 3:
                        volume = np.expand_dims(volume, 3)
                    
                    if slice_dir == 'H':
                        vol_slice = volume[n:n+self.n_slices, :, :, :]
                        vol_slice = vol_slice.transpose(1, 2, 0, 3)
                    elif slice_dir == 'W':
                        vol_slice = volume[:, n:n+self.n_slices, :, :]
                        vol_slice = vol_slice.transpose(0, 2, 1, 3)
                    elif slice_dir == 'D':
                        vol_slice = volume[:, :, n:n+self.n_slices, :]

                    vol_slice = np.array(vol_slice, dtype=np.float32)
                    vol_slice = np.reshape(vol_slice, (vol_slice.shape[0], vol_slice.shape[1], -1))
                    self.cache[cache_key] = vol_slice

                slice_list.append(vol_slice)

            H_common = np.max([s.shape[0] for s in slice_list])
            W_common = np.max([s.shape[1] for s in slice_list])

            H_common = int(np.ceil(H_common/2**(4-1)))*2**(4-1)
            W_common = int(np.ceil(W_common/2**(4-1)))*2**(4-1)

            slice_array = np.array([np.pad(s, ((0, H_common-s.shape[0]), (0, W_common-s.shape[1]), (0, 0))) for s in slice_list])

            return slice_array

        X = load_slices(indexes, self.hdf5_groups[0])

        if self.to_fit:
            y = load_slices(indexes, self.hdf5_groups[1])

            if self.augment:
                for n in range(X.shape[0]):
                    augmented = self.augment_transform(image=X[n], mask=y[n])
                    X[n] = augmented['image']
                    y[n] = augmented['mask']

            if self.output_rep > 1:
                y = [y for n in range(self.output_rep)]
            return X, y
        else:
            return X
