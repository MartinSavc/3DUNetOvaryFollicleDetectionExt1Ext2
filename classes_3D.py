from tensorflow.keras.utils import Sequence

import numpy as np
import h5py
import warnings

from scipy.ndimage import zoom, gaussian_gradient_magnitude
from augment_3D import augment_data

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, vol_labels, hdf5_fname, hdf5_groups,
                 to_fit=True, batch_size=32, vol_shape=(256, 256, 256), patch_shape=None,
                 patch_stride=1, rescale=1., shuffle=True, augment=False, augment_param=None, pad_to_cube=False, label_edges=False, add_label_edge=False):
        """Initialization
        :param vol_labels: list of volume labels (file name prefix)
        :param hdf5_fid: fname of HDF5 file with data
        :param hdf5_groups: groups in HDF5 file
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param vol_shape: tuple indicating volume dimension
        :param patch_shape: tuple indicating patch dimension
        :param patch_stride: stride when moving patch across volume
        :param rescale: rescaling volume data        
        :param shuffle: True to shuffle label indexes after every epoch
        :param augment: True to augment
        :param augment_param: parameter for the augmentation        
        :param pad_to_cube: True to pad the original volume to a cube before processing
        :param label_edges: Extract edges of labeled segments, use gaussian gradient magnitude.
        :param add_label_edge: Adds an extra channel to the output, containing label edges estimated using gaussian gradien magnitude.
        :param num_patches_per_volume: Every volume is divided on overlapping and non-zero 3D patches; number of non-zero patches in volume
        :param data_list: list of all patches in all volumes, [volume, startX, startY, startZ]
        """

        self.vol_labels = vol_labels
        self.hdf5_fname = hdf5_fname
        self.hdf5_groups = hdf5_groups
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.vol_shape = vol_shape        
        self.patch_shape= patch_shape
        self.patch_stride = patch_stride
        self.rescale = rescale                        
        self.shuffle = shuffle
        self.augment = augment
        self.augment_param = augment_param
        self.pad_to_cube = pad_to_cube
        self.label_edges = label_edges
        self.add_label_edge = add_label_edge

        self.hdf5_fid = h5py.File(self.hdf5_fname, 'r')

        self.num_patches_per_volume, self.data_list = self._create_data_list()

        x_example = self.hdf5_fid[self.hdf5_groups[0]][self.data_list[0][0]]
        y_example = self.hdf5_fid[self.hdf5_groups[1]][self.data_list[0][0]]
        self.x_chns = 1 if x_example.ndim==3 else x_example.shape[3]
        self.y_chns = 1 if y_example.ndim==3 else y_example.shape[3]

        print('#Patches_per_volume: {}'.format(self.num_patches_per_volume))
        print('#All patches: {}'.format(len(self.data_list)))
                                      
        self.on_epoch_end()    
                
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Find list of IDs
        list_IDs = [self.data_list[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs)
        
        if self.to_fit:
            y = self._generate_y(list_IDs)
            
            if self.augment:
                                            
                for i in range(self.batch_size):                    
                    augX, augy = augment_data(X[i,:], y[i,:], **self.augment_param)
                    X[i] = augX
                    y[i] = augy
                    
            return  X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs):
        """Generates data containing batch_size volumes
        :param list_IDs: list of data about patches
        :return: batch of patch_volumes
        """
        
        chns=self.x_chns
        if self.patch_shape is not None:
            X = np.empty((self.batch_size, chns, *self.patch_shape))
        else:
            X = np.empty((self.batch_size, chns, *self.vol_shape))
                                
        # Generate data
        for i in range(len(list_IDs)):
            # Store sample
                                 
             vol_patch = self._load_grayscale_volume_patch(self.hdf5_groups[0], list_IDs[i])
             X[i,] = self.rescale*vol_patch

        return X

    def _generate_y(self, list_IDs):
        """Generates data containing batch_size masks
        :param list_IDs: list of data about patches
        :return: batch of patch_volumes masks
        """
        
        chns=self.y_chns
        if self.add_label_edge:
            chns = 2*chns
        if self.patch_shape is not None:
            y = np.empty((self.batch_size, chns, *self.patch_shape))
        else:
            y = np.empty((self.batch_size, chns, *self.vol_shape))                
        
        # Generate data
        for i in range(len(list_IDs)):
            # Store sample
            vol_patch = self._load_grayscale_volume_patch(self.hdf5_groups[1], list_IDs[i])
            if self.label_edges or self.add_label_edge:
                vol_edge = gaussian_gradient_magnitude(vol_patch*1., 2)
                vol_edge /= vol_edge.max()+1e-12

                if self.add_label_edge:
                    vol_patch = (vol_patch, vol_edge)
                else:
                    vol_patch = vol_edge
            y[i,] = vol_patch

        return y
    
    def _load_grayscale_volume_patch(self, group_name, patch_data):
        """Load grayscale volume
        :param group_name: name of group to read
        :param patch_data: data about patch
        :return: loaded patch_volume
        """
        
#        print("PATCH DATA: {}".format(patch_data))
        
        dset_name = patch_data[0]
        x = patch_data[1]
        y = patch_data[2]
        z = patch_data[3]

        volume = np.array(self.hdf5_fid[group_name][dset_name])

        if volume.ndim == 3:
            volume = volume.reshape(volume.shape+(1,))
        volume = volume.transpose(3, 2, 1, 0)

        if self.pad_to_cube:
            volume = self._pad_volume_to_cube(volume)
        C = volume.shape[0]


        if self.patch_shape is not None:                        
            volume_patch = volume[:, x:x+self.patch_shape[0], y:y+self.patch_shape[1], z:z+self.patch_shape[2]]
        else:
            zoom_fc = [self.vol_shape[0]/volume.shape[1], self.vol_shape[1]/volume.shape[2], self.vol_shape[2]/volume.shape[3]]            
            volume_patch = np.zeros((C,)+self.vol_shape)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")            
                for c in range(C):
                    volume_patch[c, :, :, :] = zoom(volume[c, :, :, :], zoom=zoom_fc)
        
        return volume_patch
    
    def _pad_volume_to_cube(self, volume):
        C, D, H, W = volume.shape
        edge = max(D, H, W)
        new_volume = np.zeros((C, edge, edge, edge), dtype=volume.dtype)
        new_volume[:, :D, :H, :W] = volume
        return new_volume

    def _create_data_list(self):

        data_list = list()
        num_patches_per_volume = list()
        
        for vol_idx in self.vol_labels:
            volume=np.array(self.hdf5_fid[self.hdf5_groups[0]][vol_idx])
            if volume.ndim == 3:
                volume = volume.reshape(volume.shape+(1,))
            volume = volume.transpose(3, 2, 1, 0)


            if self.pad_to_cube:
                volume = self._pad_volume_to_cube(volume)
            
            volume_list = list()                        
#            print("vol_idx:{}  (X:{}, Y:{}, Z:{})".format(vol_idx, volume.shape[0], volume.shape[1], volume.shape[2], ))
            
            if self.patch_shape is not None:
                
                rangeX = _create_range(volume.shape[1], self.patch_shape[0], self.patch_stride)
                rangeY = _create_range(volume.shape[2], self.patch_shape[1], self.patch_stride)
                rangeZ = _create_range(volume.shape[3], self.patch_shape[2], self.patch_stride)
                
                for x in rangeX:
                    for y in rangeY:
                        for z in rangeZ:
                            vol = volume[:, x:x+self.patch_shape[0], y:y+self.patch_shape[1], z:z+self.patch_shape[2]]
                            
                            if (np.any(vol!= 0)):
                                volume_list.append([vol_idx, x, y, z])
            else:
                volume_list.append([vol_idx, 0, 0, 0])
            
            num_patches_per_volume.append([len(volume_list)])
            data_list.extend(volume_list)
            
       
        return num_patches_per_volume, data_list
    
def _create_range (DIM, PATCH, STRIDE):

    if (STRIDE<=0):
        
        if (STRIDE == 0): 
            STRIDE=-1
        
        STRIDE = abs(STRIDE)
        
        rangeList = [*range(0,DIM-PATCH-1, STRIDE), DIM-PATCH-1]
        rangeList = np.unique(rangeList)    
        
    else:        
        rangeList = np.array([])
        ind = 0
    
        while (ind <= DIM-PATCH-1):
            
            rangeList = np.append(rangeList, ind)
            ind = ind + np.random.randint(low=round(PATCH/8), high=STRIDE+1)        
    
        rangeList = np.append(rangeList, DIM-PATCH-1)
        rangeList = np.unique(rangeList)
            
    return rangeList.astype(int)
    
