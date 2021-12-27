import os
import h5py
import argparse

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

#from unet3d.data import write_data_to_file, open_data_file
from unet3d.model.unet_guided import unet_guided_model_3d, DataGeneratorMod
from unet3d.training import load_old_model, train_model
from unet3d.metrics import dice_coeff, rho_coeff, acc_coeff, bce_dice_rho_loss, dice_loss, rho_loss, acc_loss


import tensorflow as tf
from tensorflow.keras.models import load_model


config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["kernel_size"] = (3, 3, 3)  # pool size for the max pooling operations
#config["patch_shape"] = (64, 64, 64)  # switch to None to train on the whole image
config["patch_shape"] = None  # switch to None to train on the whole image
config["vol_shape"] = (128, 128, 128)  # switch to None to train on the whole image
config['pad_to_cube'] = True
config['label_edges'] = False
config['add_label_edge'] = False
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution
config["batch_normalization"] = True

config["model_depth"] = 5
config["num_filters"] = 8
config['loss_weights'] = [1, 1, 1, 1, 1]

#config["batch_size"] = 4
config["batch_size"] = 1
config["n_epochs"] = 200  # cutoff the training after this many epochs
config["patience"] = 15  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training

#config["model_file"] = os.path.abspath("model_USOVA3D.h5")
config["model_file"] = "model_USOVA3D_unet_guided_v005.h5"
config["overwrite"] = True  # If True, will overwrite previous model file. If False, will use previously stored model file.

config["HDF5_train_fname"] = "../Data/Ovary/trainSet.h5"#"D:\\WORK\\USOVA3D\\Tensorflow_solutions\\Data\\trainSet.h5"
#config["HDF5_train_fname_LAB"] = "/home/bozidarp/3DCNN/Data/trainSet.h5" 

config["HDF5_group_names"] = ["Images", "Labels"]
config["random_seed"] = 42


def MyParser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='USOVA3D_3D Training Method')

    #parser.add_argument('-b', '--batch', type=int, help='Training batch size (int).', required=True)
    parser.add_argument('-b', '--batch', type=int,
                    help='Training batch size (int).', default=config["batch_size"])

    parser.add_argument('-e', '--epoch', type=int,
                    help='Number of training epoch (int)', default=config["n_epochs"])    

    parser.add_argument('-v', '--validationSplit', type=int,
                    help='Validation split rate', default=config["validation_split"])    

    parser.add_argument('-l', '--load', type=int,
                    help='Load weights/model (int)', default=0)    

    parser.add_argument('-r', '--rate', type=float,
                    help='Learning rate (float)', default=config["initial_learning_rate"])    

    parser.add_argument('-d', '--decay', type=float,
                    help='Decay rate (float)', default=config["learning_rate_drop"])    
    
    parser.add_argument('-p', '--patience', type=int,
                    help='LR on plateou patience (int)', default=config["patience"])    
    
    parser.add_argument('-s', '--stop', type=int,
                    help='Early stoping patience (int)', default=config["early_stop"])        

    parser.add_argument('-m', '--model', type=ascii,
                    help='Name of model to be stored and read from.', default=config["model_file"])
    
    parser.add_argument('--labServer', type=int,
                    help='Running on LAB server (1/0).', default=0)

    parser.add_argument('--modelDepth', type=int,
                    help='Depth of the model.', default=config["model_depth"])

    parser.add_argument('--numFilt', type=int,
                    help='Number of filters in the first layer.', default=config["num_filters"])

    parser.add_argument('--batchNormalization', type=int,
                    help='Batch normalization added (1/0).', default=0)

    parser.add_argument('--trainData', type=ascii,
            help='Path to hdf5 file with trainig data.', default=config['HDF5_train_fname'])
    
    return parser



def main(overwrite=False):
    
    hdf5_fname = config["HDF5_train_fname"];    
    print("TRAINING SET (fname): " + hdf5_fname)
    print("GROUPS: " + ' '.join(config["HDF5_group_names"]))
    
    hdf5_fid = h5py.File(hdf5_fname, "r")
#        # List all groups                
#        #group = list(f.keys())
#        #print(group)
#            
#        #dset=list(f[group[0]].keys())
#        #print (dset)
#
    dset=list(hdf5_fid[config["HDF5_group_names"][0]].keys())

#    volume=hdf5_fid[config["HDF5_group_names"][0]][dset[0]]
    
    hdf5_fid.close()
    
#        volume = f[config["HDF5_group_names"][0]][dset[0]][:] # adding [:] returns a numpy array        
#        label = f[config["HDF5_group_names"][1]][dset[0]][:] # adding [:] returns a numpy array        
#        print (volume.shape, volume.dtype)
#        #print (volume[153,114,92])
#        
#        # volume is not in the correct order DIMZ x DIMY x DIMX. Swap axes!
#        #volume = np.moveaxis(volume,0,-1)
#        #volume = np.moveaxis(volume,0,1)
#        volume = np.swapaxes(volume,0,2)                
#        label = np.swapaxes(label,0,2)               
#        print (volume.shape, volume.dtype)
#        
#        volume_cross_section = volume[:,:,92]
#        label_cross_section = label[:,:,92]
#        
#
#        fig = plt.figure()        
#        ax = fig.add_subplot(1, 2, 1)
#        imgplot = plt.imshow(volume_cross_section)
#        imgplot.set_cmap('gray')
#        ax.set_title('Cross section')
#        ax = fig.add_subplot(1, 2, 2)
#        imgplot = plt.imshow(label_cross_section)
#        ax.set_title('Label')
#        plt.show()
        
    print("DATASETS: " + ' '.join(dset))    
    #vol_fnames = ['/' + config["HDF5_group_names"][0] + '/' + s for s in dset]
    #lab_fnames = ['/' + config["HDF5_group_names"][1] + '/' + s for s in dset]
    
    if config['validation_split'] is not None:
        vol_train_fnames_prefix, vol_val_fnames_prefix, lab_train_fnames_prefix, lab_val_fnames_prefix = \
                        train_test_split(dset, dset, test_size=1-config["validation_split"], random_state=config["random_seed"])
    else:
        vol_train_fnames_prefix = dset
        vol_val_fnames_prefix = dset
        lab_train_fnames_prefix = dset
        lab_val_fnames_prefix = dset
                    
    print("TRAINING SET (#vol): {}".format(len(vol_train_fnames_prefix)))
    print("VALIDATION SET (#vol): {}".format(len(vol_val_fnames_prefix)))
  
    print(vol_train_fnames_prefix)
    print(vol_val_fnames_prefix)
    
    augment_param = {'flip':True,
                'permute':True,
                'gamma_delta': None,
                'scale_deviation': None
                }
           
    train_params = {'vol_shape': None,
          'patch_shape': config["patch_shape"],
          'vol_shape': config["vol_shape"],
          'pad_to_cube' : config['pad_to_cube'],
          'label_edges' : config['label_edges'],
          'add_label_edge' : config['add_label_edge'],
          'patch_stride': 32,
          'batch_size': config["batch_size"],
          'rescale': 1/255.,
          'shuffle': True,
          'to_fit': True,
          'augment': True,
          'augment_param': augment_param}
    
    val_params = {'vol_shape': None,
          'patch_shape': config["patch_shape"],
          'vol_shape': config["vol_shape"],
          'pad_to_cube' : config['pad_to_cube'],
          'label_edges' : config['label_edges'],
          'add_label_edge' : config['add_label_edge'],
          'patch_stride': 64,
          'batch_size': config["batch_size"],
          'rescale': 1/255.,
          'shuffle': False,
          'to_fit': True,
          'augment': False,
          'augment_param': None}
    
    # Generators
    training_generator = DataGeneratorMod(config['model_depth'], vol_train_fnames_prefix, hdf5_fname, config["HDF5_group_names"], **train_params)
    validation_generator = DataGeneratorMod(config['model_depth'], vol_val_fnames_prefix, hdf5_fname, config["HDF5_group_names"], **val_params)

    n_train_steps = training_generator.__len__()
    print ("TRAINING STEPS: {}".format(n_train_steps))
    n_validation_steps = validation_generator.__len__()
    print ("VALIDATION STEPS: {}".format(n_validation_steps))

    data_iterator = training_generator.__iter__()

    next_X, next_Y = next(data_iterator)
    print("Generated_X_Shape: {}".format(next_X.shape))
    print("Generated_Y_Shape: {}".format([y.shape for y in next_Y]))

#    volume = np.squeeze(next_X[0,:])
#    label = np.squeeze(next_Y[0,:])
    
#    planeZ = 37
#    volume_cross_section = np.squeeze(volume[:,:,planeZ])
#    label_cross_section = np.squeeze(label[:,:,planeZ])
      
#    fig = plt.figure()        
#    ax = fig.add_subplot(1, 2, 1)
#    imgplot = plt.imshow(volume_cross_section)
#    imgplot.set_cmap('gray')
#    ax.set_title('Cross section')
#    ax = fig.add_subplot(1, 2, 2)
#    imgplot = plt.imshow(label_cross_section)
#    ax.set_title('Label')
#    plt.show()    
    
    if not overwrite and os.path.exists(config["model_file"]):
        custom_objects = {'bce_dice_rho_loss': bce_dice_rho_loss, 'dice_coeff': dice_coeff,
                    'rho_coeff': rho_coeff, 'acc_coeff': acc_coeff}
        model = load_model(config["model_file"], custom_objects=custom_objects)
        model.compile(loss=model.loss, optimizer=model.optimizer, metrics=[dice_coeff, rho_coeff, acc_coeff])
    else:
        data, label = training_generator[0]
        n_labels=label[-1].shape[1]
        if config['add_label_edge']:
            n_labels*=2
        model = unet_guided_model_3d(input_shape=None,
                              n_labels=n_labels,
                              pool_size=config["pool_size"],
                              kernel_size=config["kernel_size"],
                              n_base_filters=config["num_filters"],
                              depth = config["model_depth"],
                              metrics=[dice_coeff, rho_coeff, acc_coeff],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              batch_normalization=config["batch_normalization"],
                              loss_weights=config['loss_weights'],
                              )
    
    model.summary()
    
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=training_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
            
    return
    

if __name__ == "__main__":
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    myargs = MyParser().parse_args()

    # print("Batch: {}, Epoch: {}, ValidationSplit: {}, LR: {}, Decay: {}, Patience: {}, Stop: {}, Model_depth: {} Number_filters: {}"
    #       .format(myargs.batch, myargs.epoch, myargs.validationSplit, myargs.rate, myargs.decay, myargs.patience, myargs.stop, myargs.modelDepth, myargs.numFilt))
    
    config["batch_size"] = myargs.batch
    config["n_epochs"] = myargs.epoch   
    config["validation_split"] = myargs.validationSplit
    config["initial_learning_rate"] = myargs.rate
    config["learning_rate_drop"] = myargs.decay    
    config["patience"] = myargs.patience    
    config["early_stop"] = myargs.stop    
    config["model_file"] = os.path.abspath(myargs.model.replace("'",""))
    config["model_depth"] = myargs.modelDepth
    config["num_filters"] = myargs.numFilt
    config['HDF5_train_fname'] = os.path.abspath(myargs.trainData.replace("'", ""))
    
    if myargs.labServer:
        config["HDF5_train_fname"] = config["HDF5_train_fname_LAB"]
    
    if myargs.load:
        config["overwrite"] = False
        
    if myargs.batchNormalization:
        config["batch_normalization"] = True
    
    print(config)
    
    main(overwrite=config["overwrite"])
