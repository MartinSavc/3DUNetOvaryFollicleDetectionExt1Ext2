import os
import h5py
import argparse

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from snet.model import snet_model, composite_bce_loss
from snet.training import train_model

from classes_sliced_3D import SliceDataGenerator

import tensorflow as tf
from tensorflow.keras.models import load_model

config = {
        'model_file' : 'test_model_USOVA3D_snet.h5',
        'overwrite' : False,
        'HDF5_train_fname' : '../Data/OvaryFollicle/trainSet.h5',
        'batch_size' : 5,
        'validation_split' : 0.8,
        'HDF5_group_names' : ["Images", "Labels"],
        'random_seed' : 42,
        }

def MyParser():
    parser = argparse.ArgumentParser(description='USOVA3D_3D Training Method')

    parser.add_argument('-v', '--validationSplit', type=int,
                    help='Validation split rate', default=config["validation_split"])    
    parser.add_argument('-l', '--load', type=int,
                    help='Load weights/model (int)', default=0)    
    parser.add_argument('-m', '--model', type=ascii,
                    help='Name of model to be stored and read from.', default=config["model_file"])
    parser.add_argument('-b', '--batch', type=int,
                    help='Training batch size (int).', default=config["batch_size"])
    parser.add_argument('--trainData', type=ascii,
            help='Path to hdf5 file with trainig data.', default=config['HDF5_train_fname'])

    return parser

def main(overwrite=False):
    
    hdf5_fname = config["HDF5_train_fname"];    
    print("TRAINING SET (fname): " + hdf5_fname)
    print("GROUPS: " + ' '.join(config["HDF5_group_names"]))
    
    hdf5_fid = h5py.File(hdf5_fname, "r")
    dset=list(hdf5_fid[config["HDF5_group_names"][0]].keys())
    hdf5_fid.close()
        
    print("DATASETS: " + ' '.join(dset))    
    
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
    
    train_params = {
            'to_fit' : True,
            'n_slices' : 3,
            'output_rep' : 4,
            'batch_size' : config['batch_size'],
            'shuffle' : True,
            'augment' : True,
            }
    
    val_params = {
            'to_fit' : True,
            'n_slices' : 3,
            'output_rep' : 4,
            'batch_size' : config['batch_size'],
            'shuffle' : False,
            }
    
    # Generators
    training_generator = SliceDataGenerator(vol_train_fnames_prefix, hdf5_fname, config["HDF5_group_names"], **train_params)
    validation_generator = SliceDataGenerator(vol_val_fnames_prefix, hdf5_fname, config["HDF5_group_names"], **val_params)

    n_train_steps = training_generator.__len__()
    print ("TRAINING STEPS: {}".format(n_train_steps))
    n_validation_steps = validation_generator.__len__()
    print ("VALIDATION STEPS: {}".format(n_validation_steps))

    data_iterator = training_generator.__iter__()

    next_X, next_Y = next(data_iterator)
    print("Generated_X_Shape: {}".format(next_X.shape))
    print("Generated_Y_Shape: {}".format([y.shape for y in next_Y]))

    if not overwrite and os.path.exists(config["model_file"]):
        custom_objects = {'composite_bce_loss' : composite_bce_loss}
        model = load_model(config["model_file"], custom_objects=custom_objects)
        model.compile(loss=model.loss, optimizer=model.optimizer)
    else:
        model = snet_model()
    
    model.summary()
    
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=training_generator,
                validation_generator=validation_generator,
                n_epochs=200)
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

    config["batch_size"] = myargs.batch
    config["validation_split"] = myargs.validationSplit
    config["model_file"] = os.path.abspath(myargs.model.replace("'",""))
    config['HDF5_train_fname'] = os.path.abspath(myargs.trainData.replace("'", ""))
    
    if myargs.load:
        config["overwrite"] = False
        
    print(config)
    
    main(overwrite=config["overwrite"])
