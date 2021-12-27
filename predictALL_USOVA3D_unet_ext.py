import os
import h5py
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from unet3d.metrics import dice_coeff, rho_coeff, acc_coeff, bce_dice_rho_loss, dice_loss, rho_loss, acc_loss
from classes_3D import _create_range

import tensorflow as tf
from scipy.ndimage import zoom


config = dict()
config["patch_shape"] = None  # switch to None to train on the whole image
config["vol_shape"] = (128, 128, 128)  # switch to None to train on the whole image

config["HDF5_group_names"] = ["Images", "Labels"]
config["random_seed"] = 42


def MyParser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='USOVA3D_3D Inference method')

    parser.add_argument('-m', '--model', type=ascii, required=True,
                    help='Name of the stored model.')

    parser.add_argument('--outOvaries', type=ascii, required=True,
                    help='Name of the HDF5 file to store ovarie results.')
    parser.add_argument('--outFollicles', type=ascii, required=True,
                    help='Name of the HDF5 file to store follicle results.')

    parser.add_argument('--testData', type=ascii, required=True,
            help='Path to hdf5 file with testing data.')

    return parser


def pad_volume_to_cube( volume):
    D, H, W = volume.shape
    edge = max(D, H, W)
    new_volume = np.zeros((edge, edge, edge), dtype=volume.dtype)
    new_volume[:D, :H, :W] = volume
    return new_volume

def load_grayscale_volume(hdf5_fname, vol_idx, group_name="Images", rescale=1/255.):

    dset_name = vol_idx

    hdf5_fid = h5py.File(hdf5_fname, "r")

    volume = hdf5_fid[group_name][dset_name]
    volume = np.swapaxes(volume,0,2)
    orig_shape = volume.shape
    volume = pad_volume_to_cube(volume)

    hdf5_fid.close()

    return rescale*volume, orig_shape


def resize_volume(volume, vol_shape=(108, 108, 108)):

    C=volume.shape[0]
    orig_shape = volume.shape[1:]
    zoom_fc = [vol_shape[0]/orig_shape[0], vol_shape[1]/orig_shape[1], vol_shape[2]/orig_shape[2]]

    volume_new = np.zeros((C,)+vol_shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in range(C):
            volume_new[c, :, :, :] = zoom(volume[c, :, :, :], zoom=zoom_fc)

    return volume_new, orig_shape


def predict_volume_wise (model, volume, vol_shape=(96,96,96)):

    vol = np.expand_dims(volume,axis=0)
    vol, orig_shape = resize_volume(vol, vol_shape=vol_shape)
    vol = np.expand_dims(vol,axis=0)
               
    if len(model.outputs)==1:
        pred_vol = model.predict(vol)
    else:
        pred_vol = model.predict(vol)[-1]
    pred_vol = np.squeeze(pred_vol)
        
    segmented, _ = resize_volume(pred_vol, orig_shape)
    segmented = (segmented>0.5).astype('uint8')

    return segmented, orig_shape


def main(SAVED_MODEL_PATH, HDF5_OVARIES_OUT_FNAME, HDF5_FOLLICLE_OUT_FNAME):

    hdf5_fname = config["HDF5_test_fname"]
    print("TEST SET (fname): " + hdf5_fname)
    print("GROUPS: " + ' '.join(config["HDF5_group_names"]))

    hdf5_fid = h5py.File(hdf5_fname, "r")
    vol_test_fnames_prefix=list(hdf5_fid[config["HDF5_group_names"][0]].keys())
    hdf5_fid.close()

    print("DATASETS: " + ' '.join(vol_test_fnames_prefix))

    custom_objects = {'bce_dice_rho_loss': bce_dice_rho_loss, 'dice_coeff': dice_coeff,
                    'rho_coeff': rho_coeff, 'acc_coeff': acc_coeff}

    model = load_model(SAVED_MODEL_PATH, custom_objects=custom_objects)

    hdf5_ov_fid_out = h5py.File(HDF5_OVARIES_OUT_FNAME, "w")
    hdf5_fl_fid_out = h5py.File(HDF5_FOLLICLE_OUT_FNAME, "w")

    group_name = config["HDF5_group_names"][1]
    group_ov=hdf5_ov_fid_out.create_group(group_name)
    group_fl=hdf5_fl_fid_out.create_group(group_name)


    print("\n>>>>> PREDICTING <<<<< \n")

    vol_shape = config['vol_shape']

    for vol_idx in vol_test_fnames_prefix:

        print("VOL_IDX.....{}".format(vol_idx))

        vol_orig, shape_orig = load_grayscale_volume(hdf5_fname,vol_idx=vol_idx)

        result, _ = predict_volume_wise(model, vol_orig, vol_shape)
        result_orig = result[:, :shape_orig[0], :shape_orig[1], :shape_orig[2]]

        result_orig = np.swapaxes(result_orig,1,3)
        print("TYPE: {} + {}".format(result_orig.dtype, np.unique(result_orig)))

        dset_ov = group_ov.create_dataset(vol_idx, result_orig.shape[1:], dtype=result_orig.dtype, compression='gzip', compression_opts=9)
        dset_fl = group_fl.create_dataset(vol_idx, result_orig.shape[1:], dtype=result_orig.dtype, compression='gzip', compression_opts=9)
        dset_ov[:]=result_orig[0]
        dset_fl[:]=result_orig[1]

    print("\n>>>>> END PREDICTING & STORING DATA <<<<< \n")
    hdf5_ov_fid_out.close()
    hdf5_fl_fid_out.close()


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

    model_name = os.path.abspath(myargs.model.replace("'",""))
    print("MODEL_NAME: {}".format(model_name))

    hdf5_ovaries_out_fname = os.path.abspath(myargs.outOvaries.replace("'",""))
    print("HDF5_OVARIES_FNAME: {}".format(hdf5_ovaries_out_fname))

    hdf5_follicles_out_fname = os.path.abspath(myargs.outFollicles.replace("'",""))
    print("HDF5_FOLLICLES_FNAME: {}".format(hdf5_follicles_out_fname))

    config['HDF5_test_fname'] = os.path.abspath(myargs.testData.replace("'", ""))

    main(model_name, hdf5_ovaries_out_fname, hdf5_follicles_out_fname)
