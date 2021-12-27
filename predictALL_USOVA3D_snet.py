import os
import h5py
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from snet.model import composite_bce_loss

import tensorflow as tf


config = dict()

config["HDF5_group_names"] = ["Images", "Labels"]

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


def main(SAVED_MODEL_PATH, HDF5_OVARIES_OUT_FNAME, HDF5_FOLLICLE_OUT_FNAME):

    hdf5_fname = config["HDF5_test_fname"]
    print("TEST SET (fname): " + hdf5_fname)
    print("GROUPS: " + ' '.join(config["HDF5_group_names"]))

    hdf5_fid = h5py.File(hdf5_fname, "r")
    vol_test_fnames_prefix=list(hdf5_fid[config["HDF5_group_names"][0]].keys())

    print("DATASETS: " + ' '.join(vol_test_fnames_prefix))

    custom_objects = {'composite_bce_loss' : composite_bce_loss}

    model = load_model(SAVED_MODEL_PATH, custom_objects=custom_objects)
    n_slices = model.input_shape[3]

    hdf5_ov_fid_out = h5py.File(HDF5_OVARIES_OUT_FNAME, "w")
    hdf5_fl_fid_out = h5py.File(HDF5_FOLLICLE_OUT_FNAME, "w")

    group_name = config["HDF5_group_names"][1]
    group_ov=hdf5_ov_fid_out.create_group(group_name)
    group_fl=hdf5_fl_fid_out.create_group(group_name)


    print("\n>>>>> PREDICTING <<<<< \n")

    for vol_idx in vol_test_fnames_prefix:

        print("VOL_IDX.....{}".format(vol_idx))

        volume = hdf5_fid['Images'][vol_idx]

        H, W, D = volume.shape
        result = np.zeros(volume.shape+(2,))

        H_adj = int(np.ceil(H/2**(4-1)))*2**(4-1)
        W_adj = int(np.ceil(W/2**(4-1)))*2**(4-1)
        D_adj = int(np.ceil(D/2**(4-1)))*2**(4-1)

        result = np.zeros(volume.shape+(2,))

        # predict row slices
        vol_h_slices = np.zeros((H-n_slices, W_adj, D_adj, 3))
        for n in range(H-n_slices):
            vol_slice = volume[n:n+n_slices, :, :]
            vol_slice = vol_slice.transpose(1, 2, 0)

            vol_h_slices[n] = np.pad(vol_slice, ((0, W_adj-W), (0, D_adj-D), (0, 0)))
        res_h_slices = model.predict(vol_h_slices)[-1]

        for n in range(n_slices):
            result[n:-n_slices+n, :, :, :] += res_h_slices[:, :W, :D, 2*n:2*(n+1)]

        # predict column slices
        vol_w_slices = np.zeros((W-n_slices, H_adj, D_adj, 3))
        for n in range(W-n_slices):
            vol_slice = volume[:, n:n+n_slices, :]
            vol_slice = vol_slice.transpose(0, 2, 1)

            vol_w_slices[n] = np.pad(vol_slice, ((0, H_adj-H), (0, D_adj-D), (0, 0)))
        res_w_slices = model.predict(vol_w_slices)[-1]
        res_w_slices = res_w_slices.transpose(1, 0, 2, 3)

        for n in range(n_slices):
            result[:, n:-n_slices+n, :, :] += res_w_slices[:H, :, :D, 2*n:2*(n+1)]

        # predict depth slices
        vol_d_slices = np.zeros((D-n_slices, H_adj, W_adj, 3))
        for n in range(W-n_slices):
            vol_slice = volume[:, :, n:n+n_slices]

            vol_d_slices[n] = np.pad(vol_slice, ((0, H_adj-H), (0, W_adj-W), (0, 0)))
        res_d_slices = model.predict(vol_d_slices)[-1]
        res_d_slices = res_d_slices.transpose(1, 2, 0, 3)

        for n in range(n_slices):
            result[:, :, n:-n_slices+n, :] += res_d_slices[:H, :W, :, 2*n:2*(n+1)]

        result /= 3*n_slices
        result_seg = (result>0.5).astype('uint8')

        print("TYPE: {} + {}".format(result_seg.dtype, np.unique(result_seg)))

        dset_ov = group_ov.create_dataset(vol_idx, result_seg.shape[:-1], dtype=result_seg.dtype, compression='gzip', compression_opts=9)
        dset_fl = group_fl.create_dataset(vol_idx, result_seg.shape[:-1], dtype=result_seg.dtype, compression='gzip', compression_opts=9)
        dset_ov[:]=result_seg[:, :, :, 0]
        dset_fl[:]=result_seg[:, :, :, 1]

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
