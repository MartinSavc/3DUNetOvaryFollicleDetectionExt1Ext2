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
#config["model_file"] = os.path.abspath("model_USOVA3D.h5")
config["patch_shape"] = None  # switch to None to train on the whole image
config["vol_shape"] = (128, 128, 128)  # switch to None to train on the whole image
config["model_file"] = "model_USOVA3D_unet_guided_v001.h5"
config["result_file"] = "result_USOVA3D_unet_guided_v001.h5"

config["HDF5_train_fname"] = "../Data/Ovary/trainSet.h5"#"D:\\WORK\\USOVA3D\\Tensorflow_solutions\\Data\\trainSet.h5"
#config["HDF5_train_fname_LAB"] = "/home/bozidarp/3DCNN/Data/trainSet.h5" 

config["HDF5_test_fname"] = "../Data/Ovary/testSet.h5"#"D:\\WORK\\USOVA3D\\Tensorflow_solutions\\Data\\testSet.h5"
#config["HDF5_test_fname_LAB"] = "/home/bozidarp/3DCNN/Data/testSet.h5" 


config["HDF5_group_names"] = ["Images", "Labels"]
config["random_seed"] = 42

config["display_results"] = False

def MyParser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='USOVA3D_3D Inference method')

    parser.add_argument('-m', '--model', type=ascii,
                    help='Name of the stored model.', default=config["model_file"])


    parser.add_argument('-o', '--out', type=ascii,
                    help='Name of the HDF5 file to store results.', default=config["result_file"])
    

    parser.add_argument('--display', type=ascii,
                    help='Display sample results.', default=config["display_results"])
    
    parser.add_argument('--labServer', type=int,
                    help='Running on LAB server (1/0).', default=0)

    parser.add_argument('--testData', type=ascii,
            help='Path to hdf5 file with testing data.', default=config['HDF5_test_fname'])
        
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

    orig_shape = volume.shape
    zoom_fc = [vol_shape[0]/volume.shape[0], vol_shape[1]/volume.shape[1], vol_shape[2]/volume.shape[2]]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            
        volume = zoom(volume, zoom=zoom_fc)
        
    return volume, orig_shape


def predict_volume_wise (model, volume, vol_shape=(96,96,96)):

    vol, orig_shape = resize_volume(volume, vol_shape=vol_shape)
    
    vol = np.expand_dims(vol,axis=0)
    vol = np.expand_dims(vol,axis=0)
               
    pred_vol = model.predict(vol)[-1]
    if pred_vol.shape[1] > 1:
        pred_vol = pred_vol[:, :1]
    pred_vol = np.squeeze(pred_vol)
        
    segmented, _ = resize_volume(pred_vol, orig_shape)
    segmented = (segmented>0.5).astype('uint8')

    return segmented, orig_shape


def main(SAVED_MODEL_PATH, HDF5_OUT_FNAME):
    
    hdf5_fname = config["HDF5_test_fname"];    
    print("TEST SET (fname): " + hdf5_fname)
    print("GROUPS: " + ' '.join(config["HDF5_group_names"]))
    
    hdf5_fid = h5py.File(hdf5_fname, "r")
    vol_test_fnames_prefix=list(hdf5_fid[config["HDF5_group_names"][0]].keys())
    hdf5_fid.close()
    
    print("DATASETS: " + ' '.join(vol_test_fnames_prefix))
    
#    save_model_path = config["model_file"]

    custom_objects = {'bce_dice_rho_loss': bce_dice_rho_loss, 'dice_coeff': dice_coeff,
                    'rho_coeff': rho_coeff, 'acc_coeff': acc_coeff}

    model = load_model(SAVED_MODEL_PATH, custom_objects=custom_objects)
    
#    hdf5_fname_out = "rezultat.h5"
    hdf5_fid_out = h5py.File(HDF5_OUT_FNAME, "w")
    
    group_name = config["HDF5_group_names"][1]
    group=hdf5_fid_out.create_group(group_name)
    
    
    print("\n>>>>> PREDICTING <<<<< \n")

    vol_shape = config['vol_shape']
    
    for vol_idx in vol_test_fnames_prefix:
        
        print("VOL_IDX.....{}".format(vol_idx))
        
        vol_orig, shape_orig = load_grayscale_volume(hdf5_fname,vol_idx=vol_idx)
        
        result, _ = predict_volume_wise(model, vol_orig, vol_shape)
        result_orig = result[:shape_orig[0], :shape_orig[1], :shape_orig[2]]
        
        if config["display_results"]:
        
            planeZ = 37
            volume_cross_section = np.squeeze(vol_orig[:,:,planeZ])
            label_cross_section = np.squeeze(result_orig[:,:,planeZ])
                
            fig = plt.figure()        
            ax = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(volume_cross_section)
            imgplot.set_cmap('gray')
            ax.set_title('Cross section')
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(label_cross_section)
            ax.set_title('Label')

            plt.show()
                        
        result_orig = np.swapaxes(result_orig,0,2)        
        print("TYPE: {} + {}".format(result_orig.dtype, np.unique(result_orig)))

        dset = group.create_dataset(vol_idx, result_orig.shape, dtype=result_orig.dtype, compression='gzip', compression_opts=9)
        dset[:]=result_orig
        

    print("\n>>>>> END PREDICTING & STORING DATA <<<<< \n")
    hdf5_fid_out.close()
                

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
    
    hdf5_out_fname = os.path.abspath(myargs.out.replace("'",""))    
    print("HDF5_FNAME: {}".format(hdf5_out_fname))

    config['HDF5_test_fname'] = os.path.abspath(myargs.testData.replace("'", ""))

    if myargs.labServer:
        config["HDF5_test_fname"] = config["HDF5_test_fname_LAB"]
        
    if type(myargs.display) == str:
        
        myargs.display = myargs.display.replace("'","")
        
        if myargs.display.lower() == 'true':
            config["display_results"]=True
        else:
            config["display_results"]=False
            
    print("DISPLAY_RESULTS: {}".format(config["display_results"]))
    
    main(model_name, hdf5_out_fname)
