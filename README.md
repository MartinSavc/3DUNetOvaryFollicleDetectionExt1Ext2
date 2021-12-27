# 3D U-Net ovary and follicle detection, Ext1 and Ext2

### USOVA Dataset

The dataset is available [here](https://usova3d.um.si/wordpress/).

### Network training (extension 1 or 2)

Training command:
```
python train_USOVA3D_unet_ext.py \
	-m model_ext1.h5 \
	-r 1e-3 \
	--numFilt 8 \
	--extensionType ext1 \
	-e 200 \
	--trainData trainSet.h5
```

will train a model with learing rate 1-e3, 8 filters on the first U-net level, with extension ext1. It will train for 200 epochs with the data loaded from trainSet.h5. The model will be saved to model_ext1.h5.


To evaluate on the training data:
```
python predictALL_USOVA3D_unet_ext.py \
	-m model_ext1.h5 \
	--outOvaries predictionsOvaries.h5 \
	--outFollicles predictionsFollicles.h5 \
	--testData testSet.h5
```
