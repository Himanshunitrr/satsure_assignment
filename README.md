# Satsure Assignment

## Literature Review (LR): 
Semantic segmentation: Recognize Stuff 
Instance segmentation: Recognize Objects
Panoptic segmentation: Recognize both stuff and objects (https://arxiv.org/pdf/1801.00868.pdf)

The task can be standardized as PASTIS(Panoptic Segmentation of satellite image TImes Series). As this was new to me (I have worked on Semantic, Instance and Panoptic Segmentation but not on PASTIS so I did a LR. Most of the work has focused on including the temporal component and other sensory information such as:)
* https://arxiv.org/pdf/2112.07558v1.pdf
* https://arxiv.org/pdf/2107.07933v4.pdf
* https://arxiv.org/pdf/2301.04944v3.pdf
* https://arxiv.org/pdf/2204.00951v2.pdf
* https://arxiv.org/pdf/2303.12533v1.pdf
* https://arxiv.org/pdf/2305.02086v2.pdf
* https://arxiv.org/pdf/2204.01952v1.pdf

# Visualizations
Its in Visulisations.ipynb

# Deck:
https://docs.google.com/presentation/d/1OUDEnCBy7s1uv_iFWJf4ctz61Zgc0XqxAFQuWQIFEDY/edit?usp=sharing

# Working around code
## Requirements

### PASTIS Dataset download
The Dataset is freely available for download [here](https://github.com/VSainteuf/pastis-benchmark). 



### Python requirements
To install requirements:

```setup
pip install -r requirements.txt
```

(`torch_scatter` is required for the panoptic experiments. 
Installing this library requires a little more effort, see [the official repo](https://github.com/rusty1s/pytorch_scatter))



## Inference with pre-trained models

### Panoptic segmentation


Pre-trained weights of U-TAE+Paps are available [here](https://zenodo.org/record/5172301)

To perform inference of the pre-trained model on the test set of PASTIS run:

```test
python test_panoptic.py --dataset_folder PATH_TO_DATASET --weight_folder PATH_TO_WEIGHT_FOLDER --res_dir OUPUT_DIR
```


### Semantic segmentation


Pre-trained weights of U-TAE are available [here](https://zenodo.org/record/5172293)

To perform inference of the pre-trained model on the test set of PASTIS run:

```test
python test_semantic.py --dataset_folder PATH_TO_DATASET --weight_folder PATH_TO_WEIGHT_FOLDER --res_dir OUPUT_DIR
```


## Training models from scratch

### Panoptic segmentation

To reproduce the main result for panoptic segmentation (with U-TAE+PaPs) run the following :

```train
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR
```
Other modes of training

```train
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_NoCNN --no_mask_conv
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UConvLSTM --backbone uconvlstm
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_shape24 --shape_size 24
```

Note: By default this script runs the 5 folds of the cross validation, which can be quite long (~12 hours per fold on a Tesla V100). 
Use the fold argument to execute one of the 5 folds only 
(e.g. for the 3rd fold : `python train_panoptic.py --fold 3 --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR`).

### Semantic segmentation

To reproduce results for semantic segmentation (with U-TAE) run the following :

```train
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR
```

Other modes of trainig:

```train
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UNET3d --model unet3d
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UConvLSTM --model uconvlstm
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_FPN --model fpn
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_BUConvLSTM --model buconvlstm
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_COnvGRU --model convgru
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_ConvLSTM --model convlstm

