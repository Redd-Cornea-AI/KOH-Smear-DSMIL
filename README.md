# 🔬 DSMIL-KOH: AI-Powered Fungal Keratitis Detection on Whole Slide Imaging

[![Paper](https://img.shields.io/badge/Paper-Ophthalmology_Science-blue)](https://www.ophthalmologyscience.org/article/S2666-9145(24)00189-1/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 🎯 **Automated Detection of Filamentous Fungal Keratitis Using KOH Smears Whole Slide Imaging with Dual-Stream Multiple Instance Learning (DSMIL)**

## 📋 Overview

This repository provides an algorithm developed to detect filamentous fungal keratitis in potassium hydroxide (KOH) smears through multiple instance learning, enhanced by self-supervised contrastive learning. The model was trained on a dataset from Aravind Eye Hospital in Madurai, India, comprising data from 568 patients (51% testing positive for filamentous fungi).

In tests (15% of data), the algorithm achieved an AUC of 0.88 and an accuracy of 0.79. It also generates heatmaps that identify regions in the smear slide most relevant to classification, aiding interpretability. This tool has potential for rapid, automated KOH smear interpretation, supporting timely clinical decision-making.

### 📚 Publication

Our work has been published in **Ophthalmology Science**. Read the full paper [here](https://www.ophthalmologyscience.org/article/S2666-9145(24)00189-1/fulltext).

#### 📄 Paper Citation

```bibtex
@article{assaf2024automated,
  title={Automated Detection of Filamentous Fungal Keratitis Using KOH Smears with Dual-Stream Multiple Instance Learning},
  author={Assaf, Jad Farid and Srinivasan, Muthiah and Lalitha, Prajna and Rajaraman, Revathi and Mascarenhas, Jeena and Kaliamurthy, Jayaraman and Bharathi, M Jayahar and Prajna, Lalitha and Prajna, N Venkatesh and Ting, Daniel Shu Wei},
  journal={Ophthalmology Science},
  volume={4},
  number={3},
  pages={100350},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.xops.2024.100350},
  issn={2666-9145},
  url={https://www.ophthalmologyscience.org/article/S2666-9145(24)00189-1/fulltext}
}
```

**APA Format**  
Assaf, J. F., Srinivasan, M., Lalitha, P., Rajaraman, R., Mascarenhas, J., Kaliamurthy, J., Bharathi, M. J., Prajna, L., Prajna, N. V., & Ting, D. S. W. (2024). Automated Detection of Filamentous Fungal Keratitis Using KOH Smears with Dual-Stream Multiple Instance Learning. *Ophthalmology Science*, *4*(3), 100350.

#### 💻 Repository Citation

```bibtex
@software{KOH-Smear-DSMIL,
  author = {Assaf, Jad F.},
  title = {DSMIL-KOH: AI-Powered Fungal Keratitis Detection on Whole Slide Imaging},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Redd-Cornea-AI/KOH-Smear-DSMIL}
}
```

This implementation is based on PyTorch and modified from the DSMIL-WSI repository by Bin Li et al. For reference, see the original DSMIL-WSI [repository](https://github.com/binli123/dsmil-wsi) and their paper [here](https://arxiv.org/abs/2011.08939).

- Future updates to this repositories are anticipated to make the code more user friendly and easily reproducible.

## Figure 1: Whole Slide Image of a KOH Smear

![Figure 1](thumbnails/gr1_lrg.jpg)

*Whole slide image of a KOH smear demonstrating fungal filaments. The WSI file (visualized on the left with low magnification) allows visualization of patches at different magnifications, enabling detailed observation of fungal filaments (see red box).*

## Figure 2: DSMIL Framework for Analyzing Fungal Infections in WSIs

![Figure 2](thumbnails/gr2_lrg.jpg)

*Schematic of the DSMIL Framework for Analyzing Fungal Infections in WSIs. Feature vectors from the 20x and its corresponding 10x patch are fused together and used as inputs in the aggregator.*

*Schematic of the DSMIL Framework for Analyzing Fungal Infections in WSIs. Feature vectors from the 20x and its corresponding 10x patch are fused together and used as inputs in the aggregator.*

## Figure 3: Attention Heatmap of a Consensus Case

![Figure 3](thumbnails/gr4_lrg.jpg)

## Installation

1. Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)  
2. Required packages:

```bash
  conda env create --name dsmil --file env.yml
  conda activate dsmil
```

1. Install [PyTorch](https://pytorch.org/get-started/locally/)  
2. Install [OpenSlide and openslide-python](https://pypi.org/project/openslide-python/).  
[Tutorial 1](https://openslide.org/) and [Tutorial 2 (Windows)](https://www.youtube.com/watch?v=0i75hfLlPsw).  

## KOH Smear Models

Download the KOH smear fungal keratitis detection models [here](https://drive.google.com/open?id=1dxcuqHGNSOIvMs0dD2BsuuUhCZjx-sZA&usp=drive_fs). The link contains:

- Two embedders (for 10x and 20x magnifications).
- One aggregator model.

Place the embedder models in the `simclr/runs` folder and the aggregator model in the working directory.

## Training on Your Dataset

1. Place WSI files as `WSI\[DATASET_NAME]\[CATEGORY_NAME]\[SLIDE_FOLDER_NAME] (optional)\SLIDE_NAME.svs`. 

> For binary classifier, the negative class should have `[CATEGORY_NAME]` at index `0` when sorted alphabetically. For multi-class classifier, if you have a negative class (not belonging to any of the positive classes), the folder should have `[CATEGORY_NAME]` at **the last index** when sorted alphabetically. The naming of the class folders does not matter if you do not have a negative class.

2. Crop patches using:  

```bash
python deepzoom_tiler.py -m 0 -b 20 -d [DATASET_NAME]
```

>Set flag `-m [LEVEL 1] [LEVEL 2]` to crop patches from multiple magnifications. 

   ```bash
   python deepzoom_tiler.py -m 0 1 -b 20 -d [DATASET_NAME]
   ```

We used:

```bash
python deepzoom_tiler.py --magnifications 0 1  --base_mag 20 --dataset KOH_Dataset_test_lambda --background_t 7 --quality 100
python deepzoom_tiler.py --magnifications 0 1  --base_mag 20 --dataset KOH_Dataset_train_lambda --background_t 7 --quality 100
```

with:

```text
--magnifications 0 1  # Crop patches from magnification 0 and 1.
--base_mag 20         # Base magnification.
--dataset [DATASET_NAME]  # Dataset folder name.
--background_t 7       # Threshold for background detection.
--quality 100          # JPEG quality (compression or not).
```

3. Train the embedder. Edit `simclr/config.yaml` to set embedder parameters (epochs, batch size, etc) and execute:

```bash
  cd simclr
  python run.py --dataset=[DATASET_NAME]
```

>Set flag `--multiscale=1` and flag `--level=low` or `--level=high` to train an embedder for each magnification if the patches are cropped from multiple magnifications:  

   ```bash
   cd simclr
   python run.py --dataset=[DATASET_NAME] --multiscale=1 --level=low
   python run.py --dataset=[DATASET_NAME] --multiscale=1 --level=high
   ```

We used:

```bash
python run.py --dataset=KOH_Dataset_train_lambda --multiscale=1 --level=low
python run.py --dataset=KOH_Dataset_train_lambda --multiscale=1 --level=high
```

4. Compute features using the embedder.  

```bash
cd ..
python compute_feats.py --dataset=[DATASET_NAME]
```

>Set flag `--magnification=tree` to compute the features for multiple magnifications.
>This will use the last trained embedder to compute the features, if you want to use an embedder from a specific run, add the option `--weights=[RUN_NAME]`, where `[RUN_NAME]` is a folder name inside `simclr/runs/`. If you have an embedder you want to use, you can place the weight file as `simclr/runs/[RUN_NAME]/checkpoints/model.pth` and pass the `[RUN_NAME]` to this option. To use a specific embedder for each magnification, set option `--weights_low=[RUN_NAME]` (embedder for low magnification) and `--weights_high=[RUN_NAME]` (embedder for high magnification). The embedder architecture is ResNet18 with **instance normalization**.

```bash
cd ..
python compute_feats.py --dataset=[DATASET_NAME] --magnification tree --weights_low=low_mag_embedder --weights_high=high_mag_embedder
```

We used:

```bash
python compute_feats.py --dataset=KOH_Dataset_train_lambda --num_classes 1 --magnification tree --weights_low=low_mag_embedder --weights_high=high_mag_embedder
python compute_feats.py --dataset=KOH_Dataset_test_lambda --num_classes 1 --magnification tree --weights_low=low_mag_embedder --weights_high=high_mag_embedder
```

5. Training:

```bash
python train_tcga.py --dataset=[DATASET_NAME]
```

>You will need to adjust `--num_classes` option if the dataset contains more than 2 positive classes or only 1 positive class and 1 negative class (binary classifier). See the next section for details.  

### Useful arguments:

```text
[--num_classes]       # Number of non-negative classes, for a binary classification (postive/negative), this is set to 1
[--feats_size]        # Size of feature vector (depends on the CNN backbone and whether patch fusion was used)
[--lr]                # Initial learning rate [0.0001]
[--num_epochs]        # Number of training epochs [50]
[--stop_epochs]       # Skip remaining epochs if training has not improved after N epochs [10]
[--weight_decay]      # Weight decay [1e-3]
[--dataset]           # Dataset folder name
[--split]             # Training/validation split [0.2]
[--dropout_patch]     # Randomly dropout a portion of patches and replace with duplicates during training [0]
[--dropout_node]      # Randomly dropout a portion of nodes in the value vector generation network during training [0]
```

### Understanding different evaluation schemes and metrics

>Different training and evaluation schemes can be choosen by setting the arugment (--eval_scheme).

#### --eval_scheme=5-fold-cv

>A 5-fold cross-validation. For each fold, AUC and accuracy score will be computed on the validation set. Averaged values across the 5 folds will be computed after all folds complete.

#### --eval_scheme=5-fold-cv-standalone-test

>A standalone test set consisting of 20% samples is reserved, remaining 80% samples are used to construct a 5-fold cross-validation.  
>For each fold, the best model and corresponding threshold are saved.    
>After the 5-fold cross-validation, 5 best models along with the corresponding optimal thresholds are obtained which are used to perform inference on the reserved test set. A final prediction for a test sample is the majority vote of the 5 models.  
>For a binary classification, accuracy and balanced accuracy score are computed. For a multi-label classification, hamming loss (smaller the better) and subset accuracy are computed.  

In our case, we created a customized train_tcga_v2.py for 5 fold cross validation training with a completely separate test set. This script also includes WSI feature merging if they belong to the same patient (starts with the same name). Customize to your liking.

```bash
python train_tcga_v2.py --dataset=KOH_Dataset_train_lambda --dataset_test=KOH_Dataset_test_lambda --num_classes=1 --feats_size=1024 --num_epochs 200 --stop_epochs 25
```

6. Testing.

```bash
python attention_map.py --bag_path test/patches --map_path test/output --thres 0.73 0.28
```

What we used:

```bash
$ python attention_map.py --bag_path test_bags/Fungus --map_path test/output --thres 0.5693772435188293 --aggregator_weights aggregator.pth --embedder_weights embedder_low.pth
```

Useful arguments:

```text
[--num_classes]         # Number of non-negative classes.
[--feats_size]          # Size of feature vector (depends on the CNN backbone).
[--thres]               # List of thresholds for the classes returned by the training function.
[--embedder_weights]    # Path to the embedder weights file (saved by SimCLR). Use 'ImageNet' if ImageNet pretrained embedder is used.
[--aggregator_weights]  # Path to the aggregator weights file.
[--bag_path]            # Path to a folder containing folders of patches.
[--patch_ext]            # File extensino of patches.
[--map_path]            # Path of output attention maps.
```

- To obtain useful heatmap overlays, run the script WSI_heatmap.py after generating the heatmaps.  Make sure to edit the directories to reflect your folder structure. This code will loop over WSI images, generate high resolution images, and overlay the heatmaps generated previously on the images.

```bash
python WSI_heatmap.py
```

## Folder structure

Data is organized in two folders, `WSI` and `datasets`. `WSI` folder contains the images and `datasets` contains the computed features.

```text
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- CLASS_1
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_2
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
```

Once patch extraction is performed, `sinlge` folder or `pyramid` folder will appear.

```text
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- single
|   |   |   |-- CLASS_1
|   |   |   |   |-- SLIDE_1
|   |   |   |   |   |-- PATCH_1.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...
|   |   |-- pyramid
|   |   |   |-- CLASS_1
|   |   |   |   |-- SLIDE_1
|   |   |   |   |   |-- PATCH_LOW_1
|   |   |   |   |   |   |-- PATCH_HIGH_1.jpeg
|   |   |   |   |   |   |-- ...
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- PATCH_LOW_1.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...
```

Once feature computing is performed, `DATASET_NAME` folder will appear inside `datasets` folder.

```text
root
|-- datasets
|   |-- DATASET_NAME
|   |   |-- CLASS_1
|   |   |   |-- SLIDE_1.csv
|   |   |   |-- ...
|   |   |-- CLASS_2
|   |   |   |-- SLIDE_1.csv
|   |   |   |-- ...
|   |   |-- CLASS_1.csv
|   |   |-- CLASS_2.csv
|   |   |-- DATASET_NAME.csv
```
  
## Feature vector csv files explanation

1. For each bag, there is a .csv file where each row contains the feature of an instance. The .csv is named as "_bagID_.csv" and put into a folder named "_dataset-name_/_category_/".  

<div align="center">
  <img src="thumbnails/bag.png" width="700px" />
</div>  

2. There is a "_dataset-name_.csv" file with two columns where the first column contains the paths to all _bagID_.csv files, and the second column contains the bag labels.  

<div align="center">
  <img src="thumbnails/bags.png" width="700px" />
</div>  

3. Labels.

> For binary classifier, use `1` for positive bags and `0` for negative bags. Use `--num_classes=1` at training.  
> For multi-class classifier (`N` positive classes and one optional negative class), use `0~(N-1)` for positive classes. If you have a negative class (not belonging to any one of the positive classes), use `N` for its label. Use `--num_classes=N` (`N` equals the number of **positive classes**) at training.