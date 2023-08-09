# Barlow Adaptor

This repository contains the code for **Cross-Dataset Adaptation for Instrument
Classification in Cataract Surgery Videos**, accepted at MICCAI 2023

## Environment File
Create a new conda environment with the config file given in the repository as follows:
```
conda env create --file=VIU.yaml
conda activate VIU
```

## Data Directory Structure
```
root
|--train
    |--label1
        |--img
        |--img
        ...
    |--label2
    ...
|--val
    |--label1
        |--img
        |--img
        ...
    |--label2
    ...
|--test
    |--label1
        |--img
        |--img
        ...
    |--label2
    ...
```

## General file descriptions
- transform_utils.py - data transforms defined here
- data_utils.py - functions to generate dataloaders for different datasets
- model.py - model architectures defined here
- train.py - driver code for training and testing model
- test.py - driver code for testing models

## Example Usage for Training
```
python train.py <dataset1 name> <dataset2 name> <save_path> "cuda:0" 16 0.0001 1e-3 0.1 loss7 True 1
```

## Citation
```
@misc{paranjape2023crossdataset,
      title={Cross-Dataset Adaptation for Instrument Classification in Cataract Surgery Videos}, 
      author={Jay N. Paranjape and Shameema Sikder and Vishal M. Patel and S. Swaroop Vedula},
      year={2023},
      eprint={2308.04035},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
