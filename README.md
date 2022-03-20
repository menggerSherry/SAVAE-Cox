# SAVAE-Cox

SAVAE-Cox: A novel attention-mechanism based Cox survival model by exploiting pan-cancer empirical genomic information

## SAVAE-Cox Architecture

![avatar](figures/model.png)

## Install

### Requirements

* Python 3
* Pytorch 1.8.1 or higher
* Install numpy, tqdm, scikit-learn, imblearn, pandas and lifelines
```
pip install numpy
pip install tqdm
pip install scikit-learn
pip install pandas
conda install -c conda-forge imbalanced-learn
conda install -c conda-forge lifelines
```
## Download dataset
Here, we provide a cleaned pan-cancer dataset and a dataset of 16 cancer types:

[Google Drive](https://drive.google.com/drive/folders/1KuDVRkPJZWYfQ2Z4YRxDo6e8lo5s68za?usp=sharing)
## Run Weight Reduction

```
python train_encoder.py
```

## Run Survival Analysis
```
python train_cox.py
```
