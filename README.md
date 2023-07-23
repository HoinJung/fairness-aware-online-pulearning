# Fairness-Aware Online Positive-Unlabeled Learning

## Data Preparation
Raw data files for Adult, COMPAS, German, Drug, and Bank datasets are prepared. To download the MEPS dataset, please follow the description in the AIF360 library. [Link to AIF360](https://github.com/Trusted-AI/AIF360/tree/502ff47a519582a653417e0957668eb5264bedcd/aif360/data/raw/meps)

Run ```sh data_prep.sh``` to create ```*.npz``` files for training. Each dataset will have 20 different sets of train/test split to fairly evaluate the performance of our approach.

## Training

Run ```python main.py --model MODEL --dataset DATASET --r1 R1 --batch_size BATCH_SIZE --epoch EPOCH --pu_type PU_TYPE --lr LR --round ROUND --fairness FAIRNESS_CONSTRAINT --online --fair --lam_f LAMBDA_F```

#### Options
- MODEL: ```mlp``` and ```linear```.
- DATASET: ```adult```, ```compas```, ```german```, ```drug```, ```bank```, and ```meps```.
- R1: the ratio of unlabeled samples in positive instances.
- BATCH_SIZE: the batch size for offline learning. The batch size will be set as a full batch in online learning.
- EPOCH: the number of epochs for offline learning. The number of epochs will be set as 1 in online learning.
- PU_TYPE: ```upu''' and ```nnpu```.
- LR: ```lr=0.01``` is recommended for offline learning. ```lr=0.1, 0.5, 1.0``` is recommended for online learning.
- ROUND: The number of total rounds for online learning.
- FAIRNESS_CONSTRAINT: fairness constraint to compute a fairness loss. ```ddp``` and ```deo``.
- LAMBDA_F: a hyperparameter to weigh the fairness loss.
- ONLINE: command ```--online``` to run online learning. To execute offline learning, drop this command.
- FAIR: command ```--fair``` to use fairness loss. To execute the baseline, drop this command.

#### example
```python main.py --model linear --dataset german --r1 0.1 --pu_type nnpu --lr 0.1 --round 20 --fairness ddp --online --fair --lam_f 0.1```
