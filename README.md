# Fairness-Aware Online Positive-Unlabeled Learning

## Data Preparation
Install necessary packages.
```
pip install datasets
pip install gensim
python -m spacy download en
```

Run preprocessing code,
```
python chat_preprocessing.py
python wiki_preprocessing.py
```
For a pre-trained encoder, e.g. BERT, run BERT-version preprocessing,
```
python bert_chat_preprocessing.py
python bert_wiki_preprocessing.py
```


## Training

Run 
```
python main.py --model MODEL --dataset DATASET --r1 R1 --batch_size BATCH_SIZE --epoch EPOCH --pu_type PU_TYPE --lr LR --round ROUND --fairness FAIRNESS_CONSTRAINT --online --fair --lam_f LAMBDA_F --penalty --lam_penalty --LAM_PENALTY
```

### Options
- MODEL: ```mlp``` and ```linear```.
- DATASET: ```wiki```, ```chat_toxicity```
- R1: the ratio of unlabeled samples in positive instances.
- BATCH_SIZE: the batch size for offline learning. The batch size will be set as a full batch in online learning.
- EPOCH: the number of epochs for offline learning. The number of epochs will be set as 1 in online learning.
- PU_TYPE: ```upu``` and ```nnpu```.
- LR: ```lr=0.01, 0.1``` is recommended for offline learning. ```lr=0.01, 0.1, 1.0``` is recommended for online learning.
- ROUND: The number of total rounds for online learning.
- ONLINE: command ```--online``` to run online learning. To execute offline learning, drop this command.
- FAIR: command ```--fair``` to use fairness loss. To execute the baseline, drop this command.
- FAIRNESS_CONSTRAINT: fairness constraint to compute a fairness loss. ```ddp``` and ```deo```.
- LAMBDA_F: a hyperparameter to weigh the fairness loss.
- PENALTY: command ```penalty``` to use penalty term.
- LAM_PENALTY: a hyperparameter to weigh the penalty term.

### Example
```
python main.py --dataset chat_toxicity --online --model linear --pu_type nnpu --gpu_id 1 --lr 1 --r1 0.5 --fair --fairness deo --lam_f 0.0001 --penalty --lam_penalty 1
```
