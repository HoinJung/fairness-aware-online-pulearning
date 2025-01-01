from datasets import load_dataset
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy
import torch

# Load the chat-toxicity dataset
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
data_train = pd.DataFrame(dataset['train'])
data_test = pd.DataFrame(dataset['test'])

# Split the training set into train and validation sets if validation set is not predefined
if 'validation' in dataset:
    data_val = pd.DataFrame(dataset['validation'])
else:
    data_train, data_val = np.split(data_train.sample(frac=1, random_state=42), [int(0.8 * len(data_train))])

# Extract text and labels
X_train = data_train['user_input']
Y_train = data_train['toxicity']
X_val = data_val['user_input']
Y_val = data_val['toxicity']
X_test = data_test['user_input']
Y_test = data_test['toxicity']

# Define terms for groups
terms = {
    'sexuality': ['gay', 'lesbian', 'bisexual', 'homosexual', 'straight', 'heterosexual'], 
    'gender identity': ['trans', 'transgender', 'cis', 'nonbinary'],
    'religion': ['christian', 'muslim', 'jewish', 'buddhist', 'catholic', 'protestant', 'sikh', 'taoist'],
    'race': ['african', 'african american', 'black', 'white', 'european', 'hispanic', 'latino', 'latina', 
            'latinx', 'mexican', 'canadian', 'american', 'asian', 'indian', 'middle eastern', 'chinese', 
            'japanese']
}

# Function to get groups
def get_groups(text, terms):
    group_names = list(terms.keys())
    num_groups = len(group_names)
    groups = np.zeros((text.shape[0], num_groups))
    for ii in range(num_groups):
        groups[:, ii] = text.str.contains('|'.join(terms[group_names[ii]]), case=False)
    return groups

A_train = get_groups(X_train, terms)[:, 0]
A_val = get_groups(X_val, terms)[:, 0]
A_test = get_groups(X_test, terms)[:, 0]

# Tokenization and vectorization
spacy_en = spacy.load("en_core_web_sm")

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_and_tag_data(data):
    tagged_data = [TaggedDocument(words=tokenize_en(_d), tags=[str(i)]) for i, _d in enumerate(data)]
    return tagged_data

tagged_train_data = tokenize_and_tag_data(X_train.tolist())

# Train a Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100)
doc2vec_model.build_vocab(tagged_train_data)
doc2vec_model.train(tagged_train_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

def text_to_vector(text):
    return doc2vec_model.infer_vector(tokenize_en(text))

def texts_to_matrix(texts):
    return torch.tensor([text_to_vector(text) for text in texts], dtype=torch.float)

X_train = texts_to_matrix(X_train.tolist()).numpy()
X_val = texts_to_matrix(X_val.tolist()).numpy()
X_test = texts_to_matrix(X_test.tolist()).numpy()

# Convert labels and groups
Y_train = np.where(Y_train==1,1,-1)
Y_val = np.where(Y_val==1,1,-1)
Y_test = np.where(Y_test==1,1,-1)
A_train = np.where(A_train==1,1,-1)
A_val = np.where(A_val==1,1,-1)
A_test = np.where(A_test==1,1,-1)

# Save preprocessed data
np.savez_compressed('chat_toxicity_text_matrix.npz', X_train=X_train, X_val=X_val, X_test=X_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, A_train=A_train, A_val=A_val, A_test=A_test)
