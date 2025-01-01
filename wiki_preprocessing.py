import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import spacy
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def get_groups(text,terms):
    group_names = list(terms.keys())
    num_groups = len(group_names)

    # Returns a boolean NumPy array of shape (n, k), where n is the number of comments, 
    # and k is the number of groups. Each entry (i, j) indicates if the i-th comment 
    # contains a term from the j-th group.
    groups = np.zeros((text.shape[0], num_groups))
    for ii in range(num_groups):
        groups[:, ii] = text.str.contains('|'.join(terms[group_names[ii]]), case=False)
    return groups
toxicity_data_url = ("https://github.com/conversationai/unintended-ml-bias-analysis/"
                    "raw/e02b9f12b63a39235e57ba6d3d62d8139ca5572c/data/")
data_train = pd.read_csv(toxicity_data_url + "wiki_train.csv")
data_val = pd.read_csv(toxicity_data_url + "wiki_dev.csv")
data_test = pd.read_csv(toxicity_data_url + "wiki_test.csv")
# tokenizer = get_tokenizer("basic_english")
Y_train = data_train["is_toxic"].values.astype(float)
Y_val = data_val["is_toxic"].values.astype(float)
Y_test = data_test["is_toxic"].values.astype(float)

terms = {
    'sexuality': ['gay', 'lesbian', 'bisexual', 'homosexual', 'straight', 'heterosexual'], 
    'gender identity': ['trans', 'transgender', 'cis', 'nonbinary'],
    'religion': ['christian', 'muslim', 'jewish', 'buddhist', 'catholic', 'protestant', 'sikh', 'taoist'],
    'race': ['african', 'african american', 'black', 'white', 'european', 'hispanic', 'latino', 'latina', 
            'latinx', 'mexican', 'canadian', 'american', 'asian', 'indian', 'middle eastern', 'chinese', 
            'japanese']}


A_train = get_groups(data_train["comment"],terms)[:,0]
A_val = get_groups(data_val["comment"],terms)[:,0]
A_test = get_groups(data_test["comment"],terms)[:,0]

X_train = data_train['comment'].tolist()
X_val = data_val['comment'].tolist()
X_test = data_test['comment'].tolist()


spacy_en = spacy.load("en_core_web_sm")
# word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_and_tag_data(data):
    tagged_data = [TaggedDocument(words=tokenize_en(_d), tags=[str(i)]) for i, _d in enumerate(data)]
    return tagged_data

tagged_train_data = tokenize_and_tag_data(X_train)

# Train a Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100)
doc2vec_model.build_vocab(tagged_train_data)
doc2vec_model.train(tagged_train_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

def text_to_vector(text):
    return doc2vec_model.infer_vector(tokenize_en(text))

def texts_to_matrix(texts):
    return torch.tensor([text_to_vector(text) for text in texts], dtype=torch.float)
X_test = texts_to_matrix(X_test).numpy()
X_val = texts_to_matrix(X_val).numpy()
X_train = texts_to_matrix(X_train).numpy()



Y_train = np.where(Y_train==1,1,-1)
Y_val = np.where(Y_val==1,1,-1)
Y_test = np.where(Y_test==1,1,-1)
A_train = np.where(A_train==1,1,-1)
A_val = np.where(A_val==1,1,-1)
A_test = np.where(A_test==1,1,-1)
# Example for saving train data:


np.savez_compressed('wiki_text_matrix.npz', X_train=X_train,X_val=X_val,  X_test=X_test, Y_train=Y_train,Y_val=Y_val, Y_test=Y_test, A_train=A_train, A_val = A_val,A_test = A_test)