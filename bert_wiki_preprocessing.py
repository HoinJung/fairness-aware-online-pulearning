import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
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


import numpy as np

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
A_train = get_groups(data_train["comment"],terms)[:,0]
A_val = get_groups(data_val["comment"],terms)[:,0]
A_test = get_groups(data_test["comment"],terms)[:,0]

X_train = data_train['comment']
X_val = data_val['comment']
X_test = data_test['comment']



Y_train = np.where(Y_train==1,1,-1)
Y_val = np.where(Y_val==1,1,-1)
Y_test = np.where(Y_test==1,1,-1)
A_train = np.where(A_train==1,1,-1)
A_val = np.where(A_val==1,1,-1)
A_test = np.where(A_test==1,1,-1)
# Example for saving train data:

# Function to generate BERT embeddings for a given dataset using CUDA
def get_bert_embeddings(data, tokenizer, model, max_length=128):
    embeddings = []
    
    # Loop through the input data to get embeddings
    for text in data.tolist():
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        input_ids = inputs['input_ids'].to('cuda')  # Move input_ids to GPU
        attention_mask = inputs['attention_mask'].to('cuda')  # Move attention_mask to GPU
        
        # Get BERT embeddings for the input (CLS token)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token representation
        
        # Transfer embeddings back to CPU and append to the list
        embeddings.append(cls_embedding.squeeze().cpu().numpy())
    
    return np.array(embeddings)

# Initialize tokenizer and model, and move model to CUDA
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to('cuda')  # Move model to GPU

# Ensure BERT model is in evaluation mode (so it doesn't update during embedding extraction)
bert_model.eval()

# Generate and save BERT embeddings for the training, validation, and test sets
X_train_embeddings = get_bert_embeddings(X_train, tokenizer, bert_model)
X_val_embeddings = get_bert_embeddings(X_val, tokenizer, bert_model)
X_test_embeddings = get_bert_embeddings(X_test, tokenizer, bert_model)

# Save the embeddings
np.savez_compressed('wiki_X_train_bert_embeddings.npz', embeddings=X_train_embeddings)
np.savez_compressed('wiki_X_val_bert_embeddings.npz', embeddings=X_val_embeddings)
np.savez_compressed('wiki_X_test_bert_embeddings.npz', embeddings=X_test_embeddings)

# Save numerical labels (Y and A) as a separate file
np.savez_compressed('wiki_bert_matrix.npz', Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, A_train=A_train, A_val=A_val, A_test=A_test)