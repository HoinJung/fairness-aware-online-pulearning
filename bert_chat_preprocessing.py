import pandas as pd
import numpy as np
from datasets import load_dataset
import torch
from transformers import BertTokenizer, BertModel
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

# Convert labels and groups
Y_train = np.where(Y_train==1,1,-1)
Y_val = np.where(Y_val==1,1,-1)
Y_test = np.where(Y_test==1,1,-1)
A_train = np.where(A_train==1,1,-1)
A_val = np.where(A_val==1,1,-1)
A_test = np.where(A_test==1,1,-1)

# Save numerical data
np.savez_compressed('chat_toxicity_bert_matrix.npz', Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, A_train=A_train, A_val=A_val, A_test=A_test)

# # Save text data separately as CSV
# data_train[['user_input']].to_csv('chat_toxicity_X_train.csv', index=False)
# data_val[['user_input']].to_csv('chat_toxicity_X_val.csv', index=False)
# data_test[['user_input']].to_csv('chat_toxicity_X_test.csv', index=False)

# Tokenize data and save input_ids and attention_mask
def tokenize_and_save(data, tokenizer, file_prefix):
    encodings = tokenizer(data.tolist(), padding='max_length', truncation=True, max_length=128, return_tensors='np')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # Save as .npz
    np.savez_compressed(f'{file_prefix}_tokenized.npz', input_ids=input_ids, attention_mask=attention_mask)
# Function to generate BERT embeddings for a given dataset
def get_bert_embeddings(data, tokenizer, model, max_length=128):
    embeddings = []
    attention_masks = []

    for text in data.tolist():
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        input_ids = inputs['input_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        
        # Get BERT embeddings for the input (CLS token)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token representation
        
        embeddings.append(cls_embedding.squeeze().cpu().numpy())
        
    
    return np.array(embeddings)
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to('cuda')

# Ensure BERT model is in evaluation mode (so it doesn't update during embedding extraction)
bert_model.eval()
# # Tokenize and save the training, validation, and test sets
# tokenize_and_save(X_train, tokenizer, 'chat_toxicity_X_train')
# tokenize_and_save(X_val, tokenizer, 'chat_toxicity_X_val')
# tokenize_and_save(X_test, tokenizer, 'chat_toxicity_X_test')


# Generate and save BERT embeddings for the training, validation, and test sets
X_train_embeddings = get_bert_embeddings(X_train, tokenizer, bert_model)
X_val_embeddings= get_bert_embeddings(X_val, tokenizer, bert_model)
X_test_embeddings = get_bert_embeddings(X_test, tokenizer, bert_model)

# Save the embeddings and attention masks
np.savez_compressed('chat_toxicity_X_train_bert_embeddings.npz', embeddings=X_train_embeddings)
np.savez_compressed('chat_toxicity_X_val_bert_embeddings.npz', embeddings=X_val_embeddings)
np.savez_compressed('chat_toxicity_X_test_bert_embeddings.npz', embeddings=X_test_embeddings)

# Save numerical labels (Y and A) as a separate file
np.savez_compressed('chat_toxicity_bert_matrix.npz', Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, A_train=A_train, A_val=A_val, A_test=A_test)