import pandas as pd
import numpy as np
import pickle
import os
import torch
import pickle
from tqdm import tqdm
import gc
from  datetime import datetime as dt

DATAPATH = '/root/autodl-tmp'


# data = pd.read_csv('data/150w_result.csv')

# data = pd.read_csv('data/food_usage_need_training_test_set.csv')

# data = data.sample(frac=1, random_state=2024).reset_index(drop=True)

data = pd.DataFrame()


fs = os.listdir(DATAPATH + '/20M')

fs = sorted(fs)

fs = fs[:10]
print(fs)

for fp in tqdm(fs):
    df_ = pickle.load(open(f'{DATAPATH}/20M/{fp}', 'rb'))
    data = pd.concat([data, df_], axis = 0) 
    del df_
data = data.reset_index()

print(data.shape)

data = data.reset_index(drop = True)[['id', 'content']]

import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value


from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
# model = SentenceTransformer('BAAI/bge-base-zh-v1.5')

# model = SentenceTransformer('BAAI/bge-m3')
# model = SentenceTransformer('BAAI/llm-embedder')
# model = SentenceTransformer('google-bert/bert-base-chinese')
# model = SentenceTransformer('google-bert/bert-base-multilingual-uncased')
# model = SentenceTransformer('FacebookAI/xlm-roberta-base')
model = SentenceTransformer('Qwen/Qwen2-0.5B')
# model = SentenceTransformer('openai-community/gpt2')

sum([np.prod(p.size()) for p in model.parameters()])


import json 

CHUNK_SIZE = 10000
BATCH_SIZE = 50
OUTPUT_DIR = f'{DATAPATH}/qwen2_05B_embeddings'

os.makedirs(OUTPUT_DIR, exist_ok=True)

embeding_idx = []
for filename in os.listdir(OUTPUT_DIR):
  with open(f'{OUTPUT_DIR}/{filename}', 'rb') as f:
    embeding_idx += list((pickle.load(f)).keys())

data_rest = data[~data['id'].isin(embeding_idx)]

print("dropped data size: ", data[data['id'].isin(embeding_idx)].shape)
print("remaining data size: ", data_rest.shape)
del embeding_idx

data_rest['batch'] = [int(i/CHUNK_SIZE) for i in range(len(data_rest))]

print('cuda: ', torch.cuda.is_available())
torch.cuda.empty_cache()
gc.collect()
for b, d in tqdm(data_rest.groupby('batch')):
  timestamp = dt.now().strftime('%Y-%m-%d_%H:%M:%S')
  with torch.no_grad():
    embeddings = model.encode(d['content'].tolist(), show_progress_bar=True, batch_size=BATCH_SIZE, device='cuda')
    embeddings_dict = dict(zip(d['id'].tolist(), embeddings.tolist()))
    #pickle.dump(embeddings_dict, open(os.path.join(OUTPUT_DIR, f'batch_{timestamp}.pkl'), 'wb'))
    with open(f'{OUTPUT_DIR}/batch1_{timestamp}.pickle', 'wb') as f :
        pickle.dump(embeddings_dict, f)
  del embeddings
  torch.cuda.empty_cache()
  gc.collect()

