import os
import json
import pickle
import pandas as pd
from funcs import *


config_path = './configs.json'
with open(config_path, 'r') as config_file:
    all_combos = json.load(config_file)

active_combo = [combo['params'] for combo in all_combos if combo['use']][0]

if active_combo['dataset'] == 'criteo':
    train_x, train_y, test_x, test_y, cat_dims = preprocess_criteo(
        active_combo['train_start'], active_combo['train_end'], active_combo['num_test'])
elif active_combo['dataset'] == 'tencent':
    train_x, train_y, test_x, test_y, cat_dims = preprocess_tencent(
        active_combo['train_start'], active_combo['train_end'], active_combo['num_test'])
elif active_combo['dataset'] == 'taobao':
    train_x, train_y, test_x, test_y, cat_dims = preprocess_taobao(
        active_combo['train_start'], active_combo['train_end'], active_combo['num_test'], 
        active_combo['midpoint'])

# Saving all preprocessed artifacts.
data_dir = f"{active_combo['save_path']}/{active_combo[
    'dataset']}_{active_combo['train_start']}_{active_combo[
        'train_end']}_{active_combo['num_test']}"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
train_y.to_csv(f'{data_dir}/train_y.csv', index=False)
train_x.to_csv(f'{data_dir}/train_x.csv', index=False)
test_y.to_csv(f'{data_dir}/test_y.csv', index=False)
test_x.to_csv(f'{data_dir}/test_x.csv', index=False)
with open(f"{data_dir}/cat_dims_{active_combo['dataset']}.pkl", 'wb') as handle:
    pickle.dump(cat_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)
