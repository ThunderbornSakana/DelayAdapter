import os
import json
import torch
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from utils.datasets import CVR_Dataset
from utils.helpers import base_train
warnings.filterwarnings("ignore")

NUM_SEC_PER_MINUTE = 60
NUM_SEC_PER_HOUR = NUM_SEC_PER_MINUTE * 60
NUM_SEC_PER_DAY = NUM_SEC_PER_HOUR * 24


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_idx', type=int, default=0, help="GPU index")
    parser.add_argument('--seed', type=int, default=2002, help="random seed")
    parser.add_argument('--tag', type=int, default=0, help="experiment tag")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")

    # Set random seed and directory information; create directories for saving training results.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    currentDateAndTime = datetime.now()
    model_directory = f'{currentDateAndTime.strftime("%m_%d_%YM%H_%M_%S")}__{args.seed}__{args.tag}'
    model_path = os.path.join('../checkpoints/', model_directory)
    os.mkdir(model_path)
    print(model_directory)

    # Which setting to experiment with.
    config_path = './configs.json'
    with open(config_path, 'r') as config_file:
        all_hp_combos = json.load(config_file)

    hp = [combo for combo in all_hp_combos if combo['tag'] == args.tag][0]

    # Write the hyperparameters to a .txt file.
    text_path = f'../checkpoints/{model_directory}/info.txt'
    with open(text_path, 'w') as file:
        file.write(json.dumps(hp, indent=4))

    # Load in preprocessed data.
    data_params = hp['data_params']
    dataset = data_params['dataset']
    train_start = data_params['train_start']
    train_end = data_params['train_end']
    num_test = data_params['num_test']
    data_dir = f'../data/{dataset}/{dataset}_{train_start}_{train_end}_{num_test}'
    train_x = pd.read_csv(f'{data_dir}/train_x.csv')
    train_y = pd.read_csv(f'{data_dir}/train_y.csv')
    test_x = pd.read_csv(f'{data_dir}/test_x.csv')
    test_y = pd.read_csv(f'{data_dir}/test_y.csv')
    with open(f'{data_dir}/cat_dims_{dataset}.pkl', 'rb') as handle:
        cat_dims = pickle.load(handle)
    if dataset == "criteo":
        timestamp_col = 'timestamp'
        convert_col = 'convertTimestamp'
        num_fields = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8']
        cat_fields = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
        num_num = len(num_fields)
        test_num = np.array(test_x[num_fields])
        attribution_window = 30.1
    elif dataset == "taobao":
        timestamp_col = 'pvTime'
        convert_col = 'buyTime'
        num_num = 0
        cat_fields = ['UserID', 'ProductID', 'Product_type_ID']
        test_num = None
        train_num = None
        attribution_window = 4.5
    elif dataset == "tencent":
        timestamp_col = 'clickTime_sec'
        convert_col = 'conversionTime_sec'
        num_num = 0
        cat_fields = ["connectionType", "telecomsOperator", "age", "gender", 
                      "education", "marriageStatus", "haveBaby", "hometown", 
                      "sitesetID", "positionType", "appID", "appPlatform", 
                      "appCategory", "residence", "advertiserID"]
        test_num = None
        train_num = None
        attribution_window = 4.97
    else:
        raise ValueError('Dataset not supported.')
    test_convertTime = np.array(test_x[convert_col])
    test_start_time = test_x[timestamp_col].min()
    timestamp_max = train_x[timestamp_col].max()
    test_cat = np.array(test_x[cat_fields])
    test_y = np.array(test_y).squeeze().astype(float)
    test_data = CVR_Dataset(test_cat, test_y, test_start_time, 
                            test_convertTime, test_num)
    test_loader = DataLoader(test_data, 
                             batch_size=hp['batch_size'], shuffle=False)
    
    # Regular: disregard the delayed feedback effect and use complete training data.
    # Oracle: assume we have access to true labels, even unobserved.
    # Safe: only use the portion of data that have all labels fixed (except taobao).
    if hp['mode'] in ["regular", "oracle", "safe"]:
        if hp['mode'] == "safe":
            train_y = train_y[train_x[timestamp_col] < (
                timestamp_max - attribution_window * NUM_SEC_PER_DAY)]
            train_x = train_x[train_x[timestamp_col] < (
                timestamp_max - attribution_window * NUM_SEC_PER_DAY)]
        if hp['mode'] == "oracle":
            # Overwrite the conversion time all to 0.
            train_x[convert_col] = 0.0
        if dataset == "criteo":
            train_num = np.array(train_x[num_fields])
        train_cat = np.array(train_x[cat_fields])
        train_convertTime = np.array(train_x[convert_col])
        train_y = np.array(train_y).squeeze().astype(float)
        train_data = CVR_Dataset(train_cat, train_y, test_start_time, 
                                 train_convertTime, train_num)
        train_loader = DataLoader(train_data, 
                                  batch_size=hp['batch_size'], 
                                  shuffle=True)
    elif hp['mode'] in ["only pretrain", "adapt"]:
        target_ds = data_params['num_target']
        source_x = train_x[train_x[timestamp_col] 
                           <= (timestamp_max - target_ds * NUM_SEC_PER_DAY)]
        source_y = train_y[train_x[timestamp_col] 
                           <= (timestamp_max - target_ds * NUM_SEC_PER_DAY)]
        target_x = train_x[train_x[timestamp_col] 
                           > (timestamp_max - target_ds * NUM_SEC_PER_DAY)]
        target_y = train_y[train_x[timestamp_col] 
                           > (timestamp_max - target_ds * NUM_SEC_PER_DAY)]
        if dataset == "criteo":
            source_num = np.array(source_x[num_fields])
            target_num = np.array(target_x[num_fields])
        else:
            source_num = None
            target_num = None
        source_cat = np.array(source_x[cat_fields])
        source_convertTime = np.array(source_x[convert_col])
        source_y = np.array(source_y).squeeze().astype(float)
        target_cat = np.array(target_x[cat_fields])
        target_convertTime = np.array(target_x[convert_col])
        target_y = np.array(target_y).squeeze().astype(float)
        if hp['pretrain']:
            # Create a dataset that only has safe labels.
            safe_x = train_x[train_x[timestamp_col] < (
                timestamp_max - attribution_window * NUM_SEC_PER_DAY)]
            safe_y = train_y[train_x[timestamp_col] < (
                timestamp_max - attribution_window * NUM_SEC_PER_DAY)]
            safe_y = np.array(safe_y).squeeze().astype(float)
            # Extract positive target datas (oracle) from target dataset.
            pos_target_x = target_x[target_y == 1.0]
            # Only keep observed positives.
            pos_target_x = pos_target_x[pos_target_x[convert_col] < (
                test_start_time - NUM_SEC_PER_HOUR)]
            '''
                Sub-sample some safe negatives to balance training.
            '''
            pos_safe_x = safe_x[safe_y == 1.0]
            neg_safe_x = safe_x[safe_y == 0.0]
            neg_safe_x = neg_safe_x.sample(int(len(pos_safe_x)))
            pos_safe_y = np.ones(len(pos_safe_x))
            neg_safe_y = np.zeros(len(pos_safe_x))
            # Combine x and y for safe set and target positive set.
            pos_y = np.ones(len(pos_target_x))
            pretrain_x = pd.concat((pos_target_x, pos_safe_x, neg_safe_x), axis=0)
            pretrain_y = np.concatenate((pos_y, pos_safe_y, neg_safe_y), axis=0)
            pretrain_cat = np.array(pretrain_x[cat_fields])
            pretrain_num = None
            if dataset == "criteo":
                pretrain_num = np.array(pretrain_x[num_fields])
            pretrain_convertTime = np.array(pretrain_x[convert_col])
            pretrain_data = CVR_Dataset(pretrain_cat, pretrain_y, test_start_time, 
                                        pretrain_convertTime, pretrain_num)
            pretrain_loader = DataLoader(pretrain_data, 
                                         batch_size=hp['batch_size'], 
                                         shuffle=True)
        if hp['mode'] == "adapt":
            source_data = CVR_Dataset(source_cat, source_y, test_start_time, 
                                      source_convertTime, source_num)
            source_loader = DataLoader(source_data, 
                                       batch_size=hp['batch_size'], 
                                       shuffle=True)
            target_data = CVR_Dataset(target_cat, target_y, test_start_time, 
                                      target_convertTime, target_num)
            target_loader = DataLoader(target_data, 
                                       batch_size=hp['batch_size'], 
                                       shuffle=True)
    else:
        raise ValueError('Wrong mode.')

    print("Data loading and preprocessing are finished.")
    predict = False
    if hp['mode'] in ["regular", "oracle", "safe"]:
        predict = True
    base_dict = hp['base_params']
    base_dict["predict"] = predict
    base_dict["num_num"] = num_num
    base_dict["cat_dims"] = cat_dims
    if hp['base'] == "MLP":
        from base.mlp import MLP
        if hp['mode'] in ["regular", "oracle", "safe"]:
            model = MLP(**base_dict).to(device)
    elif hp['base'] == "DCN":
        from base.dcn import DCN
        if hp['mode'] in ["regular", "oracle", "safe"]:
            model = DCN(**base_dict).to(device)
    elif hp['base'] == "AutoInt":
        from base.autoint import AutoInt
        if hp['mode'] in ["regular", "oracle", "safe"]:
            model = AutoInt(**base_dict).to(device)
    elif hp['base'] == "FiBiNET":
        from base.fibinet import FiBiNET
        if hp['mode'] in ["regular", "oracle", "safe"]:
            model = FiBiNET(**base_dict).to(device)
    else:
        raise ValueError('Base model not supported.')

    print("Base model import is finished.")
    column_names = ['NLL', 'ROC AUC', 'PR AUC']
    save_path = f'../checkpoints/{model_directory}'
    if hp['mode'] in ["regular", "oracle", "safe"]:
        print(f'#Parameters: {sum(param.numel() for param in model.parameters())}')
        nll_log, roc_auc_log, pr_auc_log = base_train(model, hp['train_params']['lrate'], 
                                                      hp['train_params']['num_epoch'], 
                                                      train_loader, test_loader, 
                                                      save_path, device)
        df = pd.DataFrame(list(zip(nll_log, roc_auc_log, pr_auc_log)), columns=column_names)
        csv_file_path = f'{save_path}/results.csv'
        df.to_csv(csv_file_path, index=False)
        print("An experiment without domain adaptation is finished.")

    if hp['adaptation'] == "DANN":
        from adaptation.dann import DANN
        model = DANN(hp['base'], base_dict, hp['adapt_params']).to(device)
    elif hp['adaptation'] == "ADDA":
        from adaptation.adda import ADDA
        model = ADDA(hp['base'], base_dict, hp['adapt_params']).to(device)
    elif hp['adaptation'] == "CDAN":
        from adaptation.cdan import CDAN
        model = CDAN(hp['base'], base_dict, hp['adapt_params']).to(device)
    elif hp['adaptation'] == "MCD":
        from adaptation.mcd import MCD
        model = MCD(hp['base'], base_dict, hp['adapt_params']).to(device)
    elif hp['adaptation'] == "MDD":
        from adaptation.mdd import MDD
        model = MDD(hp['base'], base_dict, hp['adapt_params']).to(device)
    else:
        print("No domain adaptation.")
        
    if hp['pretrain'] == True:
        print(f'#Parameters: {sum(param.numel() for param in model.parameters())}')
        nll_log, roc_auc_log, pr_auc_log = base_train(model, hp['train_params']['pretrain_lrate'], 
                                                      hp['train_params']['pretrain_num_epoch'], 
                                                      pretrain_loader, test_loader, save_path, device)
        df = pd.DataFrame(list(zip(nll_log, roc_auc_log, pr_auc_log)), columns=column_names)
        csv_file_path = f'{save_path}/pretrain_results.csv'
        df.to_csv(csv_file_path, index=False)
        print("Pretraining is finished.")
        
    if hp['mode'] == "adapt":
        nll_log, roc_auc_log, pr_auc_log = model.learn(source_loader, target_loader, 
                                                       test_loader, save_path, device)
        df = pd.DataFrame(list(zip(nll_log, roc_auc_log, pr_auc_log)), columns=column_names)
        csv_file_path = f'{save_path}/results.csv'
        df.to_csv(csv_file_path, index=False)
        print("Domain adaptation is finished.")


    min_nll_index = nll_log.index(min(nll_log))
    print(f"Best NLL: {nll_log[min_nll_index]}")
    print(f"Best ROC AUC: {roc_auc_log[min_nll_index]}")
    print(f"Best PR AUC: {pr_auc_log[min_nll_index]}")
