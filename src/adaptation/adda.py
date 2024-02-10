import torch
import itertools
import numpy as np
import torch.nn as nn
from base.mlp import MLP
from base.dcn import DCN
from base.autoint import AutoInt
from base.fibinet import FiBiNET
from utils.helpers import evaluate


class ADDA(nn.Module):
    def __init__(self, base, base_configs, training_configs, clf_size=1):
        super(ADDA, self).__init__()
        if base == 'MLP':
            self.feature_extractor = MLP(**base_configs)
            self.target_fex = MLP(**base_configs)
        elif base == 'DCN':
            self.feature_extractor = DCN(**base_configs)
            self.target_fex = DCN(**base_configs)
        elif base == 'AutoInt':
            self.feature_extractor = AutoInt(**base_configs)
            self.target_fex = AutoInt(**base_configs)
        elif base == 'FiBiNET':
            self.feature_extractor = FiBiNET(**base_configs)
            self.target_fex = FiBiNET(**base_configs)
        else:
            raise ValueError('Base model not supported.')
        self.clf = nn.Linear(base_configs['output_size'], clf_size)
        self.discriminator = nn.Linear(base_configs['output_size'], 1)
        self.training_configs = training_configs

    def forward(self, cat, num=None):
        hid = self.target_fex(cat, num)
        return torch.sigmoid(self.clf(hid))

    def pretrain(self, train_loader, device):
        self.train()
        lrate = self.training_configs['adda_pretrain_lrate']
        num_epoch = self.training_configs['adda_pretrain_num_epoch']
        params = list(self.feature_extractor.parameters()) + list(
            self.clf.parameters())
        optim = torch.optim.Adam(params, lr=lrate)
        criterion = nn.BCELoss()
        print(f'ADDA-Pretraining...')
        train_mean_loss_per_epoch = []
        for epoch in range(num_epoch):
            epoch_train_loss = []
            for i, (num, cat, _, y) in enumerate(train_loader):
                cat = cat.to(dtype=torch.long, device=device)
                y = y.to(dtype=torch.float32, device=device)
                if torch.all(num == False):
                    pred = torch.sigmoid(self.clf(
                        self.feature_extractor(cat))).squeeze()
                else:
                    num = num.to(dtype=torch.float32, device=device)
                    pred = torch.sigmoid(self.clf(
                        self.feature_extractor(cat, num))).squeeze()
                loss = criterion(pred, y)
                epoch_train_loss.append(loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()
            train_mean_loss_per_epoch.append(np.mean(epoch_train_loss))
            print('Epoch: [{}/{}], Average Loss: {:.9f}'.format(
                epoch+1, num_epoch, train_mean_loss_per_epoch[-1]))
        self.eval()

    def learn(self, source_loader, target_loader, test_loader, save_path, device):
        self.pretrain(source_loader, device)
        print('ADDA-Pretraining finished. Adversarial adaptation starts.')
        num_epoch = self.training_configs['num_epoch']
        lrate = self.training_configs['lrate']
        target_optim = torch.optim.Adam(self.target_fex.parameters(), lr=lrate)
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lrate)
        criterion = nn.BCELoss()
        # Initialize the target feature extractor with the pretrained extractor.
        self.target_fex.load_state_dict(self.feature_extractor.state_dict())
        train_mean_d_loss_per_epoch = []
        nll_log = []; roc_auc_log = []; pr_auc_log = []
        for epoch in range(num_epoch):
            epoch_d_loss = []
            target_it = iter(itertools.cycle(target_loader))
            for i, (src_num, src_cat, o, y) in enumerate(source_loader):
                tgt_num, tgt_cat, _, __ = next(target_it)
                src_cat = src_cat.to(dtype=torch.long, device=device)
                tgt_cat = tgt_cat.to(dtype=torch.long, device=device)
                y = y.to(dtype=torch.float32, device=device)
                # Extract features for source & target domain data.
                self.discriminator.train()
                if torch.all(src_num == False):
                    src_feat = self.feature_extractor(src_cat)
                    tgt_feat = self.target_fex(tgt_cat)
                else:
                    src_num = src_num.to(dtype=torch.float32, device=device)
                    tgt_num = tgt_num.to(dtype=torch.float32, device=device)
                    src_feat = self.feature_extractor(src_cat, src_num)
                    tgt_feat = self.target_fex(tgt_cat, tgt_num)
                # Create domain labels.
                src_d = torch.ones(src_cat.shape[0], 1, device=device)
                tgt_d = torch.zeros(tgt_cat.shape[0], 1, device=device)
                # Make domain classifications and update the discriminator.
                src_d_pred = torch.sigmoid(self.discriminator(src_feat))
                tgt_d_pred = torch.sigmoid(self.discriminator(tgt_feat))
                d_loss = criterion(src_d_pred, src_d) + criterion(tgt_d_pred,
                                                                  tgt_d)
                epoch_d_loss.append(d_loss.item())
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()
                # Flip domain labels for target domain data.
                tgt_d = torch.ones(tgt_cat.shape[0], 1, device=device)
                # Make domain classifications and update the feature extractor.
                self.target_fex.train(); self.discriminator.eval()
                if torch.all(src_num == False):
                    tgt_feat = self.target_fex(tgt_cat)
                else:
                    tgt_feat = self.target_fex(tgt_cat, tgt_num)
                tgt_d_pred = torch.sigmoid(self.discriminator(tgt_feat))
                d_loss2 = criterion(tgt_d_pred, tgt_d)
                target_optim.zero_grad()
                d_loss2.backward()
                target_optim.step()
            train_mean_d_loss_per_epoch.append(np.mean(epoch_d_loss))
            print('Epoch: [{}/{}], Average Discriminator Loss: {:.9f}'.format(
                epoch+1, num_epoch, train_mean_d_loss_per_epoch[-1]))
            self.eval()
            pred_list = []; y_list = []
            for (test_num, test_cat, test_o, _) in test_loader:
                test_cat = test_cat.to(dtype=torch.long, device=device)
                test_o = test_o.to(dtype=torch.float32, device=device)
                if torch.all(test_num == False):
                    with torch.no_grad():
                        pred = self(test_cat).squeeze()
                else:
                    test_num = test_num.to(dtype=torch.float32, device=device)
                    with torch.no_grad():
                        pred = self(test_cat, test_num).squeeze()
                pred_list.append(pred); y_list.append(test_o)
            pred = torch.cat(pred_list); test_y = torch.cat(y_list)
            roc_auc, pr_auc, nll = evaluate(pred, test_y)
            nll_log.append(nll)
            roc_auc_log.append(roc_auc)
            pr_auc_log.append(pr_auc)
            print('NLL: {:.9f}; AUC: {:.9f}; PR_AUC: {:.9f}'.format(
                nll, roc_auc, pr_auc))
        return nll_log, roc_auc_log, pr_auc_log
