import torch
import itertools
import numpy as np
import torch.nn as nn
from base.mlp import MLP
from base.dcn import DCN
from base.autoint import AutoInt
from base.fibinet import FiBiNET
from utils.helpers import evaluate


class MCD(nn.Module):
    def __init__(self, base, base_configs, training_configs, clf_size=1):
        super(MCD, self).__init__()
        if base == 'MLP':
            self.feature_extractor = MLP(**base_configs)
        elif base == 'DCN':
            self.feature_extractor = DCN(**base_configs)
        elif base == 'AutoInt':
            self.feature_extractor = AutoInt(**base_configs)
        elif base == 'FiBiNET':
            self.feature_extractor = FiBiNET(**base_configs)
        else:
            raise ValueError('Base model not supported.')
        self.clf_1 = nn.Linear(base_configs['output_size'], clf_size)
        self.clf_2 = nn.Linear(base_configs['output_size'], clf_size)
        self.training_configs = training_configs

    def forward(self, cat, num=None):
        hid = self.feature_extractor(cat, num)
        pred_1 = torch.sigmoid(self.clf_1(hid))
        pred_2 = torch.sigmoid(self.clf_2(hid))
        return (pred_1 + pred_2) / 2.0

    def step_a(self, num, cat, y, optim, device):
        criterion = nn.BCELoss()
        self.train()
        if torch.all(num == False):
            hid = self.feature_extractor(cat)
        else:
            num = num.to(dtype=torch.float32, device=device)
            hid = self.feature_extractor(cat, num)
        pred_1 = torch.sigmoid(self.clf_1(hid)).squeeze()
        pred_2 = torch.sigmoid(self.clf_2(hid)).squeeze()
        loss_1 = criterion(pred_1, y)
        loss_2 = criterion(pred_2, y)
        loss = (loss_1 + loss_2) / 2
        optim.zero_grad()
        loss.backward()
        optim.step()
        self.eval()
        return loss_1, loss_2

    def step_b(self, src_num, src_cat, tgt_num, tgt_cat, y, optim, device):
        criterion = nn.BCELoss()
        self.clf_1.train()
        self.clf_2.train()
        # Classify source domain data and compute accuracy loss.
        if torch.all(src_num == False):
            hid = self.feature_extractor(src_cat)
        else:
            src_num = src_num.to(dtype=torch.float32, device=device)
            hid = self.feature_extractor(src_cat, src_num)
        pred_1 = torch.sigmoid(self.clf_1(hid)).squeeze()
        pred_2 = torch.sigmoid(self.clf_2(hid)).squeeze()
        clf_loss = (criterion(pred_1, y) + criterion(pred_2, y)) / 2
        # Classify target domain data and compute discrepancy loss.
        if torch.all(tgt_num == False):
            hid = self.feature_extractor(tgt_cat)
        else:
            tgt_num = tgt_num.to(dtype=torch.float32, device=device)
            hid = self.feature_extractor(tgt_cat, tgt_num)
        tgt_pred_1 = torch.sigmoid(self.clf_1(hid)).squeeze()
        tgt_pred_2 = torch.sigmoid(self.clf_2(hid)).squeeze()
        discrepancy = (tgt_pred_1 - tgt_pred_2).abs().mean()
        # Minimize the accuracy loss and maximize the discrepancy.
        loss = clf_loss - discrepancy
        optim.zero_grad()
        loss.backward()
        optim.step()
        self.clf_1.eval(); self.clf_2.eval()

    def step_c(self, num, cat, optim, device):
        self.feature_extractor.train()
        for _ in range(self.training_configs['num_steps_c']):
            # Classify target domain data.
            if torch.all(num == False):
                hid = self.feature_extractor(cat)
            else:
                num = num.to(dtype=torch.float32, device=device)
                hid = self.feature_extractor(cat, num)
            pred_1 = torch.sigmoid(self.clf_1(hid)).squeeze()
            pred_2 = torch.sigmoid(self.clf_2(hid)).squeeze()
            # Compute discrepancy loss and minimize it.
            discrepancy = (pred_1 - pred_2).abs().mean()
            optim.zero_grad()
            discrepancy.backward()
            optim.step()
        self.eval()
        return discrepancy

    def learn(self, source_loader, target_loader, test_loader, save_path, device):
        num_epoch = self.training_configs['num_epoch']
        lrate = self.training_configs['lrate']
        step_a_optim = torch.optim.Adam(self.parameters(), lr=lrate)
        params_b = list(self.clf_1.parameters()) + list(self.clf_2.parameters())
        step_b_optim = torch.optim.Adam(params_b, lr=lrate)
        params_c = list(self.feature_extractor.parameters())
        step_c_optim = torch.optim.Adam(params_c, lr=lrate)
        clf_1_loss_per_epoch = []; clf_2_loss_per_epoch = []
        discrepancy_per_epoch = []
        nll_log = []; roc_auc_log = []; pr_auc_log = []
        for epoch in range(num_epoch):
            epoch_clf_1_loss = []; epoch_clf_2_loss = []
            epoch_discrepancy = []
            target_it = iter(itertools.cycle(target_loader))
            for i, (src_num, src_cat, _, y) in enumerate(source_loader):
                tgt_num, tgt_cat, _, __ = next(target_it)
                src_cat = src_cat.to(dtype=torch.long, device=device)
                tgt_cat = tgt_cat.to(dtype=torch.long, device=device)
                y = y.to(dtype=torch.float32, device=device)
                loss_1, loss_2 = self.step_a(src_num, src_cat, y, step_a_optim, device)
                epoch_clf_1_loss.append(loss_1.item())
                epoch_clf_2_loss.append(loss_2.item())
                # Step A finished.
                self.step_b(src_num, src_cat, tgt_num, tgt_cat, y, step_b_optim, device)
                # Step B finished.
                discrepancy = self.step_c(tgt_num, tgt_cat, step_c_optim, device)
                epoch_discrepancy.append(discrepancy.item())
                # Step C finished.
            clf_1_loss_per_epoch.append(np.mean(epoch_clf_1_loss))
            clf_2_loss_per_epoch.append(np.mean(epoch_clf_2_loss))
            discrepancy_per_epoch.append(np.mean(epoch_discrepancy))
            print('Epoch: [{}/{}], CLF 1/2: {:.9f}/{:.9f}, DISC: {:.9f}'.format(
                epoch+1, num_epoch, clf_1_loss_per_epoch[-1],
                clf_2_loss_per_epoch[-1], discrepancy_per_epoch[-1]))
            # Start evaluation.
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
