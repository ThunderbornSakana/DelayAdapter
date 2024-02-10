import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from base.mlp import MLP
from base.dcn import DCN
from base.autoint import AutoInt
from base.fibinet import FiBiNET
from utils.helpers import evaluate


class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, input, iter_num=0, alpha=1.0,
                low_value=0.0, high_value=0.1, max_iter=1000.0):
        ctx.iter_num = iter_num
        ctx.alpha = alpha
        ctx.low_value = low_value
        ctx.high_value = high_value
        ctx.max_iter = max_iter
        ctx.save_for_backward(input)
        output = input.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        iter_num = ctx.iter_num
        alpha = ctx.alpha
        low_value = ctx.low_value
        high_value = ctx.high_value
        max_iter = ctx.max_iter
        iter_num += 1
        coeff = float(2.0 * (high_value - low_value) / (
            1.0 + np.exp(-alpha * iter_num / max_iter)) - (
                high_value - low_value) + low_value)
        return -coeff * grad_output, None, None, None, None, None
    

class MDD(nn.Module):
    def __init__(self, base, base_configs, training_configs, clf_size=1):
        super(MDD, self).__init__()
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
        self.clf = nn.Linear(base_configs['output_size'], clf_size)
        self.aux_clf = nn.Linear(base_configs['output_size'], clf_size)
        self.grl = GradientReverseLayer.apply
        self.training_configs = training_configs

    def forward(self, cat, num=None):
        hid = self.feature_extractor(cat, num)
        pred = torch.sigmoid(self.clf(hid))
        return pred

    # Returns: a tensor of shape (batch size, 2).
    def inverse_softmax(self, probabilities):
        logits_relative = torch.log(probabilities[:, 1] / probabilities[:, 0])
        logits = torch.stack([torch.zeros_like(logits_relative),
                              logits_relative], dim=1)
        return logits

    def learn(self, source_loader, target_loader, test_loader, save_path, device):
        clf_lrate = self.training_configs['clf_lrate']
        fex_lrate = self.training_configs['fex_lrate']
        num_epoch = self.training_configs['num_epoch']
        params = list(self.clf.parameters()) + list(self.aux_clf.parameters())
        clf_optim = torch.optim.Adam(params, lr=clf_lrate)
        fex_optim = torch.optim.Adam(self.feature_extractor.parameters(),
                                     lr=fex_lrate)
        ce = nn.CrossEntropyLoss()
        bce = nn.BCELoss()
        train_mean_clf_loss_per_epoch = []; mdd_per_epoch = []
        nll_log = []; roc_auc_log = []; pr_auc_log = []
        for epoch in range(num_epoch):
            self.train()
            epoch_clf_loss = []; epoch_mdd = []
            target_it = iter(itertools.cycle(target_loader))
            for i, (src_num, src_cat, _, y) in enumerate(source_loader):
                tgt_num, tgt_cat, _, __ = next(target_it)
                src_cat = src_cat.to(dtype=torch.long, device=device)
                tgt_cat = tgt_cat.to(dtype=torch.long, device=device)
                y = y.to(dtype=torch.float32, device=device)
                # Make predictions in an aggregated manner.
                cat = torch.cat((src_cat, tgt_cat), dim=0)
                if torch.all(src_num == False):
                    feat = self.feature_extractor(cat)
                else:
                    num = torch.cat((src_num, tgt_num), dim=0)
                    num = num.to(dtype=torch.float32, device=device)
                    feat = self.feature_extractor(cat, num)
                feat_adv = self.grl(feat)
                pred = torch.sigmoid(self.clf(feat))
                pred_adv = torch.sigmoid(self.aux_clf(feat_adv))
                one_minus = 1 - pred
                one_minus_adv = 1 - pred_adv
                sfm_out = torch.cat((one_minus, pred), dim=1)
                sfm_out_adv = torch.cat((one_minus_adv, pred_adv), dim=1)
                out_adv = self.inverse_softmax(sfm_out_adv)
                # Compute the classification loss on source data.
                clf_loss = bce(pred.narrow(0, 0, y.shape[0]).squeeze(), y)
                # Compute MDD source part.
                target_adv_src = sfm_out.narrow(0, 0, y.shape[0])
                mdd_src = ce(out_adv.narrow(0, 0, y.shape[0]), target_adv_src)
                # Compute MDD target part.
                target_adv_tgt = sfm_out.narrow(0, y.shape[0],
                                                cat.size(0) - y.shape[0])
                target_adv_tgt = target_adv_tgt.max(1)[1]
                logloss_tgt = torch.log(1 - F.softmax(
                    out_adv.narrow(0, y.shape[0], cat.size(0) - y.shape[0]
                                   ), dim = 1))
                mdd_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)
                # Aggregate all losses.
                mdd = self.training_configs['src_w'] * mdd_src + mdd_tgt
                loss = clf_loss + mdd * self.training_configs['lambda']
                epoch_clf_loss.append(clf_loss.item())
                epoch_mdd.append(mdd.item())
                clf_optim.zero_grad()
                fex_optim.zero_grad()
                loss.backward()
                clf_optim.step()
                fex_optim.step()
            train_mean_clf_loss_per_epoch.append(np.mean(epoch_clf_loss))
            mdd_per_epoch.append(np.mean(epoch_mdd))
            print('Epoch: [{}/{}], CLF: {:.9f}, MDD: {:.9f}'.format(
                epoch+1, num_epoch, train_mean_clf_loss_per_epoch[-1],
                mdd_per_epoch[-1]))
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
