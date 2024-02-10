import torch
import itertools
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from base.mlp import MLP
from base.dcn import DCN
from base.autoint import AutoInt
from base.fibinet import FiBiNET
from utils.helpers import evaluate


class ReverseLayerF(Function):
    '''
     Forward pass of the layer which simply returns the input as is.
     @param ctx   The context that stores information for backpropagation.
     @return      The input tensor itself (identity operation).
    '''
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    '''
     Backward pass of the layer where the gradients are scaled
     and the sign is reversed.
     @param grad_output The gradient tensor from the previous layer.
     @return            The gradient tensor with its sign reversed
                        and scaled by alpha.
     '''
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    

class CDAN(nn.Module):
    def __init__(self, base, base_configs, training_configs, clf_size=1):
        super(CDAN, self).__init__()
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
        self.discriminator = nn.Linear(base_configs['output_size'] * 2, 1)
        self.training_configs = training_configs

    def forward(self, cat, num=None):
        hid = self.feature_extractor(cat, num)
        return torch.sigmoid(self.clf(hid))

    # Scheduler for alpha being applied in the reverse gradient layer.
    def get_lambda(self, epoch, max_epoch):
        p = epoch / max_epoch
        return 2. / (1 + np.exp(-10. * p)) - 1.

    def learn(self, source_loader, target_loader, test_loader, save_path, device):
        lrate = self.training_configs['lrate']
        num_epoch = self.training_configs['num_epoch']
        optim = torch.optim.Adam(self.parameters(), lr=lrate)
        criterion = nn.BCELoss()
        train_mean_clf_loss_per_epoch = []; train_mean_d_loss_per_epoch = []
        train_mean_loss_per_epoch = []
        nll_log = []; roc_auc_log = []; pr_auc_log = []
        for epoch in range(num_epoch):
            self.train()
            epoch_train_loss = []; epoch_clf_loss = []; epoch_d_loss = []
            target_it = iter(itertools.cycle(target_loader))
            for i, (src_num, src_cat, _, y) in enumerate(source_loader):
                tgt_num, tgt_cat, _, __ = next(target_it)
                src_cat = src_cat.to(dtype=torch.long, device=device)
                tgt_cat = tgt_cat.to(dtype=torch.long, device=device)
                y = y.to(dtype=torch.float32, device=device)
                # Create domain labels.
                src_d = torch.ones(src_cat.shape[0], 1, device=device)
                tgt_d = torch.zeros(tgt_cat.shape[0], 1, device=device)
                # Extract features for source & target domain data.
                if torch.all(src_num == False):
                    src_feat = self.feature_extractor(src_cat)
                    tgt_feat = self.feature_extractor(tgt_cat)
                else:
                    src_num = src_num.to(dtype=torch.float32, device=device)
                    tgt_num = tgt_num.to(dtype=torch.float32, device=device)
                    src_feat = self.feature_extractor(src_cat, src_num)
                    tgt_feat = self.feature_extractor(tgt_cat, tgt_num)
                # Make classifications on source & target domain data.
                src_pred = torch.sigmoid(self.clf(src_feat))
                tgt_pred = torch.sigmoid(self.clf(tgt_feat))
                one_minus_src = 1 - src_pred
                one_minus_tgt = 1 - tgt_pred
                src_pred_sfm = torch.cat((one_minus_src, src_pred), dim=1)
                tgt_pred_sfm = torch.cat((one_minus_tgt, tgt_pred), dim=1)
                # Multilinear conditioning.
                src_info = torch.bmm(src_pred_sfm.unsqueeze(2),
                                     src_feat.unsqueeze(1)).flatten(start_dim=1)
                tgt_info = torch.bmm(tgt_pred_sfm.unsqueeze(2),
                                     tgt_feat.unsqueeze(1)).flatten(start_dim=1)
                # Make domain classifications.
                alpha = self.get_lambda(epoch, num_epoch)
                src_info = ReverseLayerF.apply(src_info, alpha)
                tgt_info = ReverseLayerF.apply(tgt_info, alpha)
                src_d_pred = torch.sigmoid(self.discriminator(src_info))
                tgt_d_pred = torch.sigmoid(self.discriminator(tgt_info))
                # Compute loss.
                clf_loss = criterion(src_pred.squeeze(), y)
                d_loss = criterion(src_d_pred, src_d) + criterion(tgt_d_pred,
                                                                  tgt_d)
                loss = clf_loss + d_loss * self.training_configs['lambda']
                epoch_train_loss.append(loss.item())
                epoch_clf_loss.append(clf_loss.item())
                epoch_d_loss.append(d_loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()
            train_mean_loss_per_epoch.append(np.mean(epoch_train_loss))
            train_mean_clf_loss_per_epoch.append(np.mean(epoch_clf_loss))
            train_mean_d_loss_per_epoch.append(np.mean(epoch_d_loss))
            print('Epoch: [{}/{}], Mean: {:.9f}, CLF: {:.9f}, D: {:.9f}'.format(
                epoch+1, num_epoch, train_mean_loss_per_epoch[-1],
                train_mean_clf_loss_per_epoch[-1],
                train_mean_d_loss_per_epoch[-1]))
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
