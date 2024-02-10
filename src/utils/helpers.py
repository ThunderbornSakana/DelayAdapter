import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate(pred, true):
    # ROC AUC.
    roc_auc = roc_auc_score(true.detach().cpu().numpy(),
                            pred.detach().cpu().numpy())
    # PR AUC.
    pr_auc = average_precision_score(true.detach().cpu().numpy(),
                                     pred.detach().cpu().numpy())
    # Negative Log Likelihood.
    nll = F.binary_cross_entropy(pred.detach().cpu(), true.detach().cpu(),
                                 reduction='mean').item()
    return roc_auc, pr_auc, nll


def base_train(model, 
               lrate, 
               num_epoch, 
               train_loader, 
               test_loader, 
               save_path, 
               device):
    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    criterion = nn.BCELoss()
    train_mean_loss_per_epoch = []
    nll_log = []; roc_auc_log = []; pr_auc_log = []
    for epoch in range(num_epoch):
        model.train()
        epoch_train_loss = []
        for i, (num, cat, _, y) in enumerate(train_loader):
            cat = cat.to(dtype=torch.long, device=device)
            y = y.to(dtype=torch.float32, device=device)
            if torch.all(num == False):
                pred = model(cat).squeeze()
            else:
                num = num.to(dtype=torch.float32, device=device)
                pred = model(cat, num).squeeze()
            loss = criterion(pred, y)
            epoch_train_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        train_mean_loss_per_epoch.append(np.mean(epoch_train_loss))
        print('Epoch: [{}/{}], Average Loss: {:.9f}'.format(
            epoch+1, num_epoch, train_mean_loss_per_epoch[-1]))
        model.eval()
        pred_list = []; y_list = []
        for (test_num, test_cat, test_o, _) in test_loader:
            test_cat = test_cat.to(dtype=torch.long, device=device)
            test_o = test_o.to(dtype=torch.float32, device=device)
            if torch.all(test_num == False):
                with torch.no_grad():
                    pred = model(test_cat).squeeze()
            else:
                test_num = test_num.to(dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred = model(test_cat, test_num).squeeze()
            pred_list.append(pred); y_list.append(test_o)
        pred = torch.cat(pred_list); test_y = torch.cat(y_list)
        roc_auc, pr_auc, nll = evaluate(pred, test_y)
        nll_log.append(nll)
        roc_auc_log.append(roc_auc)
        pr_auc_log.append(pr_auc)
        print('NLL: {:.9f}; AUC: {:.9f}; PR_AUC: {:.9f}'.format(nll, 
                                                                roc_auc, 
                                                                pr_auc))
    return nll_log, roc_auc_log, pr_auc_log
