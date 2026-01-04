import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # preds: [batch, num_quantiles]
        # target: [batch, 1]
        loss = 0
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.quantile_loss = QuantileLoss(quantiles)
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, q_preds, prob_preds, target_return, target_dir):
        # q_preds: [batch, num_quantiles] (Predicted returns)
        # prob_preds: [batch, 1] (Predicted probability of up)
        # target_return: [batch] (Actual return)
        # target_dir: [batch] (Actual direction 0/1)
        
        q_loss = self.quantile_loss(q_preds, target_return.unsqueeze(1))
        p_loss = self.bce_loss(prob_preds, target_dir.unsqueeze(1).float())
        
        return self.alpha * q_loss + self.beta * p_loss
