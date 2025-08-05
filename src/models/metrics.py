
import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        """Вычисляет Dice Loss"""
        preds = torch.sigmoid(logits)
        
        preds_flat = preds.contiguous().view(preds.shape[0], -1)
        labels_flat = labels.contiguous().view(labels.shape[0], -1)
        
        intersection = (preds_flat * labels_flat).sum(1)
        union = preds_flat.sum(1) + labels_flat.sum(1)
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss