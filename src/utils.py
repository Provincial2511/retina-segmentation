
import torch
from tqdm import tqdm
import torch.nn as nn


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

def inference(model, data_loader, metric_computer, device):
    """Выполняет инференс и оценку модели"""
    predicted_masks = []
    model.eval()
    
    # Перед использованием сбрасываем состояние метрики с прошлых запусков
    metric_computer.reset()
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Inference"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Обновляем состояние метрики
            metric_computer.update(outputs, labels)
            
            # Сохраняем маски для визуализации
            preds_proba = torch.sigmoid(outputs)
            preds_binary = (preds_proba > 0.5).float()
            predicted_masks.append(preds_binary.cpu())
            
    # Вычисляем итоговую метрику
    final_metric_value = metric_computer.compute()
    
    all_predicted_masks = torch.cat(predicted_masks, dim=0)
    
    return final_metric_value.item(), all_predicted_masks