
import torch
from tqdm import tqdm
import torch.nn as nn




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