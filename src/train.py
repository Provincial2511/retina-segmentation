import os
from glob import glob

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import argparse

from dataset import RetinaDataset
from network import R2UNet, ResAttU_Net
import utils

def get_args():
    parser = argparse.ArgumentParser(description="Train U-Net models for segmentation")
    parser.add_argument('--model', type=str, default='R2UNet', help='Model to train (UNet or R2AttU_Net)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    return parser.parse_args()

def fit_epoch(model, criterion, optimizer, metric, train_loader, device):
    """Обучает модель на одной эпохе."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    epoch_iou = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        epoch_iou += metric(outputs, labels).item()

    epoch_loss = running_loss / total_samples
    epoch_iou /= len(train_loader)
    return epoch_loss, epoch_iou

def val_epoch(model, criterion, metric, val_loader, device):
    """Валидирует модель на одной эпохе"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    epoch_iou = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            epoch_iou += metric(outputs, labels).item()

    epoch_loss = running_loss / total_samples
    epoch_iou /= len(val_loader)
    return epoch_loss, epoch_iou

def train_model(model, criterion, optimizer, metric, train_loader, val_loader, epochs, device):
    """Обучает модель на протяжении нескольких эпох."""
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        tqdm.write(f'--- Эпоха {epoch + 1}/{epochs} ---')

        train_loss, train_iou = fit_epoch(model, criterion, optimizer, metric, train_loader, device)
        val_loss, val_iou = val_epoch(model, criterion, metric, val_loader, device)
        
        # logger для ClearML. при необходимости - убрать комментарий
        # logger.report_scalar(title="Loss", series="Train", value=train_loss, iteration=epoch)
        # logger.report_scalar(title="IoU", series="Train", value=train_iou, iteration=epoch)
        # logger.report_scalar(title="Loss", series="Validation", value=val_loss, iteration=epoch)
        # logger.report_scalar(title="IoU", series="Validation", value=val_iou, iteration=epoch)
        
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")

    plot_losses(train_losses, val_losses, train_ious, val_ious)

def plot_losses(train_losses, val_losses, train_iou, val_iou):
    """Визуализирует потери и IoU на графиках."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_iou, label="Train IoU")
    plt.plot(val_iou, label="Val IoU")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.title("Train and Validation IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = '/kaggle/input/retina-blood-vessel/Data'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    train_image_path = os.path.join(train_dir, 'image')
    train_mask_path = os.path.join(train_dir, 'mask')

    test_image_path = os.path.join(test_dir, 'image')
    test_mask_path = os.path.join(test_dir, 'mask')

    paired_train_images = []
    paired_train_masks = []

    for img_path in sorted(glob(os.path.join(train_image_path, '*.png'))):
        basename = os.path.basename(img_path)
        expected_mask_path = os.path.join(train_mask_path, basename)

        
        if os.path.exists(expected_mask_path):
            paired_train_images.append(img_path)
            paired_train_masks.append(expected_mask_path)

    print(f"Найдено {len(paired_train_images)} пар в тренировочной выборке")

    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    compute_mean = [0.504, 0.275, 0.164] # Рассчет был произведен на тренировочной выборке
    compute_std = [0.345, 0.189, 0.107]

    train_transforms = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=compute_mean, std=compute_std),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=compute_mean, std=compute_std),
        ToTensorV2(),
    ])

    num_workers = 2
    batch_size = 16

    # Разобьем данные и сделаем даталоадеры
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        paired_train_images,
        paired_train_masks,
        test_size=0.2,
        random_state=42 
    )

    train_data = RetinaDataset(train_img_paths, train_mask_paths, transforms=train_transforms)
    val_data = RetinaDataset(val_img_paths, val_mask_paths, transforms=val_transforms)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if args.model == 'UNet':
        model = R2UNet(in_channels=3, out_channels=1).to(device)
    elif args.model == 'R2AttU_Net':
        model = ResAttU_Net(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("Model not supported")

    metric = BinaryJaccardIndex(threshold=0.5).to(device)
    criterion = utils.DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_model(model,
                    criterion, 
                    optimizer,
                    metric, 
                    train_loader, 
                    val_loader, 
                    epochs=50, 
                    device=device)
    
    # Проверяем точность
    dice, _ = utils.inference(val_loader, model, device=device)
    print(f"Dice score on val set: {dice:.4f}")

if __name__ == "__main__":
    main()