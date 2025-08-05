import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


from network import R2AttU_Net 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

def load_model(weights_path):
    """Загружает модель и веса, переводит в режим инференса."""
    model = R2AttU_Net(in_channels=3, out_channels=1)
    model.to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    
    model.eval()
    print("Модель успешно загружена.")
    return model

def predict(model, image_path, transforms):
    """Выполняет предсказание на одном изображении"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    augmented = transforms(image=image)
    input_tensor = augmented['image'].to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    probs = torch.sigmoid(logits)
    mask_np = probs.squeeze().cpu().numpy()
    
    binary_mask = (mask_np > 0.5).astype(np.uint8)
    
    return binary_mask

if __name__ == '__main__':
    
    WEIGHTS_PATH = "/Users/maks2/OneDrive/projects/retina-segmentation/src/models/best_weights.pth"
    IMAGE_TO_TEST = "Data/test/image/0.png"
    
    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.504, 0.275, 0.164], std=[0.345, 0.189, 0.107]),
        ToTensorV2(),
    ])
    
    model = load_model(WEIGHTS_PATH)
    
    predicted_mask = predict(model, IMAGE_TO_TEST, val_transforms)    
    original_image = cv2.imread(IMAGE_TO_TEST)
    original_image = cv2.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.show()