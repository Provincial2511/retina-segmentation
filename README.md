# Сегментация сосудов сетчатки глаза с помощью   U-Net, ResU-net, AttentionU-Net, R2AttU-Net

## Описание задачи

Автоматическая сегментация сосудов сетчатки является важной задачей в медицинской диагностике. Анализ морфологии сосудистой сетки (например, диаметра, извилистости) позволяет выявлять на ранних стадиях такие заболевания, как диабетическая ретинопатия и гипертония.

Цель данного проекта — разработать и обучить модель на архитектуре U-Net (и ее различных модификациях) для сегментирования сосудов на изображениях из датасета [Retina Blood Vessel Segmentation](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel/data).

## Структура проекта

-   `/notebooks`: Jupyter-ноутбуки с анализом данных и main ноутбук с полным циклом и моделями.
-   `/src`: Основной исходный код:
    -   `dataset.py`: Пайплайн данных.
    -   `models/`: Реализации архитектур U-Net / Attention U-Net / R2Attu-Unet.
    -   `train.py`: Скрипт для запуска обучения и валидации.
    -   `utils.py`: Вспомогательные функции.
    - `/inference`: Скрипт для предсказания на одном изображении

## Результаты

-   **Трекинг экспериментов:** Все запуски обучения и их результаты отслеживались с помощью **ClearML**.

В ходе экспериментов было установлено, что [Attention U-Net показывает прирост метрики Dice на 5% по сравнению с baseline U-Net].

| Модель | Val Dice Score | Val IoU |
| :--- | :---: | :---: |
| ResAttU-Net | **0.67** | **0.20** |
| ResU-Net | 0.66 | 0.21 |
| Attention U-Net | 0.64 | 0.23 |
| U-Net (Baseline) | 0.56 | 0.31 |
| R2AttU-Net | 0.37 | 0.67 |


**Пример работы модели:**

| Исходное изображение | Истинная маска          | Предсказание (ResAttU-Net) |
|----------------------|-------------------------|---------------------------------|
| ![](https://github.com/user-attachments/assets/0a21f7d9-a55c-40f4-b999-80a0509b27dc) | ![](https://github.com/user-attachments/assets/5ed69963-a4a0-4a8b-b121-fc03ddef8155) | ![](https://github.com/user-attachments/assets/04e59ee6-15c7-4e68-aca5-e5f457db0e01) |

## Как запустить

### 1. Установка зависимостей

Рекомендуется использовать виртуальное окружение.

```bash
git clone https://github.com/your_username/retina-segmentation-project.git
cd retina-segmentation
pip install -r requirements.txt
```
Запуск обучения
```bash
python src/train.py --model attention_unet --epochs 50 --batch_size 8
```
Чтобы использовать обученную модель, вы можете использовать веса лучшей модели (находятся по ссылке ниже)

https://disk.yandex.ru/d/vwIbmIHR5tA-3Q
