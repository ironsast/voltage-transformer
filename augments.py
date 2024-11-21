import cv2
import os
import albumentations as A
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# Функция для создания аугментированных изображений
def augment_images(input_folder, output_folder, augmentations_count=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Определяем набор аугментаций
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ])

    # Проходим по каждому изображению в папке
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Проверяем, что это изображение
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Загружаем изображение
        image = cv2.imread(input_path)
        if image is None:
            print(f"Ошибка при загрузке изображения: {filename}")
            continue
        
        # Генерируем несколько аугментированных версий изображения
        for i in range(augmentations_count):
            augmented = transform(image=image)['image']
            
            # Сохраняем аугментированное изображение
            aug_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            output_path = os.path.join(output_folder, aug_filename)
            cv2.imwrite(output_path, augmented)
            print(f"Аугментированное изображение сохранено: {output_path}")

# Параметры
input_folder = 'output_images'  # Папка с исходными изображениями
output_folder = 'augmented_images'  # Папка для сохранения аугментированных изображений

# Вызов функции
augment_images(input_folder, output_folder, augmentations_count=20)


