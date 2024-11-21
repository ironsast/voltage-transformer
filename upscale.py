import cv2
from PIL import Image, ImageEnhance, ImageFilter
import os


# Функция для увеличения разрешения до Full HD и улучшения качества изображения
def upscale_to_fullhd_and_enhance(input_folder, output_folder, target_size=(1920, 1080)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Проходим по каждому изображению в папке
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Проверяем, что это изображение
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Загружаем изображение с помощью OpenCV
        image = cv2.imread(input_path)
        if image is None:
            print(f"Ошибка при загрузке изображения: {filename}")
            continue

        # Увеличиваем разрешение изображения до Full HD
        upscaled_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

        # Преобразуем изображение в формат PIL для дальнейшего улучшения
        upscaled_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))

        # Улучшаем резкость, контраст и яркость изображения
        upscaled_image = upscaled_image.filter(ImageFilter.DETAIL)  # Повышение детализации
        enhancer = ImageEnhance.Sharpness(upscaled_image)
        upscaled_image = enhancer.enhance(2.0)  # Повышение резкости
        enhancer = ImageEnhance.Contrast(upscaled_image)
        upscaled_image = enhancer.enhance(1.5)  # Повышение контраста
        enhancer = ImageEnhance.Brightness(upscaled_image)
        upscaled_image = enhancer.enhance(1.2)  # Легкое повышение яркости

        # Сохранение улучшенного изображения
        output_path = os.path.join(output_folder, filename)
        upscaled_image.save(output_path)
        print(f"Изображение {filename} улучшено и сохранено в {output_path}")

# Параметры
input_folder = 'augmented_images'  # Папка с исходными изображениями
output_folder = 'upscaled_images'  # Папка для сохранения улучшенных изображений

# Вызов функции
upscale_to_fullhd_and_enhance(input_folder, output_folder, target_size=(1280, 1280))
