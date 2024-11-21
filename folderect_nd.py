from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
import math

# Загрузка модели YOLOv8
model = YOLO(r"metrics_best.pt")

# Список цветов для различных классов
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

# Функция для вычисления расстояния между двумя точками
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Функция для вычисления угла между двумя точками относительно центра (в градусах)
def calculate_angle(center, point1, point2):
    delta1 = (point1[0] - center[0], point1[1] - center[1])
    delta2 = (point2[0] - center[0], point2[1] - center[1])
    
    dot_product = delta1[0] * delta2[0] + delta1[1] * delta2[1]
    magnitude1 = math.sqrt(delta1[0] ** 2 + delta1[1] ** 2)
    magnitude2 = math.sqrt(delta2[0] ** 2 + delta2[1] ** 2)
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))  # Ограничиваем значение, чтобы избежать ошибок с вычислением угла
    
    return math.degrees(angle)  # Возвращаем угол в градусах

# Функция для вычисления углового коэффициента линии
def calculate_slope(center, point):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    if dx == 0:  # Избегаем деления на ноль
        return float('inf')  # Бесконечный угловой коэффициент для вертикальной линии
    return dy / dx

# Функция для нахождения точки пересечения линии с эллипсом
def intersection_with_ellipse(center, angle, point, major_axis, minor_axis):
    # Переводим точку в полярные координаты относительно центра
    dx, dy = point[0] - center[0], point[1] - center[1]
    distance = math.sqrt(dx**2 + dy**2)
    
    # Находим угол линии относительно оси x
    line_angle = math.atan2(dy, dx)
    
    # Пропорция, по которой линия пересечет эллипс
    k = max(major_axis, minor_axis) / distance
    
    # Находим точку пересечения
    x_intersect = center[0] + k * distance * math.cos(line_angle)
    y_intersect = center[1] + k * distance * math.sin(line_angle)
    
    return int(x_intersect), int(y_intersect)

# Функция для обработки изображения
def process_image(image_path, output_folder, confidence_threshold=0.5):
    # Загрузка изображения
    image = cv2.imread(image_path)
    results = model(image)[0]
    
    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    confidences = results.boxes.conf.cpu().numpy()

    # Словарь для хранения координат объектов
    objects = {"scalestart": None, "center": None, "scaleend": None, "needle": None}
    colors_map = {}  # Словарь для хранения цветов рамок для каждого объекта

    # Рисование рамок и извлечение координат объектов
    for class_id, box, conf in zip(classes, boxes, confidences):
        if conf < confidence_threshold:  # Фильтрация объектов с низкой уверенностью
            continue
        
        class_name = classes_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]
        colors_map[class_name] = color  # Сохраняем цвет для данного класса

        if class_name in objects:
            # Записываем координаты центра объекта
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            if objects[class_name] is not None:
                print(f"На изображении {image_path} больше одного объекта типа {class_name}, пропускаем.")
                return  # Пропускаем изображение, если найдено больше одного объекта того же типа
            objects[class_name] = (center_x, center_y)
        
        # Рисование рамки и текста
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Проверка, все ли объекты распознаны
    if all(objects.values()):
        # Извлекаем координаты для рисования
        scalestart, center, scaleend, needle = objects["scalestart"], objects["center"], objects["scaleend"], objects["needle"]
        
        # Вычисляем длины полуосей для эллипса
        major_axis = calculate_distance(center, scalestart)  # От центра до scalestart
        minor_axis = calculate_distance(center, scaleend)   # От центра до scaleend

        # Вычисляем угол наклона эллипса
        angle = calculate_angle(center, scaleend, scalestart)

        # Рисуем эллипс
        cv2.ellipse(image, center, (int(major_axis), int(minor_axis)), angle, 0, 360, (0, 255, 255), 2)
        
        # Вычисление точек пересечения для линий
        scalestart_intersect = intersection_with_ellipse(center, angle, scalestart, major_axis, minor_axis)
        scaleend_intersect = intersection_with_ellipse(center, angle, scaleend, major_axis, minor_axis)
        needle_intersect = intersection_with_ellipse(center, angle, needle, major_axis, minor_axis)

        # Рисуем линии до пересечения с эллипсом
        cv2.line(image, center, scalestart_intersect, colors_map["scalestart"], 2)
        cv2.line(image, center, scaleend_intersect, colors_map["scaleend"], 2)
        cv2.line(image, center, needle_intersect, colors_map["needle"], 2)

        # Вычисление углов между точками (в градусах)
        scalestart_angle = calculate_angle(center, scalestart, needle)
        scaleend_angle = calculate_angle(center, scaleend, needle)

        # Отображаем углы красным цветом
        cv2.putText(image, f"Start - Needle Angle: {scalestart_angle:.2f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image, f"End - Needle Angle: {scaleend_angle:.2f}°", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Сохранение обработанного изображения
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_yolo{os.path.splitext(image_name)[1]}")
        cv2.imwrite(new_image_path, image)
        print(f"Изображение с распознанными объектами и эллипсом сохранено: {new_image_path}")
    else:
        print(f"Изображение {image_path} не содержит все необходимые объекты, не сохранено.")

# Функция для обработки всех изображений в папке
def process_images_in_folder(input_folder, output_folder, confidence_threshold=0.5):
    # Создание выходной папки, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Поиск всех изображений в папке
    image_paths = glob.glob(os.path.join(input_folder, '*.png')) + glob.glob(os.path.join(input_folder, '*.jpg'))
    
    # Обработка каждого изображения
    for image_path in image_paths:
        process_image(image_path, output_folder, confidence_threshold)

# Пример вызова функции
input_folder = 'output_images'  # Папка с изображениями для обработки
output_folder = 'detected_nd'  # Папка для сохранения обработанных изображений
process_images_in_folder(input_folder, output_folder, confidence_threshold=0.5)
