from ultralytics import YOLO
import cv2
import numpy as np
import os

# Загрузка модели YOLOv8
model = YOLO(r"gauge_best.pt")

# Функция для обработки видео и сохранения распознанных областей с объектами
def process_video_and_save_objects(video_path, output_folder, confidence_threshold=0.5, interval_seconds=1):
    # Открытие видеопотока
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка при открытии видео.")
        return

    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров

    # Создание папки для сохранения объектов, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    saved_count = 0
    while True:
        # Чтение кадра
        ret, frame = cap.read()
        if not ret:
            break  # Если кадры закончились, выходим

        # Если текущий кадр не соответствует интервалу, пропускаем его
        if frame_count % int(fps * interval_seconds) != 0:
            frame_count += 1
            continue

        # Детекция объектов на текущем кадре
        results = model(frame)[0]

        # Получение результатов
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
        confidences = results.boxes.conf.cpu().numpy()

        # Сохранение каждого объекта как отдельного изображения
        for i, (class_id, box, conf) in enumerate(zip(classes, boxes, confidences)):
            if conf < confidence_threshold:  # Фильтрация объектов с низкой уверенностью
                continue

            # Вырезаем область, соответствующую объекту
            x1, y1, x2, y2 = box
            object_image = frame[y1:y2, x1:x2]  # Вырезка области объекта

            # Имя класса и путь сохранения
            class_name = classes_names[int(class_id)]
            output_class_folder = os.path.join(output_folder, class_name)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

            # Сохранение области с объектом
            object_filename = os.path.join(output_class_folder, f"frame_{frame_count:04d}_obj_{i}.jpg")
            cv2.imwrite(object_filename, object_image)
            saved_count += 1

        frame_count += 1
        print(f"Processed frame {frame_count}")

    # Освобождение ресурсов
    cap.release()
    print(f"Обработано {frame_count} кадров. Сохранено {saved_count} объектов в папку {output_folder}")

# Вызов функции для обработки видео
process_video_and_save_objects('video.mp4', 'output_images', confidence_threshold=0.55, interval_seconds=1)
