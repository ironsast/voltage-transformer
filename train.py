from ultralytics import YOLO


model = YOLO("yolo11s.pt")  


if __name__ == '__main__':
    results = model.train(
        data="data.yaml",
        epochs=2000,
        patience=500,
        batch=192,
        imgsz=640,
        device=[0, 1, 2] 
    )
