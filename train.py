from ultralytics import YOLO


model = YOLO("yolo11n.pt")  


if __name__ == '__main__':
    results = model.train(
        data="data.yaml",
        epochs=1000,
        patience=100,
        batch=64,
        imgsz=640,
        device=0 
    )