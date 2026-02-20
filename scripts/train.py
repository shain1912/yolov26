from ultralytics import YOLO
import os

def train_model():
    # 1. Load the YOLOv8 Nano model
    # Note: Using yolov8n.pt as requested
    model = YOLO("yolov8n.pt")

    # 2. Train the model
    # data: path to data.yaml
    # epochs: 50
    # imgsz: 640 (standard)
    # batch: 16
    # device: 0 (if GPU available, else 'cpu')
    results = model.train(
        data="data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        project="models",
        name="drone_defect_detection",
        exist_ok=True
    )

    print("Training Completed.")

    # 3. Validation
    metrics = model.val()
    print(f"Validation mAP50-95: {metrics.box.map}")
    print(f"Validation mAP50: {metrics.box.map50}")

if __name__ == "__main__":
    train_model()
