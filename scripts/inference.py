import cv2
from ultralytics import YOLO
import os
import json

def run_inference(image_path, model_path="models/drone_defect_detection/weights/best.pt"):
    # 1. Load model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using yolov8n.pt for demo.")
        model_path = "yolov8n.pt"
    
    model = YOLO(model_path)
    
    # 2. Inference
    results = model.predict(image_path, save=False, conf=0.25)
    
    defect_data = []
    
    # 3. Process results
    for result in results:
        img_name = os.path.basename(result.path)
        img_bgr = result.orig_img.copy()
        
        for box in result.boxes:
            # Extract data
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            
            # BBox coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            
            # BBox Center and Size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Store data
            defect_data.append({
                "이미지 파일명": img_name,
                "하자 클래스": label,
                "위치 좌표(X, Y)": (round(center_x, 2), round(center_y, 2)),
                "가로": round(width, 2),
                "세로": round(height, 2),
                "Confidence Score": round(conf, 4)
            })
            
            # 4. Visualization
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{label} {conf:.2f}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        save_path = "outputs/result_img.jpg"
        cv2.imwrite(save_path, img_bgr)
        print(f"Reasoned visualization saved to {save_path}")

    return defect_data

if __name__ == "__main__":
    # Test with a placeholder if needed
    test_img = "data/images/val/sample.jpg"
    if os.path.exists(test_img):
        data = run_inference(test_img)
        print(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        print(f"Sample image not found at {test_img}")
