from ultralytics import YOLO
import os
import torch


def load_base_model():
    candidates = ["yolo26n.pt", "yolo11n.pt", "C:/Users/amber/OneDrive/Documents/GitHub/Monash--AI/runs/detect/train3/weights/best.pt"]

    for candidate in candidates:
        try:
            model = YOLO(candidate)
            print(f"Loaded base model: {candidate}")
            return model
        except Exception as exc:
            print(f"Could not load {candidate}: {exc}")

    raise RuntimeError("No usable base model could be loaded")

def retrain_model():
    """Retrain the model with a stronger pretrained checkpoint."""
    
    # Check if data exists
    data_path = "data.yaml"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return False
    
    print("Starting model retraining with a YOLOv26-first configuration...")
    print("This will create a new model compatible with the current ultralytics version.")
    
    try:
        model = load_base_model()
        
        # Train the model
        results = model.train(
            data=data_path,
            epochs=100,
            imgsz=768,
            batch=2,
            device='0' if torch.cuda.is_available() else 'cpu',
            workers=2,
            patience=20,
            project='runs/detect',
            name='retrain_v8',
            exist_ok=True,
            verbose=True
        )
        
        print("Training completed!")
        print(f"New model saved at: runs/detect/retrain_v8/weights/best.pt")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    retrain_model()