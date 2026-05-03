from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import os
import psutil # To get CPU count for workers

def main():
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

    # Set environment variable for memory management
    # expandable_segments is not reliable on Windows; avoid setting it here.
    # Use a conservative cuDNN benchmark for fixed-size training to improve throughput.
    try:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled for fixed-size inputs")
    except Exception:
        pass

    # Check GPU availability
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = '0'  # Use GPU 0
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Load a stronger pretrained checkpoint first, then fall back to the
    # local baseline if the requested weights are not available.
    last_pt = r"C:\yolo_runs\train6\weights\last.pt"
    resume_flag = False
    if os.path.exists(last_pt):
        try:
            model = YOLO(last_pt)
            resume_flag = True
            print(f"Loaded checkpoint: {last_pt} (will resume)")
        except Exception as e:
            print(f"Found last.pt but failed to load it: {e}. Falling back to pretrained families.")
            model = None
            resume_flag = False
    else:
        model = None

    if model is None:
        try:
            model = YOLO("yolo26s.pt")
            print("Loaded yolo26s.pt")
        except Exception as e:
            print(f"Could not load yolo26s.pt: {e}. Trying yolo26n.pt and the local baseline.")
            try:
                model = YOLO("yolo26n.pt")
                print("Loaded yolo26n.pt")
            except Exception as e:
                print(f"Could not load yolo26n.pt: {e}. Trying local train3 best.pt.")
                try:
                    model = YOLO(r"C:\\Users\\amber\\OneDrive\\Documents\\GitHub\\Monash--AI\\runs\\detect\\train3\\weights\\best.pt")
                    print("Loaded local train3 best.pt")
                except Exception as e:
                    print(f"Could not load local train3 best.pt: {e}. Exiting.")
                    return # Exit if no model can be loaded

    # Determine optimal workers based on CPU count
    # Using half of the available CPU cores is a good starting point,
    # but cap to 2 on Windows to avoid heavy multiprocessing overhead
    num_workers = 4
    if num_workers == 0:
        print("Warning: Could not determine CPU count, defaulting workers to 0. This may slow down data loading.")
    else:
        print(f"Using {num_workers} workers for data loading.")
    # Reduce Python thread usage slightly to reserve CPU for IO/augmentation
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

    # Train the model
    results = model.train(
        data=r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\tealeaf\data.yaml",
        epochs=2000,
        imgsz=768,
        batch=8,                # reduce CPU/GPU pressure on Windows
        workers=num_workers,    # more CPU workers
        device=device,
        patience=50,
        save=True,
        project=r"C:\yolo_runs",
        name="train6",
        resume=resume_flag,
        save_period=10,
        exist_ok=True,
        plots=False,           
        cache=False,
        degrees=25,
        translate=0.1,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        shear=0.1,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        lr0=0.00008,
        lrf=0.01,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=4,
        amp=True,
        deterministic=False,
    )

    return results

if __name__ == '__main__':
    freeze_support()
    main()
