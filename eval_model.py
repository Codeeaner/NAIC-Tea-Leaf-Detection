from ultralytics import YOLO
from multiprocessing import freeze_support
import sys

MODEL = r"C:\yolo_runs\tealeaf_highres\weights\best.pt"
DATA = r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\tealeaf\data.yaml"


def run_eval():
    print('Evaluating model:', MODEL)
    model = YOLO(MODEL)
    results = model.val(data=DATA, imgsz=1280, batch=2, device='0')
    # results.box contains metrics
    box = results.box
    print('mAP@0.5:', getattr(box, 'map50', None))
    print('mAP@0.5:0.95:', getattr(box, 'map', None))
    print('Precision:', getattr(box, 'mp', None))
    print('Recall:', getattr(box, 'mr', None))
    return 0


if __name__ == '__main__':
    freeze_support()
    try:
        rc = run_eval()
        sys.exit(rc)
    except Exception as e:
        print('EVAL ERROR:', e)
        sys.exit(2)
