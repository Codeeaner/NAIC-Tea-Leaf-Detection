# ============================================================
#  Tea Leaf YOLO  –  High-Resolution Polish  (Phase 2 of 2)
#
#  Run AFTER train_optimized.py (Phase 1) completes.
#
#  Phase 2 goal: push mAP@0.5:0.95 further by training at
#  imgsz=1280. Larger resolution forces the model to learn
#  tighter bounding boxes and improves detection of small /
#  partially-occluded leaves – the main driver of mAP50-95.
#
#  RTX 4060 Laptop (8 GB VRAM) + AMP:
#    imgsz=1280  batch=2  → safer default on Windows
#    imgsz=1280  batch=4  → only if you have headroom after Phase 1
#
#  Usage:
#    python finetune_highres.py
#
#  The script auto-discovers the newest Phase 1 best.pt under
#  runs/detect/.  Point BASE_MODEL manually if needed.
# ============================================================

from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import os
import gc
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
# Phase 1 output landed in tealeaf/runs/detect (relative), not root runs/detect.
RUNS_DIR  = Path(r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\tealeaf\runs\detect")
DATA_YAML = r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\tealeaf\data.yaml"
# Output goes outside OneDrive to avoid file-handle conflicts during checkpoint saves.
OUTPUT_DIR = r"C:\yolo_runs"

# Set to a specific path to skip auto-detection:
BASE_MODEL_OVERRIDE = None   # e.g. r"C:\yolo_runs\tealeaf_highres\weights\best.pt"


def find_latest_best(runs_dir: Path) -> Path:
    """
    Return the best.pt from the most-recently modified training run
    under runs_dir, so Phase 2 automatically picks up Phase 1's output.
    """
    candidates = sorted(
        runs_dir.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt weights found under {runs_dir}. "
            "Run train_optimized.py (Phase 1) first."
        )
    return candidates[0]


def setup_environment():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # expandable_segments is not supported on Windows (silently ignored).
        # cudnn.benchmark=False is required with multi_scale=True to avoid
        # ptxas kernel-compilation OOM on variable input shapes.
        torch.backends.cudnn.benchmark = False
    gc.collect()
    torch.set_num_threads(4)


def get_device():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024 ** 3
        logger.info(f"GPU : {props.name}  –  {vram:.1f} GB VRAM")
        return '0', vram
    logger.warning("No CUDA GPU detected – training on CPU (very slow!)")
    return 'cpu', 0.0


def main():
    setup_environment()
    device, vram_gb = get_device()

    # ── Find Phase 1 best model ───────────────────────────────────────────────
    if BASE_MODEL_OVERRIDE:
        base_model_path = Path(BASE_MODEL_OVERRIDE)
    else:
        base_model_path = find_latest_best(RUNS_DIR)

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    logger.info(f"Phase 2 base model : {base_model_path}")

    model = YOLO(str(base_model_path))

    # ── VRAM-aware batch size for imgsz=1280 ─────────────────────────────────
    # AMP halves memory vs FP32.  At 1280² the feature maps are 4× larger.
    if vram_gb >= 12:
        batch = 4
    elif vram_gb >= 8:
        batch = 2   # Safe default for RTX 4060 Laptop 8 GB
    else:
        batch = 1

    imgsz   = 1280
    workers = 2   # Windows pagefile limit: keep low, RAM cache handles the rest
    logger.info(f"Phase 2 config : imgsz={imgsz}  batch={batch}  workers={workers}")

    # ── Fine-tune at high resolution ─────────────────────────────────────────
    results = model.train(
        data    = DATA_YAML,
        epochs  = 100,          # Short – model is already strong from Phase 1
        imgsz   = imgsz,
        batch   = batch,
        workers = workers,
        device  = device,

        # ── Optimizer ────────────────────────────────────────────────────────
        # Phase 1 early-stopped at epoch 16: the model converges very fast on
        # this 705-image dataset.  Use an even lower lr0 for Phase 2 so the
        # high-res pass makes small, precise adjustments without overshooting.
        optimizer       = 'AdamW',
        lr0             = 0.000005,  # Low enough to avoid overshooting
        lrf             = 0.01,      # end_lr = 5e-8
        momentum        = 0.937,
        weight_decay    = 0.0005,

        # ── LR schedule ──────────────────────────────────────────────────────
        cos_lr          = True,
        warmup_epochs   = 3,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.002,

        # ── Early stopping & saving ──────────────────────────────────────────
        # Phase 1 converges quickly, so stop once the metric plateaus.
        patience        = 20,
        save            = True,
        save_period     = 10,
        val             = True,

        # ── Multi-scale training ─────────────────────────────────────────────
        # Disabled: same Windows/CUDA JIT risk as Phase 1.
        # At imgsz=1280 the fixed-resolution benefit is already large enough.
        multi_scale     = False,
        rect            = False,

        # ── Loss weights ─────────────────────────────────────────────────────
        box             = 7.5,
        cls             = 0.5,
        dfl             = 1.5,
        nbs             = 64,

        # ── Augmentation – lighter at high-res to reduce VRAM pressure ───────
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,

        degrees         = 15,       # Reduced vs Phase 1 (model already flexible)
        translate       = 0.1,
        scale           = 0.4,
        shear           = 0.05,
        perspective     = 0.0,
        flipud          = 0.5,
        fliplr          = 0.5,

        mosaic          = 0.5,      # Half mosaic at high-res (memory budget)
        mixup           = 0.1,
        copy_paste      = 0.2,
        erasing         = 0.3,
        close_mosaic    = 10,
        auto_augment    = 'randaugment',  # lighter than augmix, less CPU pressure

        # ── Training stability ────────────────────────────────────────────────
        amp             = True,     # Critical at 1280 – without it, likely OOM
        deterministic   = False,
        single_cls      = False,
        overlap_mask    = True,
        # At 1280 fewer images fit per batch so RAM cache is even more valuable.
        cache           = 'ram',

        # ── Output / logging ─────────────────────────────────────────────────
        # project outside OneDrive avoids the file-handle conflict during saves.
        project         = OUTPUT_DIR,
        name            = 'tealeaf_highres',
        plots           = True,
        verbose         = True,
        exist_ok        = True,
        seed            = 42,
        profile         = False,
        resume          = False,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 HIGH-RES POLISH COMPLETE")
    logger.info(f"  mAP@0.5      : {results.box.map50:.4f}")
    logger.info(f"  mAP@0.5:0.95 : {results.box.map:.4f}")
    logger.info(f"  Precision    : {results.box.mp:.4f}")
    logger.info(f"  Recall       : {results.box.mr:.4f}")
    logger.info("=" * 60)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    return results


if __name__ == '__main__':
    freeze_support()
    try:
        results = main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as exc:
        logger.exception(f"Phase 2 training failed: {exc}")
    finally:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
