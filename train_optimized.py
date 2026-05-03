# ============================================================
#  Tea Leaf YOLO  –  Fine-Tuning Script  (Phase 1 of 2)
#  Base model : prefers a YOLOv26 checkpoint when available, otherwise
#               falls back to the strongest local baseline.
#  Target     : higher-resolution fine-tuning that stays within an RTX 4060
#               Laptop 8 GB VRAM budget.
#
#  Key fixes over the original training run
#  ─────────────────────────────────────────
#  1. Prefers a stronger YOLOv26-style starting checkpoint
#  2. 640 -> 768 input resolution for a sharper first pass
#  3. batch tuned for 8 GB VRAM with AMP enabled
#  4. copy_paste enabled and augmentation preserved
#  5. AdamW + cosine LR + early stopping for stable fine-tuning
#  6. workers capped for Windows pagefile stability
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
DATA_YAML  = r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\tealeaf\data.yaml"
LOCAL_BASE_MODEL = r"C:\yolo_runs\train6\weights\best.pt"


def setup_environment():
    """Configure the process for maximum GPU throughput on Windows."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # NOTE: expandable_segments is NOT supported on Windows (triggers a
        # warning and is silently ignored) – do not set it.
        # NOTE: cudnn.benchmark=True is INCOMPATIBLE with multi_scale=True.
        #   With multi_scale every batch is a different size, so cuDNN tries
        #   to JIT-compile a new PTX kernel variant for each unique shape.
        #   ptxas runs out of VRAM doing so → 'ptxas fatal: Memory allocation
        #   failure'.  Leave benchmark at its default (False).
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


def resolve_base_model() -> tuple[YOLO, str]:
    candidates = [LOCAL_BASE_MODEL]
    tried = []

    for candidate in candidates:
        if not candidate:
            continue

        candidate = str(candidate).strip()
        if candidate in tried:
            continue
        tried.append(candidate)

        candidate_path = Path(candidate)
        if candidate_path.exists() or candidate_path.suffix.lower() == ".pt" or "/" not in candidate and "\\" not in candidate:
            try:
                model = YOLO(candidate)
                logger.info(f"Loaded base model: {candidate}")
                return model, candidate
            except Exception as exc:
                logger.warning(f"Could not load base model '{candidate}': {exc}")

    raise FileNotFoundError(
        "No usable YOLO base model found. Tried: " + ", ".join(tried)
    )


def batch_for_vram(vram_gb: float, imgsz: int = 640) -> int:
    """
    Safe batch size with AMP + multi_scale=True.

    The RTX 4060 Laptop reports 7.996 GB (8188 MiB / 1024³), so the model
    has to stay conservative once resolution goes above 640.  This keeps the
    batch size small enough to avoid OOM while still using the GPU well.
    """
    if vram_gb <= 0:
        return 1
    scale = (640 / imgsz) ** 2   # 1.0 at imgsz=640, 0.25 at imgsz=1280
    if vram_gb >= 24:
        base = 16
    elif vram_gb >= 16:
        base = 12
    elif vram_gb >= 10:
        base = 6
    else:
        # < 10 GB (RTX 4060 Laptop = 7.996 GB): keep the batch small.
        base = 4
    return max(1, int(base * scale))


def main():
    setup_environment()
    device, vram_gb = get_device()

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    # ── Load the base model ───────────────────────────────────────────────────
    model, base_model_path = resolve_base_model()
    logger.info(f"Loaded base model : {base_model_path}")

    # ── Hardware-aware hyper-parameters ──────────────────────────────────────
    imgsz   = 1280          # Higher-res first pass; phase 2 goes to 1280
    batch   = 2     # Safe default for RTX 4060 Laptop 8 GB VRAM with AMP + multi_scale
    # Windows spawns a fresh Python process per worker, and each one reloads
    # all CUDA DLLs (~500 MB each).  8 workers × 500 MB exhausts the pagefile.
    # With cache='ram' the images are already in RAM so workers only run
    # augmentation – 2 is enough to keep the GPU fully saturated.
    workers = 2
    logger.info(f"Training config   : imgsz={imgsz}  batch={batch}  workers={workers}")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    results = model.train(
        data    = DATA_YAML,
        epochs  = 300,          # Up to 300; patience=50 will stop early
        imgsz   = imgsz,
        batch   = batch,
        workers = workers,
        device  = device,

        # ── Optimizer ────────────────────────────────────────────────────────
        # AdamW converges faster than SGD when fine-tuning a pre-trained model.
        optimizer       = 'AdamW',
        lr0             = 0.00008,  # Slightly lower for the stronger base model
                                    # prevents destroying existing weights
        lrf             = 0.01,     # Decay: end_lr = lr0 × lrf = 0.000001
        momentum        = 0.937,
        weight_decay    = 0.0005,

        # ── LR schedule ───────────────────────────────────────────────────────
        # The previous run used lr0=lrf=0.01 (completely flat – no decay at all).
        # cosine annealing drives the model into a sharper minimum at the end.
        cos_lr          = True,
        warmup_epochs   = 4,        # Short warmup – model already trained
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.01,

        # ── Early stopping & saving ──────────────────────────────────────────
        patience        = 50,       # Stop if val mAP doesn't improve for 50 epochs
        save            = True,
        save_period     = 25,       # Save checkpoint every 25 epochs
        val             = True,

        # ── Multi-scale training ─────────────────────────────────────────────
        # Disabled: Windows + CUDA JIT can still spike VRAM on variable shapes.
        # The higher fixed resolution gives the model the extra detail you asked for.
        multi_scale     = False,
        rect            = False,

        # ── Loss weights ─────────────────────────────────────────────────────
        box             = 7.5,
        cls             = 0.5,
        dfl             = 1.5,
        nbs             = 64,       # Nominal batch size for LR auto-scaling

        # ── Augmentation ─────────────────────────────────────────────────────
        # Preserved the augmentation that worked well, plus added copy_paste
        # (was 0.0 in every previous run – completely untapped).
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,

        degrees         = 25,
        translate       = 0.1,
        scale           = 0.5,
        shear           = 0.1,
        perspective     = 0.0,
        flipud          = 0.5,
        fliplr          = 0.5,

        mosaic          = 1.0,
        mixup           = 0.15,     # Slightly above original 0.1
        copy_paste      = 0.3,      # NEW – was 0.0 in all previous runs
        erasing         = 0.4,
        close_mosaic    = 20,       # Turn off mosaic 20 ep before end for stability
        # augmix runs 3 augmentation chains per image (CPU-heavy, starves GPU).
        # randaugment applies a single random chain – far cheaper on CPU.
        auto_augment    = 'randaugment',

        # ── Training stability ────────────────────────────────────────────────
        amp             = True,     # Mixed precision – halves VRAM, allows batch=16
        deterministic   = False,    # Faster (was True in train3, ~15–20% slower)
        single_cls      = False,
        overlap_mask    = True,
        # RAM cache: preloads all 1409 training images into RAM on epoch 1,
        # then every following epoch reads from RAM (~1.7 GB) instead of disk.
        # This is the single biggest fix for low GPU utilisation on Windows.
        cache           = 'ram',

        # ── Output / logging ─────────────────────────────────────────────────
        plots           = True,
        verbose         = True,
        exist_ok        = True,     # Will create train4, train5 … automatically
        seed            = 42,
        profile         = False,
        resume          = False,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 FINE-TUNE COMPLETE")
    logger.info(f"  mAP@0.5      (before) : 0.9380  →  (after) {results.box.map50:.4f}")
    logger.info(f"  mAP@0.5:0.95 (before) : 0.6908  →  (after) {results.box.map:.4f}")
    logger.info(f"  Precision              :  {results.box.mp:.4f}")
    logger.info(f"  Recall                 :  {results.box.mr:.4f}")
    logger.info("=" * 60)
    logger.info("Next step: run  finetune_highres.py  for Phase 2 (imgsz=1280)")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    return results


if __name__ == '__main__':
    freeze_support()            # Required for Windows multiprocessing
    try:
        results = main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as exc:
        logger.exception(f"Training failed: {exc}")
    finally:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
