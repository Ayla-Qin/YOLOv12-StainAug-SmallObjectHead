from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import tifffile
import os
import random
import shutil
import csv
from tqdm import tqdm


# ===============================
# 0. Configuration
# ===============================

AUG_PROB = 0.2

ROOT_DIR = "./"

FOLD_ROOT = os.path.join(ROOT_DIR, "midog_folds")
TMP_ROOT = os.path.join(ROOT_DIR, "tmp_train_aug")
PROJECT_DIR = os.path.join(ROOT_DIR, "mitosis_yolov12_5fold")
RESULT_CSV_PATH = os.path.join(PROJECT_DIR, "fold_results.csv")

MODEL_CFG = "./yolov12m-changed-128_16.yaml"
MODEL_WEIGHTS = "./yolov12m.pt"

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Create fold YAML
# ===============================

def create_fold_yaml(fold_root, fold_idx, save_path):

    import yaml

    folds = [f"fold_{i}" for i in range(1, 6)]

    val_fold = f"fold_{fold_idx}"
    train_folds = [f for f in folds if f != val_fold]

    # Use four folds for training
    train_paths = [
        os.path.join(fold_root, f, "images")
        for f in train_folds
    ]

    # Use current fold for validation
    val_path = os.path.join(fold_root, val_fold, "images")

    data = {
        "train": train_paths,
        "val": val_path,
        "nc": 1,
        "names": ["mitotic_figure"]
    }

    with open(save_path, "w") as f:
        yaml.dump(data, f)

    return save_path


# ===============================
# Read metrics from CSV
# ===============================

def get_latest_metrics_from_csv(trainer):

    results_csv = os.path.join(trainer.save_dir, "results.csv")

    if not os.path.exists(results_csv):
        return None, None

    df = pd.read_csv(results_csv)

    if len(df) == 0:
        return None, None

    precision = df.iloc[-1]["metrics/precision(B)"]
    recall = df.iloc[-1]["metrics/recall(B)"]

    return precision, recall


# ===============================
# 1. H&E stain perturbation
# ===============================

def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):

    HERef = np.array([
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581]
    ])

    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape

    img = img.reshape((-1, 3)).astype(np.float32)
    img = np.clip(img, 1, 255)

    OD = -np.log(img / Io)

    mask = ~np.any(OD < beta, axis=1)
    ODhat = OD[mask]

    if ODhat.shape[0] < 50:

        img = img.reshape((h, w, 3)).astype(np.uint8)

        C2 = np.zeros((2, h * w), dtype=np.float32)

        return img, img, img, HERef, C2

    cov = np.cov(ODhat.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot([
        [np.cos(minPhi)],
        [np.sin(minPhi)]
    ])

    vMax = eigvecs[:, 1:3].dot([
        [np.cos(maxPhi)],
        [np.sin(maxPhi)]
    ])

    if vMin[0] > vMax[0]:
        HE = np.hstack((vMin, vMax))
    else:
        HE = np.hstack((vMax, vMin))

    Y = OD.T

    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    maxC = np.array([
        np.percentile(C[0, :], 99),
        np.percentile(C[1, :], 99)
    ])

    tmp = maxC / maxCRef

    C2 = C / tmp[:, None]
    C2 = np.clip(C2, 0, 5)

    Inorm = Io * np.exp(-HERef.dot(C2))
    Inorm = np.clip(Inorm, 0, 255)
    Inorm = Inorm.T.reshape((h, w, 3)).astype(np.uint8)

    H = Io * np.exp(-np.outer(HERef[:, 0], C2[0, :]))
    H = np.clip(H, 0, 255)
    H = H.T.reshape((h, w, 3)).astype(np.uint8)

    E = Io * np.exp(-np.outer(HERef[:, 1], C2[1, :]))
    E = np.clip(E, 0, 255)
    E = E.T.reshape((h, w, 3)).astype(np.uint8)

    # Save outputs in TIFF format
    if saveFile:

        tifffile.imwrite(saveFile + ".tiff", Inorm)
        tifffile.imwrite(saveFile + "_H.tiff", H)
        tifffile.imwrite(saveFile + "_E.tiff", E)

    return Inorm, H, E, HERef, C2


def stain_perturb(img):

    img = img.astype(np.uint8)

    try:
        Inorm, H, E, HERef, C2 = normalizeStaining(img)

    except Exception:
        return img

    h_scale = random.uniform(0.94, 1.06)
    e_scale = random.uniform(0.94, 1.06)

    h_shift = random.uniform(-0.03, 0.03)
    e_shift = random.uniform(-0.03, 0.03)

    C2_new = C2.copy()

    C2_new[0, :] = C2[0, :] * h_scale + h_shift
    C2_new[1, :] = C2[1, :] * e_scale + e_shift

    Io = 240

    img_new = Io * np.exp(-HERef.dot(C2_new))

    img_new = np.clip(img_new, 0, 255)

    img_new = img_new.T.reshape(
        img.shape[0],
        img.shape[1],
        3
    ).astype(np.uint8)

    return img_new


# ===============================
# 2. Create temporary TIFF dataset
# ===============================

def create_augmented_trainset(
    orig_yaml,
    tmp_train_dir,
    tmp_label_dir,
    tmp_yaml,
    aug_prob=0.5
):

    import yaml

    with open(orig_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    orig_train = cfg.get("train")

    # Convert to list format
    if not isinstance(orig_train, list):
        orig_train = [orig_train]

    orig_val = cfg.get("val")

    nc = cfg.get("nc", 1)
    names = cfg.get("names", ["mitotic_figure"])

    os.makedirs(tmp_train_dir, exist_ok=True)
    os.makedirs(tmp_label_dir, exist_ok=True)

    if isinstance(orig_train, list):

        img_paths = []

        for p in orig_train:

            img_paths += [
                os.path.join(p, f)
                for f in os.listdir(p)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                )
            ]

    else:

        img_paths = [
            os.path.join(orig_train, f)
            for f in os.listdir(orig_train)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            )
        ]

    for i, img_path in enumerate(
        tqdm(img_paths, desc="Augmenting images")
    ):

        img_name = os.path.basename(img_path)

        base = os.path.splitext(img_name)[0]

        # Infer label path directly from image path
        lbl_path = img_path.replace("/images/", "/labels/")
        lbl_path = os.path.splitext(lbl_path)[0] + ".txt"

        # Skip images without labels
        if not os.path.exists(lbl_path):
            continue

        if img_name.lower().endswith((".tif", ".tiff")):

            img = tifffile.imread(img_path)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            elif img.shape[2] > 3:
                img = img[:, :, :3]

        else:

            img = cv2.imread(img_path)

        if img is None:
            continue

        do_aug = random.random() < aug_prob

        img_out = stain_perturb(img) if do_aug else img

        out_img_name = f"{base}.tiff"

        out_img_path = os.path.join(
            tmp_train_dir,
            out_img_name
        )

        tifffile.imwrite(
            out_img_path,
            img_out,
            compression=None
        )

        shutil.copy(
            lbl_path,
            os.path.join(tmp_label_dir, base + ".txt")
        )

        # Progress logging
        if i % 20000 == 0:
            tqdm.write(
                f"Processed {i}/{len(img_paths)} images"
            )

    tmp_cfg = {
        "train": os.path.abspath(tmp_train_dir),
        "val": orig_val,
        "nc": nc,
        "names": names
    }

    with open(tmp_yaml, "w") as f:
        yaml.safe_dump(tmp_cfg, f)

    print(f"Temporary training dataset created: {tmp_train_dir}")

    return tmp_yaml


# ===============================
# 3. Metric logging
# ===============================

def append_f1_to_results_csv(trainer):

    results_csv = os.path.join(
        trainer.save_dir,
        "results.csv"
    )

    if not os.path.exists(results_csv):
        return

    precision, recall = get_latest_metrics_from_csv(
        trainer
    )

    if precision is None or recall is None:
        return

    if precision + recall == 0:
        return

    f1 = 2 * precision * recall / (precision + recall)

    with open(results_csv, "r", newline="") as f:
        rows = list(csv.reader(f))

    if "metrics/F1(B)" not in rows[0]:
        rows[0].append("metrics/F1(B)")

    if len(rows) > 1:

        if len(rows[-1]) < len(rows[0]):
            rows[-1].append(f"{f1:.6f}")

    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(
        f"Epoch {trainer.epoch + 1}: "
        f"F1={f1:.6f} written to results.csv"
    )


def on_fit_epoch_end(trainer):

    # Get F1 score
    try:
        f1 = float(trainer.metrics.box.f1.mean())

    except Exception:
        return

    # Save F1 log
    f1_csv = os.path.join(
        trainer.save_dir,
        "f1_log.csv"
    )

    with open(f1_csv, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            trainer.epoch,
            f1
        ])

    # Initialize best F1
    if not hasattr(trainer, "best_f1"):
        trainer.best_f1 = -1

    # Save best model
    if f1 > trainer.best_f1:

        trainer.best_f1 = f1

        save_path = os.path.join(
            trainer.save_dir,
            "weights",
            "best_f1.pt"
        )

        trainer.model.save(save_path)

        print(
            f"Epoch {trainer.epoch} | "
            f"Best F1={f1:.6f} saved"
        )


# ===============================
# 4. Five-fold training and validation
# ===============================

all_metrics = []

os.makedirs(PROJECT_DIR, exist_ok=True)

# Initialize CSV file
with open(RESULT_CSV_PATH, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "fold",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95"
    ])


for fold in range(1, 6):

    print(f"\nStarting Fold {fold}")

    # Create fold YAML file
    yaml_path = f"./temp_fold_{fold}.yaml"

    create_fold_yaml(
        FOLD_ROOT,
        fold,
        yaml_path
    )

    # Create independent augmentation directory
    tmp_train_dir = os.path.join(
        TMP_ROOT,
        f"fold_{fold}",
        "images/train"
    )

    tmp_label_dir = os.path.join(
        TMP_ROOT,
        f"fold_{fold}",
        "labels/train"
    )

    tmp_yaml_path = os.path.join(
        TMP_ROOT,
        f"fold_{fold}.yaml"
    )

    # Remove current fold temporary directory
    shutil.rmtree(
        os.path.join(TMP_ROOT, f"fold_{fold}"),
        ignore_errors=True
    )

    # Generate augmented dataset
    tmp_yaml = create_augmented_trainset(
        yaml_path,
        tmp_train_dir,
        tmp_label_dir,
        tmp_yaml_path,
        aug_prob=AUG_PROB
    )

    # Train model
    model = YOLO(MODEL_CFG).load(MODEL_WEIGHTS)

    model.add_callback(
        "on_fit_epoch_end",
        on_fit_epoch_end
    )

    model.train(

        data=tmp_yaml_path,

        epochs=50,
        imgsz=640,
        batch=64,

        device=DEVICE_STR,

        optimizer="SGD",

        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,

        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        flipud=0.5,
        fliplr=0.5,

        degrees=10,

        translate=0.0,
        scale=0.0,

        mosaic=0.1,

        mixup=0.0,
        cutmix=0.0,

        shear=0.0,
        perspective=0.0,

        project=PROJECT_DIR,

        name=f"midog_fold_{fold}_macenko",

        save_period=1
    )

    # Validation using current fold
    best_f1_path = os.path.join(
        PROJECT_DIR,
        f"midog_fold_{fold}_macenko",
        "weights",
        "best_f1.pt"
    )

    if not os.path.exists(best_f1_path):

        print(
            "best_f1.pt not found, "
            "using best.pt instead"
        )

        best_f1_path = best_f1_path.replace(
            "best_f1.pt",
            "best.pt"
        )

    model = YOLO(best_f1_path)

    val_results = model.val(

        data=tmp_yaml_path,

        split="val",

        imgsz=640,

        batch=64,

        conf=0.1,

        iou=0.3
    )

    precision, recall, mAP50, mAP50_95 = (
        val_results.mean_results()
    )

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-6)
    )

    print(
        f"Fold {fold} -> "
        f"P:{precision:.4f} "
        f"R:{recall:.4f} "
        f"F1:{f1:.4f} "
        f"mAP50:{mAP50:.4f}"
    )

    result_dict = {

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "map50": mAP50,

        "map50_95": mAP50_95
    }

    all_metrics.append(result_dict)

    # Write results to CSV
    with open(RESULT_CSV_PATH, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            fold,
            precision,
            recall,
            f1,
            mAP50,
            mAP50_95
        ])

    # Remove temporary YAML file
    os.remove(yaml_path)


# ===============================
# 5. Final statistics
# ===============================

print("\nFinal 5-Fold Results (Mean ± Std)")

with open(RESULT_CSV_PATH, "a", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([])
    writer.writerow(["Metric", "Mean", "Std"])

    for key in all_metrics[0].keys():

        values = [m[key] for m in all_metrics]

        mean = np.mean(values)
        std = np.std(values)

        writer.writerow([
            key,
            mean,
            std
        ])

        print(
            f"{key}: "
            f"{mean:.4f} ± {std:.4f}"
        )
