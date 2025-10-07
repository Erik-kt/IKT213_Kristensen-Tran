from __future__ import annotations
import os, re, time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from matchers import (
    Config, read_gray, is_uia_folder, ensure_dir,
    match_orb, match_sift_flann_ransac
)

def decide_thresholded(score: int, thr: int) -> int:
    return 1 if score >= thr else 0

def eval_pair(img1_path: str, img2_path: str, outdir: str, pair_name: str, cfg: Config):
    ensure_dir(outdir)
    g1 = read_gray(img1_path)
    g2 = read_gray(img2_path)

    mode = "gray" if is_uia_folder(pair_name) else "binary"

    t_orb = 0.0
    t_sift = 0.0

    # ORB
    t0 = time.perf_counter()
    orb_score, kpa, kpb, ma, vis_orb = match_orb(g1, g2, cfg=cfg, mode=mode)
    t_orb = time.perf_counter() - t0

    # SIFT (+ homografi for gray)
    t0 = time.perf_counter()
    sift_score, kpsa, kpsb, mb, vis_sift = match_sift_flann_ransac(g1, g2, cfg=cfg, mode=mode)
    t_sift = time.perf_counter() - t0

    orb_thr  = cfg.orb_thr_gray  if mode == "gray" else cfg.orb_thr_binary
    sift_thr = cfg.sift_thr_gray if mode == "gray" else cfg.sift_thr

    orb_pred  = decide_thresholded(orb_score,  orb_thr)
    sift_pred = decide_thresholded(sift_score, sift_thr)

    if cfg.save_vis and vis_orb is not None:
        cv.imwrite(os.path.join(outdir, f"{pair_name}_ORB_{mode}_{orb_score}.png"), vis_orb)
    if cfg.save_vis and vis_sift is not None:
        cv.imwrite(os.path.join(outdir, f"{pair_name}_SIFT_{mode}_{sift_score}.png"), vis_sift)

    return {
        "gt": 1 if "same" in pair_name.lower() else 0,
        "orb":  {"score": orb_score,  "pred": orb_pred,  "time_s": t_orb,  "mode": mode, "thr": orb_thr},
        "sift": {"score": sift_score, "pred": sift_pred, "time_s": t_sift, "thr": sift_thr},
    }

def process_dataset(dataset_dir: str, outdir_base: str, cfg: Config):
    ensure_dir(outdir_base)
    y_true, y_orb, y_sift = [], [], []
    t_orb, t_sift = [], []

    for folder in sorted(os.listdir(dataset_dir)):
        fpath = os.path.join(dataset_dir, folder)
        if not os.path.isdir(fpath):
            continue
        imgs = [p for p in sorted(os.listdir(fpath))
                if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        if len(imgs) != 2:
            print(f"Skip {folder} (need exactly 2 images, got {len(imgs)})")
            continue

        img1, img2 = os.path.join(fpath, imgs[0]), os.path.join(fpath, imgs[1])
        res = eval_pair(img1, img2, os.path.join(outdir_base, folder), folder, cfg=cfg)

        y_true.append(res["gt"])
        y_orb.append(res["orb"]["pred"])
        y_sift.append(res["sift"]["pred"])
        t_orb.append(res["orb"]["time_s"])
        t_sift.append(res["sift"]["time_s"])

        print(f"{folder:20s} | GT={res['gt']} | "
              f"ORB({res['orb']['mode']}): score={res['orb']['score']:3d} thr={res['orb']['thr']:2d} "
              f"pred={res['orb']['pred']} time={res['orb']['time_s']*1000:.1f}ms | "
              f"SIFT: score={res['sift']['score']:3d} thr={res['sift']['thr']:2d} "
              f"pred={res['sift']['pred']} time={res['sift']['time_s']*1000:.1f}ms")

    # Confusion matrices
    def save_cm(name: str, preds: list):
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Different (0)", "Same (1)"])
        plt.figure(figsize=(5, 4))
        disp.plot(cmap="Blues", values_format="d")
        plt.title(name)
        plt.tight_layout()
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
        out = os.path.join(outdir_base, f"cm_{safe}.png")
        plt.savefig(out, dpi=160)
        plt.close()
        acc = (cm.trace() / cm.sum()) if cm.sum() else 0.0
        print(f"{name} — Accuracy: {acc * 100:.1f}%  (saved {out})")

    save_cm("ORB (auto binary/gray)", y_orb)
    save_cm("SIFT + FLANN + RANSAC", y_sift)

    if t_orb and t_sift:
        print(f"\nAvg times — ORB: {np.mean(t_orb)*1000:.1f} ms | SIFT: {np.mean(t_sift)*1000:.1f} ms")
