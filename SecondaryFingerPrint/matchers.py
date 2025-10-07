from dataclasses import dataclass
import os
import cv2 as cv
import numpy as np

# ---------- Config ----------
@dataclass
class Config:
    # ORB (binary/fingerprint)
    orb_nfeatures: int = 1000
    orb_ratio: float = 0.70
    # ORB (gray/UiA)
    orb_nfeatures_gray: int = 3000
    orb_ratio_gray: float = 0.85
    # SIFT + RANSAC (generelt)
    sift_ratio: float = 0.70
    ransac_reproj: float = 3.0
    # SIFT + RANSAC (gray/UiA)
    sift_ratio_gray: float = 0.80
    ransac_reproj_gray: float = 6.0
    use_homography_gray: bool = True

    orb_thr_binary: int = 20
    orb_thr_gray: int = 30
    sift_thr: int = 12
    sift_thr_gray: int = 6

    save_vis: bool = True

# ---------- IO ----------
def read_gray(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def is_uia_folder(name: str) -> bool:
    return "uia" in name.lower()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ---------- Preprocess ----------
def preprocess_binary(img_gray: np.ndarray) -> np.ndarray:
    _, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    return img_bin

def preprocess_enhanced_gray(img_gray: np.ndarray) -> np.ndarray:
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(img_gray)
    g = cv.GaussianBlur(g, (3, 3), 0)
    return g


# ---------- ORB ----------
def match_orb(img1_gray: np.ndarray, img2_gray: np.ndarray, cfg: Config, mode: str = "binary"):
    if mode == "binary":
        a = preprocess_binary(img1_gray); b = preprocess_binary(img2_gray)
        ratio = cfg.orb_ratio; nfeatures = cfg.orb_nfeatures
    else:
        a = preprocess_enhanced_gray(img1_gray); b = preprocess_enhanced_gray(img2_gray)
        ratio = cfg.orb_ratio_gray; nfeatures = cfg.orb_nfeatures_gray

    orb = cv.ORB_create(nfeatures=nfeatures)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        return 0, k1 or [], k2 or [], [], None

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]

    # fallback om ratio-test gir 0
    if not good:
        bf_x = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        good = sorted(bf_x.match(d1, d2), key=lambda m: m.distance)

    vis = cv.drawMatches(a, k1, b, k2, good, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good), k1, k2, good, vis

# ---------- SIFT + RANSAC/H ----------
def match_sift_flann_ransac(img1_gray: np.ndarray, img2_gray: np.ndarray, cfg: Config, mode: str = "binary"):
    g1 = preprocess_enhanced_gray(img1_gray)
    g2 = preprocess_enhanced_gray(img2_gray)

    sift = cv.SIFT_create(nfeatures=2000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return 0, k1 or [], k2 or [], [], None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=128)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    raw = flann.knnMatch(d1, d2, k=2)
    ratio = cfg.sift_ratio_gray if mode == "gray" else cfg.sift_ratio
    good = [m for m, n in raw if m.distance < ratio * n.distance]

    if len(good) < 4:
        vis = cv.drawMatches(g1, k1, g2, k2, good, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return 0, k1, k2, good, vis

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if mode == "gray" and cfg.use_homography_gray:
        H, mask = cv.findHomography(pts1.reshape(-1,2), pts2.reshape(-1,2),
                                    cv.RANSAC, ransacReprojThreshold=cfg.ransac_reproj_gray)
    else:
        H, mask = cv.estimateAffinePartial2D(pts1, pts2, method=cv.RANSAC,
                                             ransacReprojThreshold=(cfg.ransac_reproj_gray if mode=="gray" else cfg.ransac_reproj))

    inliers = int(mask.sum()) if mask is not None else 0
    keep_matches = [gm for gm, ok in zip(good, (mask.ravel().tolist() if mask is not None else [])) if ok]
    vis = cv.drawMatches(g1, k1, g2, k2, keep_matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return inliers, k1, k2, keep_matches, vis
