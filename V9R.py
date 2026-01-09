import cv2
import numpy as np
import os

# =========================
# PATHS（改成你自己的）
# =========================
FRONT_IMG_PATH = r"G:\image_procsser\Image\焦前.bmp"
FOCUS_IMG_PATH = r"G:\image_procsser\Image\焦面.bmp"
BACK_IMG_PATH  = r"G:\image_procsser\Image\焦后.bmp"

OUTPUT_PATH = r"G:\image_procsser\fused_V9R.png"
DEBUG_DIR   = r"G:\image_procsser\debug_V9R"

# =========================
# Alignment safety
# =========================
MAX_ROT_DEG = 3.0
MAX_TRANS_PX = 35.0

# =========================
# Region selection params
# =========================
USE_SUPERPIXEL = True          # 有 ximgproc 就用超像素；没有会自动退化成 block
SLIC_REGION_SIZE = 35          # 超像素大小（25~60）
SLIC_RULER = 15.0              # 越大越平滑（10~30）

BLOCK = 32                     # 没有ximgproc时用块选择（16/32/48）
CONF_MARGIN = 0.08             # 置信度阈值：赢家比第二名高多少才切（0.03~0.15）
DEFAULT_TO_FOCUS = True        # 不自信时默认用焦面

# 外圈是否强制焦面（建议 True，避免3D倒角鬼影；你若坚持倒角也用winner，可改 False）
FORCE_RIM_FOCUS = True
RIM_WIDTH = 110
RIM_OUTER_EXTRA = 2

# =========================
# IO（中文路径）
# =========================
def imread_any(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def imwrite_any(path, img):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path += ext
    cv2.imencode(ext, img)[1].tofile(path)

def to_gray_u8(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def to_gray_f32(bgr):
    return to_gray_u8(bgr).astype(np.float32) / 255.0

def clahe_u8(gray_u8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_u8)

# =========================
# Circle detect + coarse align
# =========================
def detect_circle(gray_u8):
    g = cv2.GaussianBlur(gray_u8, (0, 0), 2.0)
    g = clahe_u8(g)
    H, W = g.shape[:2]
    minR = int(min(H, W) * 0.30)
    maxR = int(min(H, W) * 0.49)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(H, W) * 0.35),
        param1=120,
        param2=35,
        minRadius=minR,
        maxRadius=maxR
    )
    if circles is None:
        return None
    circles = circles[0]
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x, y, r = circles[0]
    return float(x), float(y), float(r)

def similarity_from_circles(c_mov, c_ref):
    mx, my, mr = c_mov
    rx, ry, rr = c_ref
    s = rr / (mr + 1e-6)
    M = np.array([[s, 0, rx - s * mx],
                  [0, s, ry - s * my]], dtype=np.float32)
    return M

def warp_affine(img, M, dsize_wh):
    return cv2.warpAffine(img, M, dsize_wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# =========================
# ECC refine (masked, euclidean)
# =========================
def grad_mag(img_f32):
    gx = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    return mag

def build_interior_mask(shape_hw, cx, cy, r, shrink=110):
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    m = (rr <= (r - shrink)).astype(np.uint8) * 255
    return m

def ecc_refine_masked_euclidean(mov_gray_f32, ref_gray_f32, M_init, mask_u8, iters=900, eps=1e-6):
    ref = grad_mag(ref_gray_f32)
    mov = grad_mag(mov_gray_f32)
    warpMatrix = M_init.astype(np.float32).copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, warpMatrix = cv2.findTransformECC(
            templateImage=ref,
            inputImage=mov,
            warpMatrix=warpMatrix,
            motionType=cv2.MOTION_EUCLIDEAN,
            criteria=criteria,
            inputMask=mask_u8,
            gaussFiltSize=5
        )
        return warpMatrix, float(cc), None
    except cv2.error as e:
        return warpMatrix, None, str(e)

def clamp_euclidean_delta(M_refined, M_init, max_rot_deg=3.0, max_trans_px=35.0):
    def to3(M2):
        M3 = np.eye(3, dtype=np.float32)
        M3[:2, :] = M2
        return M3
    A = to3(M_refined) @ np.linalg.inv(to3(M_init))
    a, b, tx = A[0, 0], A[0, 1], A[0, 2]
    c, d, ty = A[1, 0], A[1, 1], A[1, 2]
    theta = float(np.degrees(np.arctan2(c, a)))
    theta_c = max(min(theta, max_rot_deg), -max_rot_deg)
    tx_c = max(min(float(tx), max_trans_px), -max_trans_px)
    ty_c = max(min(float(ty), max_trans_px), -max_trans_px)
    cos_t = np.cos(np.radians(theta_c))
    sin_t = np.sin(np.radians(theta_c))
    Delta = np.array([[cos_t, -sin_t, tx_c],
                      [sin_t,  cos_t, ty_c],
                      [0,      0,     1.0]], dtype=np.float32)
    M_clamped = (Delta @ to3(M_init))[:2, :]
    return M_clamped

# =========================
# Rim mask
# =========================
def annulus_band_mask(shape_hw, cx, cy, r_inner, r_outer):
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    m = ((rr >= r_inner) & (rr <= r_outer)).astype(np.uint8) * 255
    return m

# =========================
# Focus measure (more robust than raw Laplacian)
# - 先轻微降噪，再lap
# =========================
def focus_measure(gray_f32):
    g = cv2.GaussianBlur(gray_f32, (0, 0), 1.0)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    fm = np.abs(lap)
    fm = cv2.GaussianBlur(fm, (0, 0), 1.2)
    return fm

# =========================
# Region-WTA (superpixel preferred)
# =========================
def region_wta(imgs, fms, guide_bgr, default_idx=1):
    H, W = fms[0].shape[:2]
    K = len(imgs)
    stack = np.stack(fms, axis=-1)  # H,W,K

    # 每像素置信度：top1 - top2
    s = np.sort(stack, axis=-1)
    margin = s[..., -1] - s[..., -2]  # H,W
    winner_px = np.argmax(stack, axis=-1).astype(np.uint8)

    # ---- superpixel segmentation ----
    labels = None
    if USE_SUPERPIXEL and hasattr(cv2, "ximgproc"):
        slic = cv2.ximgproc.createSuperpixelSLIC(
            guide_bgr, algorithm=cv2.ximgproc.SLICO,
            region_size=SLIC_REGION_SIZE, ruler=SLIC_RULER
        )
        slic.iterate(10)
        slic.enforceLabelConnectivity()
        labels = slic.getLabels()
    else:
        # fallback: block labels
        bh = BLOCK
        bw = BLOCK
        labels = np.zeros((H, W), np.int32)
        idx = 0
        for y in range(0, H, bh):
            for x in range(0, W, bw):
                labels[y:y+bh, x:x+bw] = idx
                idx += 1

    num_labels = int(labels.max() + 1)

    # 每个区域：计算每张图 focus measure 的平均值，选最大
    winner_reg = np.full(num_labels, default_idx, dtype=np.uint8)
    conf_reg = np.zeros(num_labels, dtype=np.float32)

    for lab in range(num_labels):
        m = (labels == lab)
        if m.sum() < 20:
            continue
        means = np.array([float(fms[k][m].mean()) for k in range(K)], dtype=np.float32)
        order = means.argsort()
        w = int(order[-1])
        second = int(order[-2])
        conf = float(means[w] - means[second])
        winner_reg[lab] = w
        conf_reg[lab] = conf

    # 生成区域winner map + 置信度门控
    winner = np.zeros((H, W), np.uint8)
    for lab in range(num_labels):
        w = int(winner_reg[lab])
        if DEFAULT_TO_FOCUS and conf_reg[lab] < CONF_MARGIN:
            w = default_idx
        winner[labels == lab] = w

    # 再做一次空间平滑（避免锯齿）
    winner = cv2.medianBlur(winner, 5)
    return winner, margin

def compose_from_winner(imgs, winner):
    fused = np.zeros_like(imgs[0])
    for i, img in enumerate(imgs):
        m = (winner == i)
        fused[m] = img[m]
    return fused

# =========================
# Main
# =========================
def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    img_f = imread_any(FRONT_IMG_PATH)
    img_0 = imread_any(FOCUS_IMG_PATH)  # reference
    img_b = imread_any(BACK_IMG_PATH)

    H, W = img_0.shape[:2]
    dsize = (W, H)

    gF = to_gray_u8(img_f)
    g0 = to_gray_u8(img_0)
    gB = to_gray_u8(img_b)

    c0 = detect_circle(g0)
    cF = detect_circle(gF)
    cB = detect_circle(gB)
    if c0 is None or cF is None or cB is None:
        raise RuntimeError("HoughCircle 失败：建议先裁剪到工件附近，或调 Hough 参数。")

    cx0, cy0, r0 = c0

    # ---- coarse align to focus frame ----
    MsF = similarity_from_circles(cF, c0)
    MsB = similarity_from_circles(cB, c0)
    front_s = warp_affine(img_f, MsF, dsize)
    back_s  = warp_affine(img_b, MsB, dsize)
    focus_w = img_0.copy()

    # ---- ECC refine on interior only ----
    g0f = to_gray_f32(focus_w)
    gFf = to_gray_f32(front_s)
    gBf = to_gray_f32(back_s)

    interior_mask = build_interior_mask((H, W), cx0, cy0, r0, shrink=110)
    I = np.array([[1, 0, 0],
                  [0, 1, 0]], dtype=np.float32)

    Mf_ecc, cc_f, err_f = ecc_refine_masked_euclidean(gFf, g0f, I, interior_mask)
    Mb_ecc, cc_b, err_b = ecc_refine_masked_euclidean(gBf, g0f, I, interior_mask)

    print("[ECC] front cc=", cc_f, "err=", err_f)
    print("[ECC] back  cc=", cc_b, "err=", err_b)

    Mf = clamp_euclidean_delta(Mf_ecc, I, MAX_ROT_DEG, MAX_TRANS_PX)
    Mb = clamp_euclidean_delta(Mb_ecc, I, MAX_ROT_DEG, MAX_TRANS_PX)

    front_w = warp_affine(front_s, Mf, dsize)
    back_w  = warp_affine(back_s,  Mb, dsize)

    # ---- focus measure ----
    fm_f = focus_measure(to_gray_f32(front_w))
    fm_0 = focus_measure(to_gray_f32(focus_w))
    fm_b = focus_measure(to_gray_f32(back_w))

    # ---- region winner take all ----
    winner, margin = region_wta(
        [front_w, focus_w, back_w],
        [fm_f, fm_0, fm_b],
        guide_bgr=focus_w,
        default_idx=1
    )

    fused = compose_from_winner([front_w, focus_w, back_w], winner)

    # ---- optional: force rim focus to guarantee "倒角只出现一次且不鬼影" ----
    if FORCE_RIM_FOCUS:
        rim = annulus_band_mask((H, W), cx0, cy0, r_inner=r0 - RIM_WIDTH, r_outer=r0 + RIM_OUTER_EXTRA)
        fused[rim > 0] = focus_w[rim > 0]
        winner[rim > 0] = 1

    imwrite_any(OUTPUT_PATH, fused)
    print("Saved:", OUTPUT_PATH)

    # ---- debug ----
    imwrite_any(os.path.join(DEBUG_DIR, "overlay_focus_front.png"),
                cv2.addWeighted(focus_w, 0.5, front_w, 0.5, 0))
    imwrite_any(os.path.join(DEBUG_DIR, "overlay_focus_back.png"),
                cv2.addWeighted(focus_w, 0.5, back_w, 0.5, 0))

    def norm_u8(x):
        x = x / (x.max() + 1e-6)
        return np.clip(x * 255, 0, 255).astype(np.uint8)

    imwrite_any(os.path.join(DEBUG_DIR, "fm_front.png"), norm_u8(fm_f))
    imwrite_any(os.path.join(DEBUG_DIR, "fm_focus.png"), norm_u8(fm_0))
    imwrite_any(os.path.join(DEBUG_DIR, "fm_back.png"),  norm_u8(fm_b))

    # winner map 可视化：0/1/2 -> 0/127/255
    imwrite_any(os.path.join(DEBUG_DIR, "winner_map.png"), (winner * 127).astype(np.uint8))
    imwrite_any(os.path.join(DEBUG_DIR, "margin_map.png"), norm_u8(margin))

    # interior/rim mask
    mask_vis = cv2.cvtColor(to_gray_u8(focus_w), cv2.COLOR_GRAY2BGR)
    mask_vis[interior_mask > 0] = (255, 0, 0)  # blue
    if FORCE_RIM_FOCUS:
        rim = annulus_band_mask((H, W), cx0, cy0, r_inner=r0 - RIM_WIDTH, r_outer=r0 + RIM_OUTER_EXTRA)
        mask_vis[rim > 0] = (0, 255, 255)  # yellow
        imwrite_any(os.path.join(DEBUG_DIR, "rim_mask.png"), rim)
    imwrite_any(os.path.join(DEBUG_DIR, "mask_interior_rim.png"), mask_vis)

    print("Debug saved:", DEBUG_DIR)

if __name__ == "__main__":
    main()
