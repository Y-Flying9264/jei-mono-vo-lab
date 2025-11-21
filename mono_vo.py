import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ================== CONFIG ==================

VIDEO_PATH = r"E:/Projects/zzzz_exp/new_walk.mp4"   # TODO: change to your own path

USE_CAMERA_CALIB = False
CAM_K_PATH = "K.npy"
CAM_DIST_PATH = "dist.npy"

RESIZE_SCALE = 0.5        # 1080p -> 540p
MAX_FEATURES = 2000
MIN_MATCHES = 100
RATIO_TEST = 0.7
ASSUMED_FOV_DEG = 60.0

STEP_SCALE = 1.0
FRAME_STEP = 2

SHOW_FRAME = True

SAVE_TRAJ_FIG = True
TRAJ_FIG_NAME_MAIN = "trajectory.png"
TRAJ_FIG_NAME_RAW_SMOOTH = "trajectory_raw_vs_smooth.png"
TRAJ_FIG_NAME_STEP_DIST = "step_distance.png"

SMOOTHING_WINDOW = 5


# ================== UTILS ==================

def build_intrinsics(width, height, fov_deg=60.0):
    """Build pinhole intrinsics K from image size and assumed horizontal FOV."""
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def extract_orb_features(gray, orb):
    """Extract ORB keypoints and descriptors from a grayscale frame."""
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(des1, des2, kp1, kp2, ratio=0.7):
    """Match ORB descriptors with BFMatcher + KNN + Lowe ratio test."""
    if des1 is None or des2 is None:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []

    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(pts1) < 8:
        return None, None

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    return pts1, pts2


def estimate_pose_from_essential(pts1, pts2, K):
    """Estimate essential matrix with RANSAC and recover relative pose (R, t)."""
    E, mask = cv2.findEssentialMat(
        pts1, pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None:
        return None, None, None

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask_pose


def smooth_1d(x, window=5):
    """1-D moving average smoothing."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


# ================== MAIN VO PIPELINE ==================

def run_visual_odometry(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    ret, frame0 = cap.read()
    if not ret:
        print("[ERROR] Video is empty or failed to read the first frame.")
        return

    original_height, original_width = frame0.shape[:2]

    proc_width = int(original_width * RESIZE_SCALE)
    proc_height = int(original_height * RESIZE_SCALE)

    frame0_resized = cv2.resize(frame0, (proc_width, proc_height))

    if USE_CAMERA_CALIB:
        if not (os.path.exists(CAM_K_PATH) and os.path.exists(CAM_DIST_PATH)):
            print("[WARN] K.npy or dist.npy not found, falling back to approximate FOV intrinsics.")
            use_calib = False
        else:
            use_calib = True
    else:
        use_calib = False

    dist = None
    if use_calib:
        K = np.load(CAM_K_PATH)
        dist = np.load(CAM_DIST_PATH)
        frame0_proc = cv2.undistort(frame0_resized, K, dist)
        print("[INFO] Using calibrated intrinsics and distortion.")
    else:
        K = build_intrinsics(proc_width, proc_height, fov_deg=ASSUMED_FOV_DEG)
        frame0_proc = frame0_resized
        print("[INFO] Using approximate intrinsics from assumed FOV.")

    print("[INFO] Camera intrinsic matrix K:")
    print(K)

    gray0 = cv2.cvtColor(frame0_proc, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    kp_prev, des_prev = extract_orb_features(gray0, orb)

    R_global = np.eye(3, dtype=np.float64)
    pos = np.zeros((3, 1), dtype=np.float64)

    trajectory = [pos.copy()]
    step_dists = []
    frame_idx = 0

    if SHOW_FRAME:
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    print("[INFO] Start processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_STEP != 0:
            continue

        frame_resized = cv2.resize(frame, (proc_width, proc_height))

        if use_calib:
            frame_proc = cv2.undistort(frame_resized, K, dist)
        else:
            frame_proc = frame_resized

        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

        kp, des = extract_orb_features(gray, orb)
        if des is None or des_prev is None or len(kp) < 10 or len(kp_prev) < 10:
            kp_prev, des_prev = kp, des
            continue

        pts_prev, pts_curr = match_features(des_prev, des, kp_prev, kp, ratio=RATIO_TEST)
        if pts_prev is None or pts_curr is None or len(pts_prev) < MIN_MATCHES:
            kp_prev, des_prev = kp, des
            continue

        pixel_motion = np.linalg.norm(pts_curr - pts_prev, axis=1)
        step_pixel_motion = float(np.median(pixel_motion))

        R_rel, t_rel, mask_pose = estimate_pose_from_essential(pts_prev, pts_curr, K)
        if R_rel is None or t_rel is None:
            kp_prev, des_prev = kp, des
            continue

        step_norm = np.linalg.norm(t_rel)
        if step_norm < 1e-6:
            kp_prev, des_prev = kp, des
            continue

        t_step = (STEP_SCALE / step_norm) * t_rel

        max_step = 2.0 * STEP_SCALE
        if np.linalg.norm(t_step) > max_step:
            t_step = t_step * (max_step / np.linalg.norm(t_step))

        pos = pos + R_global @ t_step
        R_global = R_rel @ R_global

        trajectory.append(pos.copy())
        step_dists.append(step_pixel_motion)

        if SHOW_FRAME:
            vis_frame = frame_proc.copy()
            cv2.drawKeypoints(frame_proc, kp, vis_frame, color=(0, 255, 0))
            cv2.imshow("Frame", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        kp_prev, des_prev = kp, des

    cap.release()
    if SHOW_FRAME:
        cv2.destroyAllWindows()

    if len(trajectory) < 2:
        print("[WARN] Too few valid trajectory points, cannot plot.")
        return

    traj_array = np.hstack(trajectory).T
    xs = traj_array[:, 0]
    ys = traj_array[:, 1]
    zs = traj_array[:, 2]

    xs_smooth = smooth_1d(xs, window=SMOOTHING_WINDOW)
    zs_smooth = smooth_1d(zs, window=SMOOTHING_WINDOW)

    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Main trajectory (smoothed, X–Z plane)
    plt.figure(figsize=(6, 6))
    plt.plot(xs_smooth, zs_smooth, marker='.', linewidth=1, label="Trajectory")

    plt.scatter(xs_smooth[-1], zs_smooth[-1], c='g', marker='o', s=60, label='Start')
    plt.scatter(xs_smooth[0],  zs_smooth[0],  c='r', marker='x', s=80, label='End')

    if len(xs_smooth) >= 2:
        plt.annotate(
            '',
            xy=(xs_smooth[-1], zs_smooth[-1]),
            xytext=(xs_smooth[-2], zs_smooth[-2]),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='blue')
        )

    plt.annotate('Start',
                 xy=(xs_smooth[-1], zs_smooth[-1]),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=8)
    plt.annotate('End',
                 xy=(xs_smooth[0], zs_smooth[0]),
                 xytext=(5, -10),
                 textcoords='offset points',
                 fontsize=8)

    plt.xlabel("X (arbitrary units)")
    plt.ylabel("Z (arbitrary units)")
    plt.title("Monocular Visual Odometry Trajectory (up-to-scale)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    if SAVE_TRAJ_FIG:
        fig_path_main = os.path.join(root_dir, TRAJ_FIG_NAME_MAIN)
        plt.savefig(fig_path_main, dpi=300)
        print(f"[INFO] Main trajectory figure saved to: {fig_path_main}")

    # Raw vs smoothed trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(xs, zs, color='0.6', linestyle='--', linewidth=1.5, label='Raw Trajectory')
    plt.plot(xs_smooth, zs_smooth, color='C1', linewidth=2.0, label='Smoothed Trajectory')
    plt.xlabel("X (arbitrary units)")
    plt.ylabel("Z (arbitrary units)")
    plt.title("Raw vs Smoothed Trajectory")
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    if SAVE_TRAJ_FIG:
        fig_path_raw_smooth = os.path.join(root_dir, TRAJ_FIG_NAME_RAW_SMOOTH)
        plt.savefig(fig_path_raw_smooth, dpi=300)
        print(f"[INFO] Raw vs smoothed trajectory figure saved to: {fig_path_raw_smooth}")

    # Per-step image motion
    step_dists = np.array(step_dists)
    plt.figure(figsize=(7, 4))
    plt.plot(step_dists, marker='.', linewidth=1)
    plt.xlabel("Step index (between keyframes)")
    plt.ylabel("Median pixel displacement")
    plt.title("Per-step Image Motion Magnitude")
    plt.grid(True)
    plt.tight_layout()

    if SAVE_TRAJ_FIG:
        fig_path_step = os.path.join(root_dir, TRAJ_FIG_NAME_STEP_DIST)
        plt.savefig(fig_path_step, dpi=300)
        print(f"[INFO] Per-step image motion figure saved to: {fig_path_step}")

    plt.show()

    # Numerical stats
    total_motion = float(step_dists.sum())
    final_disp = float(np.linalg.norm(traj_array[-1] - traj_array[0]))
    straightness = float(final_disp / (total_motion + 1e-8))
    avg_step = float(step_dists.mean()) if len(step_dists) > 0 else 0.0
    std_step = float(step_dists.std()) if len(step_dists) > 0 else 0.0

    print("\n========== VO Statistics (arbitrary scale units) ==========")
    print(f"Number of keyframes (trajectory points): {len(trajectory)}")
    print(f"Total image motion (sum of steps):       {total_motion:.4f}")
    print(f"Displacement from start to end:          {final_disp:.4f}")
    print(f"Straightness index:                      {straightness:.4f}")
    print(f"Mean per-step image motion:              {avg_step:.4f}")
    print(f"Std of per-step image motion:            {std_step:.4f}")
    print("===========================================================\n")

    np.savetxt("trajectory.txt", traj_array, fmt="%.6f")
    print("[INFO] Trajectory data saved to trajectory.txt")

    stats_path = os.path.join(root_dir, "vo_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Monocular Visual Odometry Statistics (up-to-scale)\n")
        f.write(f"Number of keyframes: {len(trajectory)}\n")
        f.write(f"Total image motion (sum of steps): {total_motion:.6f}\n")
        f.write(f"Displacement from start to end: {final_disp:.6f}\n")
        f.write(f"Straightness index: {straightness:.6f}\n")
        f.write(f"Average per-step image motion: {avg_step:.6f}\n")
        f.write(f"Std of per-step image motion: {std_step:.6f}\n")

    print(f"[INFO] VO statistics saved to: {stats_path}")
    print("[INFO] Processing finished.")


if __name__ == "__main__":
    run_visual_odometry(VIDEO_PATH)
