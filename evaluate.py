import numpy as np
import os
import sys
import csv
import pickle
from scipy.spatial.distance import cdist
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GT_PKL_PATH = os.path.join(PROJECT_ROOT, "datasets", "holistic_49_keypoints.pkl")
GLOSS_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "gloss_map.csv")

# Joint indices for Angle Calculation
JOINTS_ANGLE = {
    "Right Elbow": (43, 45, 47),
    "Left Elbow":  (44, 46, 48)
}

# ==========================================
# 2. FID CALCULATOR CLASS
# ==========================================
class FIDCalculator:
    """Calculates Visual Quality (FID) between two video files."""
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_features(self, video_path):
        """Extracts Inception features from video frames."""
        cap = cv2.VideoCapture(video_path)
        features = []
        if not cap.isOpened(): return None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            # Process every 5th frame to save time
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5 == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                batch = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred = self.model(batch).cpu().numpy().flatten()
                features.append(pred)
        cap.release()
        return np.array(features) if features else None

    def calculate_fid(self, real_path, gen_path):
        feat1 = self.get_features(real_path)
        feat2 = self.get_features(gen_path)
        
        if feat1 is None or feat2 is None: return None
        if len(feat1) < 2 or len(feat2) < 2: return None # Need variance

        mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1, rowvar=False)
        mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2, rowvar=False)
        
        diff = mu1 - mu2
        from scipy.linalg import sqrtm
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean): covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def find_ground_truth_id(target_gloss="HELLO"):
    if not os.path.exists(GLOSS_CSV_PATH) or not os.path.exists(GT_PKL_PATH): return None
    with open(GT_PKL_PATH, "rb") as f: gt_dataset = pickle.load(f)
    with open(GLOSS_CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1:
                a, b = row[0].strip(), row[1].strip()
                if a in gt_dataset: vid, gid = a, b
                elif b in gt_dataset: vid, gid = b, a
                else: continue
                if target_gloss.upper() in gid.upper() or "GLOSS_0" in gid.upper(): return vid
    if len(gt_dataset) > 0: return list(gt_dataset.keys())[0]
    return None

def compute_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_rad)

def dtw_distance(s1, s2):
    dists = cdist(s1.reshape(len(s1), -1), s2.reshape(len(s2), -1), metric='euclidean')
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dists[i-1, j-1]
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]

# ==========================================
# 4. MAIN EVALUATION LOGIC
# ==========================================
def main():
    target_name = "hello" # Default
    if len(sys.argv) > 1: target_name = sys.argv[1]

    print(f"📊 EVALUATING METRICS FOR: {target_name.upper()}")

    # 1. Load Prediction
    pred_path = os.path.join(PROJECT_ROOT, "output", f"{target_name}.npy")
    if not os.path.exists(pred_path):
        print(f"❌ Error: {pred_path} not found. Run pipeline first.")
        return
    pred = np.load(pred_path)
    if isinstance(pred, list): pred = pred[0]

    # 2. Load Ground Truth
    gt_id = find_ground_truth_id(target_name)
    print(f"✅ Comparing vs Ground Truth ID: {gt_id}")
    
    with open(GT_PKL_PATH, "rb") as f:
        gt_dataset = pickle.load(f)
    gt = np.array(gt_dataset[gt_id])

    # 3. Calculate Keypoint Metrics
    min_len = min(len(pred), len(gt))
    p_trim = pred[:min_len]
    g_trim = gt[:min_len]

    # MPJPE
    dist = np.linalg.norm(p_trim - g_trim, axis=2)
    mpjpe = np.mean(dist)

    # MPJAE
    angle_errors = []
    for i in range(min_len):
        for name, (idx_a, idx_b, idx_c) in JOINTS_ANGLE.items():
            ang_p = compute_angle(p_trim[i, idx_a], p_trim[i, idx_b], p_trim[i, idx_c])
            ang_g = compute_angle(g_trim[i, idx_a], g_trim[i, idx_b], g_trim[i, idx_c])
            angle_errors.append(abs(ang_p - ang_g))
    mpjae = np.mean(angle_errors)

    # DTW
    dtw_score = dtw_distance(pred, gt)
    dtw_norm = dtw_score / (len(pred) + len(gt))

    # 4. Calculate FID (Visual Metric)
    print("⏳ Calculating FID (This may take a moment)...")
    fid_score = None
    try:
        vid_gen = os.path.join(PROJECT_ROOT, "output", f"{target_name}.mp4")
        vid_gt  = os.path.join(PROJECT_ROOT, "output", f"{target_name}_gt.mp4")
        
        if os.path.exists(vid_gen) and os.path.exists(vid_gt):
            fid_calc = FIDCalculator()
            fid_score = fid_calc.calculate_fid(vid_gt, vid_gen)
        else:
            print(f"⚠️  Skipping FID: Videos not found ({target_name}.mp4 or _gt.mp4 missing)")
    except Exception as e:
        print(f"⚠️  FID Error: {e}")

    # 5. Print Results
    print("-" * 40)
    print(f"1. MPJPE (Position Error):  {mpjpe:.4f}")
    print(f"2. MPJAE (Angle Error):     {mpjae:.2f} degrees")
    print(f"3. DTW (Temporal Match):    {dtw_norm:.4f}")
    
    if fid_score is not None:
        print(f"4. FID (Visual Quality):    {fid_score:.4f}")
    else:
        print(f"4. FID (Visual Quality):    N/A (Requires video files)")
        
    print("-" * 40)
    print("✅ Done.")

if __name__ == "__main__":
    main()
