import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import pickle
import os
import sys
import csv

# ==========================================
# 1. SMART PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GT_PKL_PATH = os.path.join(PROJECT_ROOT, "datasets", "holistic_49_keypoints.pkl")
GLOSS_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "gloss_map.csv")

# ==========================================
# 2. EXACT COLOR PALETTE (From Reference)
# ==========================================
# Rainbow Fingers
C_THUMB   = "#FF0000"   # Red
C_INDEX   = "#00AA00"   # Green
C_MIDDLE  = "#0000CD"   # Blue
C_RING    = "#FF69B4"   # Pink
C_PINKY   = "#FFD700"   # Yellow

# Body Parts
C_SPINE   = "#00AA00"   # Green (Neck/Head)
C_SHOULDER= "#FF0000"   # Red
C_ARM     = "#FF8C00"   # Orange

# ==========================================
# 3. TOPOLOGY DEFINITIONS
# ==========================================
# Fingers
LH_THUMB  = [(0,1), (1,2), (2,3), (3,4)]
LH_INDEX  = [(0,5), (5,6), (6,7), (7,8)]
LH_MIDDLE = [(0,9), (9,10),(10,11),(11,12)]
LH_RING   = [(0,13),(13,14),(14,15),(15,16)]
LH_PINKY  = [(0,17),(17,18),(18,19),(19,20)]

RH_THUMB  = [(u+21, v+21) for u,v in LH_THUMB]
RH_INDEX  = [(u+21, v+21) for u,v in LH_INDEX]
RH_MIDDLE = [(u+21, v+21) for u,v in LH_MIDDLE]
RH_RING   = [(u+21, v+21) for u,v in LH_RING]
RH_PINKY  = [(u+21, v+21) for u,v in LH_PINKY]

# Arms
ARM_LEFT  = [(44, 46), (46, 48)]
ARM_RIGHT = [(43, 45), (45, 47)]

# ==========================================
# 4. DRAWING FUNCTIONS (Thin, Clean Lines)
# ==========================================
def hard_clamp_hand(points, wrist, shoulder_width):
    if shoulder_width < 0.01: shoulder_width = 0.15
    limit = shoulder_width * 0.8
    for i in range(len(points)):
        vec = points[i] - wrist
        dist = np.linalg.norm(vec[:2])
        if dist > limit:
            points[i] = wrist + (vec * (limit / dist))
    return points

def draw_skeleton_clean(ax, f):
    artists = []
    f = f.copy()

    # 1. Scale & Clamp Hands
    l_sh, r_sh = f[43][:2], f[44][:2]
    s_width = np.linalg.norm(l_sh - r_sh)
    f[0:21] += (f[47] - f[0]); f[0:21] = hard_clamp_hand(f[0:21], f[47], s_width)
    f[21:42] += (f[48] - f[21]); f[21:42] = hard_clamp_hand(f[21:42], f[48], s_width)

    # 2. Draw Shoulders (Red Line)
    # Connect 43 to 44
    line, = ax.plot([l_sh[0], r_sh[0]], [l_sh[1], r_sh[1]], color=C_SHOULDER, lw=2)
    artists.append(line)

    # 3. Draw Spine/Head (Green Line)
    cx, cy = (l_sh[0] + r_sh[0])/2, (l_sh[1] + r_sh[1])/2
    # Draw straight up
    line, = ax.plot([cx, cx], [cy, cy-0.25], color=C_SPINE, lw=2)
    artists.append(line)

    # 4. Helper to draw segments
    def plot_seg(topo, color, lw=2):
        for u, v in topo:
            l, = ax.plot([f[u,0], f[v,0]], [f[u,1], f[v,1]], color=color, lw=lw)
            artists.append(l)

    # 5. Draw Arms (Orange)
    plot_seg(ARM_LEFT, C_ARM, lw=2)
    plot_seg(ARM_RIGHT, C_ARM, lw=2)

    # 6. Draw Hands (Rainbow)
    # Left
    plot_seg(LH_THUMB, C_THUMB, 1.5); plot_seg(LH_INDEX, C_INDEX, 1.5)
    plot_seg(LH_MIDDLE, C_MIDDLE, 1.5); plot_seg(LH_RING, C_RING, 1.5); plot_seg(LH_PINKY, C_PINKY, 1.5)
    # Right
    plot_seg(RH_THUMB, C_THUMB, 1.5); plot_seg(RH_INDEX, C_INDEX, 1.5)
    plot_seg(RH_MIDDLE, C_MIDDLE, 1.5); plot_seg(RH_RING, C_RING, 1.5); plot_seg(RH_PINKY, C_PINKY, 1.5)

    return artists

# ==========================================
# 5. SCIENTIFIC LOOKUP
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
                if target_gloss in gid.upper() or "GLOSS_0" in gid.upper(): return vid
    if len(gt_dataset) > 0: return list(gt_dataset.keys())[0]
    return None

# ==========================================
# 6. MAIN (REPLICA LAYOUT & FIXED TIMING)
# ==========================================
def main():
    target_name = "hello_world"
    if len(sys.argv) > 1: target_name = sys.argv[1].replace(".npy", "").replace(".mp4", "")
    print(f"🔬 PROCESSING: {target_name}")

    pred_path = os.path.join(PROJECT_ROOT, "output", f"{target_name}.npy")
    if not os.path.exists(pred_path): print(f"❌ Error: {pred_path} not found."); return

    target_gloss = target_name.split("_")[0].upper()
    gt_id = find_ground_truth_id(target_gloss)
    if gt_id is None: print("❌ Ground Truth not found."); return
    print(f"✅ FOUND ID: {gt_id}")

    pred = np.load(pred_path);
    if isinstance(pred, list): pred = pred[0]
    with open(GT_PKL_PATH, "rb") as f: gt = pickle.load(f)[gt_id]
    gt = np.array(gt)

    # --- FIX TIMING: USE MAX LENGTH ---
    max_len = max(len(pred), len(gt))

    # --- PLOT SETUP (EXACT REPLICA) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.1)

    # 1. FULL VERTICAL DIVIDER (Top to Bottom)
    line = Line2D([0.5, 0.5], [0.05, 0.95], transform=fig.transFigure, color="black", linewidth=2)
    fig.add_artist(line)

    # 2. TEXT LABELS (Matching Font & Color)
    # Left: Red Prediction
    ax1.text(0.5, -0.1, "Predicted Sign Pose", transform=ax1.transAxes,
             ha='center', fontsize=18, color='#EE3333', fontname='sans-serif', weight='normal')

    # Right: Black Ground Truth
    ax2.text(0.5, -0.1, "Ground Truth Pose", transform=ax2.transAxes,
             ha='center', fontsize=18, color='black', fontname='sans-serif', weight='normal')

    # ID Below Right Side (Small, Black)
    ax2.text(0.5, -0.2, f"Sequence ID: {gt_id}", transform=ax2.transAxes,
             ha='center', fontsize=10, color='black', fontname='monospace', weight='bold')

    for ax in [ax1, ax2]:
        ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(0,1); ax.set_ylim(1,0)

    arts = []
    def update(i):
        nonlocal arts
        for a in arts: a.remove()
        arts = []

        # Safe Indexing: If i is past the end, use the last frame (freeze)
        idx_pred = min(i, len(pred) - 1)
        idx_gt = min(i, len(gt) - 1)

        arts.extend(draw_skeleton_clean(ax1, pred[idx_pred]))
        arts.extend(draw_skeleton_clean(ax2, gt[idx_gt]))

    ani = FuncAnimation(fig, update, frames=max_len, interval=40)
    out_name = f"compare_{target_name}.mp4"
    ani.save(os.path.join(PROJECT_ROOT, out_name), writer="ffmpeg", dpi=150)
    print(f"✅ Video Saved: {out_name}")

if __name__ == "__main__":
    main()
