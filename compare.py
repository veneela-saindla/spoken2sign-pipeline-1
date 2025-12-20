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
# 2. EXACT COLOR PALETTE
# ==========================================
C_RED_ARM       = "#FF0000"   
C_GREEN_ARM     = "#00AA00"   
C_ORANGE_DARK   = "#FF8C00"   
C_ORANGE_LIGHT  = "#FFD700"   
C_HEAD          = "#00AA00"   

C_THUMB   = "#FF0000"   
C_INDEX   = "#00AA00"   
C_MIDDLE  = "#0000CD"   
C_RING    = "#FF69B4"   
C_PINKY   = "#FFD700"   

# Topology
ARM_RIGHT_UPPER = [(43, 45)]; ARM_RIGHT_LOWER = [(45, 47)]
ARM_LEFT_UPPER  = [(44, 46)]; ARM_LEFT_LOWER  = [(46, 48)]
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

# ==========================================
# 3. DRAWING FUNCTIONS
# ==========================================
def stacked_line(ax, x0, y0, x1, y1, color, base_lw=2.5, n=4):
    artists = []
    for i in range(n):
        t0 = i / (2 * n); t1 = 1 - t0
        xm0 = x0 + (x1 - x0) * t0; ym0 = y0 + (y1 - y0) * t0
        xm1 = x0 + (x1 - x0) * t1; ym1 = y0 + (y1 - y0) * t1
        lw = base_lw * (0.4 + 0.6 * (i + 1) / n)
        line, = ax.plot([xm0, xm1], [ym0, ym1], color=color, lw=lw, solid_capstyle="round", zorder=10+i)
        artists.append(line)
    return artists

def hard_clamp_hand(points, wrist, shoulder_width):
    if shoulder_width < 0.01: shoulder_width = 0.15 
    limit = shoulder_width * 0.8 
    for i in range(len(points)):
        vec = points[i] - wrist
        dist = np.linalg.norm(vec[:2]) 
        if dist > limit:
            points[i] = wrist + (vec * (limit / dist))
    return points

def draw_skeleton_high_quality(ax, f):
    artists = []
    f = f.copy()
    l_sh, r_sh = f[43][:2], f[44][:2]
    s_width = np.linalg.norm(l_sh - r_sh)
    f[0:21] += (f[47] - f[0]); f[0:21] = hard_clamp_hand(f[0:21], f[47], s_width)
    f[21:42] += (f[48] - f[21]); f[21:42] = hard_clamp_hand(f[21:42], f[48], s_width)

    def draw_part(topo, color, lw_scale=1.0):
        for (u, v) in topo:
            artists.extend(stacked_line(ax, f[u,0], f[u,1], f[v,0], f[v,1], color, base_lw=2.5*lw_scale))

    cx, cy = (l_sh[0] + r_sh[0])/2, (l_sh[1] + r_sh[1])/2
    cy_dip = cy + 0.005
    artists.extend(stacked_line(ax, l_sh[0], l_sh[1], cx, cy_dip, C_RED_ARM, base_lw=3.0))
    artists.extend(stacked_line(ax, r_sh[0], r_sh[1], cx, cy_dip, C_RED_ARM, base_lw=3.0))
    artists.extend(stacked_line(ax, cx, cy_dip, cx, cy-0.20, C_GREEN_ARM, base_lw=2.5))

    draw_part(ARM_RIGHT_UPPER, C_ORANGE_DARK); draw_part(ARM_RIGHT_LOWER, C_ORANGE_LIGHT)
    draw_part(ARM_LEFT_UPPER,  C_ORANGE_DARK); draw_part(ARM_LEFT_LOWER,  C_ORANGE_LIGHT)

    s = 0.8
    draw_part(LH_THUMB, C_THUMB, s); draw_part(LH_INDEX, C_INDEX, s)
    draw_part(LH_MIDDLE, C_MIDDLE, s); draw_part(LH_RING, C_RING, s); draw_part(LH_PINKY, C_PINKY, s)
    draw_part(RH_THUMB, C_THUMB, s); draw_part(RH_INDEX, C_INDEX, s)
    draw_part(RH_MIDDLE, C_MIDDLE, s); draw_part(RH_RING, C_RING, s); draw_part(RH_PINKY, C_PINKY, s)

    return artists

# ==========================================
# 4. SCIENTIFIC LOOKUP
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
    if len(gt_dataset) > 0: return list(gt_dataset.keys())[0] # Fallback
    return None

# ==========================================
# 5. MAIN (CLEAN FONTS)
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
    
    # --- PLOT SETUP ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.1)
    
    # 1. FULL VERTICAL DIVIDER
    line = Line2D([0.5, 0.5], [0.05, 0.95], transform=fig.transFigure, color="black", linewidth=2)
    fig.add_artist(line)

    # 2. LABELS (Fixed Font)
    # Use 'sans-serif' instead of 'Arial' to avoid Linux warnings
    ax1.text(0.5, -0.1, "Predicted Sign Pose", transform=ax1.transAxes, 
             ha='center', fontsize=18, color='#EE3333', fontname='sans-serif', weight='normal')
    
    ax2.text(0.5, -0.1, "Ground Truth Pose", transform=ax2.transAxes, 
             ha='center', fontsize=18, color='black', fontname='sans-serif', weight='normal')
    
    ax2.text(0.5, -0.2, f"Sequence ID: {gt_id}", transform=ax2.transAxes, 
             ha='center', fontsize=10, color='black', fontname='monospace', weight='bold')

    for ax in [ax1, ax2]:
        ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(0,1); ax.set_ylim(1,0)

    # Animation
    arts = []
    def update(i):
        nonlocal arts
        for a in arts: a.remove()
        arts = []
        arts.extend(draw_skeleton_high_quality(ax1, pred[i]))
        arts.extend(draw_skeleton_high_quality(ax2, gt[i]))

    ani = FuncAnimation(fig, update, frames=min(len(pred), len(gt)), interval=40)
    out_name = f"compare_{target_name}.mp4"
    ani.save(os.path.join(PROJECT_ROOT, out_name), writer="ffmpeg", dpi=150)
    print(f"✅ Video Saved: {out_name}")

if __name__ == "__main__":
    main()
