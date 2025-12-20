import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import pickle
import os
import sys
import csv

# ==========================================
# 1. SETUP
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GT_PKL_PATH = os.path.join(PROJECT_ROOT, "datasets", "holistic_49_keypoints.pkl")
GLOSS_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "gloss_map.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Colors
C_RED_ARM="#FF0000"; C_GREEN_ARM="#00AA00"; C_ORANGE_DARK="#FF8C00"; C_ORANGE_LIGHT="#FFD700"
C_HEAD="#00AA00"; C_TEXT="#FF0000"
C_THUMB="#FF0000"; C_INDEX="#00AA00"; C_MIDDLE="#0000CD"; C_RING="#FF69B4"; C_PINKY="#FFD700"
C_THUMB_R="#FF0000"; C_INDEX_R="#00AA00"; C_MIDDLE_R="#0000CD"; C_RING_R="#FF69B4"; C_PINKY_R="#FFD700"

# Topology
ARM_RIGHT_UPPER=[(43,45)]; ARM_RIGHT_LOWER=[(45,47)]
ARM_LEFT_UPPER=[(44,46)]; ARM_LEFT_LOWER=[(46,48)]
LH_THUMB=[(0,1),(1,2),(2,3),(3,4)]; LH_INDEX=[(0,5),(5,6),(6,7),(7,8)]
LH_MIDDLE=[(0,9),(9,10),(10,11),(11,12)]; LH_RING=[(0,13),(13,14),(14,15),(15,16)]
LH_PINKY=[(0,17),(17,18),(18,19),(19,20)]
RH_THUMB=[(u+21,v+21) for u,v in LH_THUMB]; RH_INDEX=[(u+21,v+21) for u,v in LH_INDEX]
RH_MIDDLE=[(u+21,v+21) for u,v in LH_MIDDLE]; RH_RING=[(u+21,v+21) for u,v in LH_RING]
RH_PINKY=[(u+21,v+21) for u,v in LH_PINKY]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def resample_sequence(data, target_length):
    """
    Smartly resamples the sequence to match target_length.
    Auto-detects feature dimensions (works for x,y,z OR x,y,z,vis).
    """
    current_length = data.shape[0]
    if current_length == target_length: return data
    
    # 1. Capture the shape of a single frame (e.g., 49, 3 or 49, 4)
    feature_shape = data.shape[1:] 

    # 2. Flatten all features into a single vector per frame
    data_flat = data.reshape(current_length, -1)
    
    # 3. Interpolate
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    new_data_flat = np.zeros((target_length, data_flat.shape[1]))
    
    for i in range(data_flat.shape[1]):
        new_data_flat[:, i] = np.interp(x_new, x_old, data_flat[:, i])
        
    # 4. Reshape back to (Target_Time, Original_Features...)
    return new_data_flat.reshape((target_length,) + feature_shape)

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

def draw_skeleton_stacked(ax, f):
    artists = []
    f = f.copy()
    l_sh, r_sh = f[43][:2], f[44][:2]
    s_width = np.linalg.norm(l_sh - r_sh)
    f[0:21] += (f[47] - f[0]); f[0:21] = hard_clamp_hand(f[0:21], f[47], s_width)
    f[21:42] += (f[48] - f[21]); f[21:42] = hard_clamp_hand(f[21:42], f[48], s_width)

    cx, cy = (l_sh[0] + r_sh[0])/2, (l_sh[1] + r_sh[1])/2
    cy_dip = cy + 0.005
    artists.extend(stacked_line(ax, l_sh[0], l_sh[1], cx, cy_dip, C_RED_ARM, 3.0))
    artists.extend(stacked_line(ax, r_sh[0], r_sh[1], cx, cy_dip, C_RED_ARM, 3.0))
    artists.extend(stacked_line(ax, cx, cy_dip, cx, cy-0.20, C_HEAD, 2.5))

    def draw_part(topo, color, s=1.0):
        for u,v in topo: 
            artists.extend(stacked_line(ax, f[u,0], f[u,1], f[v,0], f[v,1], color, 2.5*s))

    draw_part(ARM_RIGHT_UPPER, C_ORANGE_DARK); draw_part(ARM_RIGHT_LOWER, C_ORANGE_LIGHT)
    draw_part(ARM_LEFT_UPPER, C_GREEN_ARM); draw_part(ARM_LEFT_LOWER, C_GREEN_ARM)
    f_s = 0.8
    draw_part(LH_THUMB,C_THUMB,f_s); draw_part(LH_INDEX,C_INDEX,f_s); draw_part(LH_MIDDLE,C_MIDDLE,f_s)
    draw_part(LH_RING,C_RING,f_s); draw_part(LH_PINKY,C_PINKY,f_s)
    draw_part(RH_THUMB,C_THUMB_R,f_s); draw_part(RH_INDEX,C_INDEX_R,f_s); draw_part(RH_MIDDLE,C_MIDDLE_R,f_s)
    draw_part(RH_RING,C_RING_R,f_s); draw_part(RH_PINKY,C_PINKY_R,f_s)
    return artists

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
# 3. MAIN (PROFESSIONAL SYNC)
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
    
    pred = np.load(pred_path)
    if isinstance(pred, list): pred = pred[0]
    with open(GT_PKL_PATH, "rb") as f: gt = pickle.load(f)[gt_id]
    gt = np.array(gt)

    # --- PROFESSIONAL SYNC ---
    # Resample the Prediction to match the Ground Truth length exactly.
    print(f"ℹ️  Original Lengths -> Pred: {len(pred)}, GT: {len(gt)}")
    print(f"⚡ Resampling Pred to match GT length ({len(gt)} frames)...")
    
    pred_synced = resample_sequence(pred, len(gt))
    
    # Now they have the exact same length!
    final_len = len(gt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.1)
    
    line = Line2D([0.5, 0.5], [0.05, 0.95], transform=fig.transFigure, color="black", linewidth=2)
    fig.add_artist(line)

    ax1.text(0.5, -0.1, "Predicted Sign Pose", transform=ax1.transAxes, 
             ha='center', fontsize=18, color=C_TEXT, fontname='sans-serif', weight='bold')
    ax2.text(0.5, -0.1, "Ground Truth Pose", transform=ax2.transAxes, 
             ha='center', fontsize=18, color='black', fontname='sans-serif', weight='bold')
    ax2.text(0.5, -0.2, f"Sequence ID: {gt_id}", transform=ax2.transAxes, 
             ha='center', fontsize=10, color='black', fontname='monospace')

    for ax in [ax1, ax2]:
        ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(0,1); ax.set_ylim(1,0)

    arts = []
    def update(i):
        nonlocal arts
        for a in arts: a.remove()
        arts = []
        
        # Simple index matching because lengths are now identical
        arts.extend(draw_skeleton_stacked(ax1, pred_synced[i]))
        arts.extend(draw_skeleton_stacked(ax2, gt[i]))

    ani = FuncAnimation(fig, update, frames=final_len, interval=40)
    out_name = f"compare_{target_name}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    ani.save(out_path, writer="ffmpeg", dpi=300)
    print(f"✅ Video Saved: {out_path} (Perfectly Synced)")

if __name__ == "__main__":
    main()
