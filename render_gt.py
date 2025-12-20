import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os
import sys
import csv

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GT_PKL_PATH = os.path.join(PROJECT_ROOT, "datasets", "holistic_49_keypoints.pkl")
GLOSS_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "gloss_map.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# --- EXACT COLORS FROM YOUR PIPELINE ---
C_RED_ARM="#FF0000"; C_GREEN_ARM="#00AA00"; C_ORANGE_DARK="#FF8C00"; C_ORANGE_LIGHT="#FFD700"
C_HEAD="#00AA00"; C_TEXT="#FF0000"
C_THUMB="#FF0000"; C_INDEX="#00AA00"; C_MIDDLE="#0000CD"; C_RING="#FF69B4"; C_PINKY="#FFD700"
C_THUMB_R="#FF0000"; C_INDEX_R="#00AA00"; C_MIDDLE_R="#0000CD"; C_RING_R="#FF69B4"; C_PINKY_R="#FFD700"

# --- TOPOLOGY (MATCHING RENDERER.PY) ---
ARM_RIGHT_UPPER=[(43,45)]; ARM_RIGHT_LOWER=[(45,47)]
ARM_LEFT_UPPER=[(44,46)]; ARM_LEFT_LOWER=[(46,48)]

LH_THUMB=[(0,1),(1,2),(2,3),(3,4)]; LH_INDEX=[(0,5),(5,6),(6,7),(7,8)]
LH_MIDDLE=[(0,9),(9,10),(10,11),(11,12)]; LH_RING=[(0,13),(13,14),(14,15),(15,16)]
LH_PINKY=[(0,17),(17,18),(18,19),(19,20)]

RH_THUMB=[(u+21,v+21) for u,v in LH_THUMB]; RH_INDEX=[(u+21,v+21) for u,v in LH_INDEX]
RH_MIDDLE=[(u+21,v+21) for u,v in LH_MIDDLE]; RH_RING=[(u+21,v+21) for u,v in LH_RING]
RH_PINKY=[(u+21,v+21) for u,v in LH_PINKY]

# ==========================================
# 2. DRAWING FUNCTIONS (EXACT COPIES)
# ==========================================
def stacked_line(ax, x0, y0, x1, y1, color, base_lw=2.5, n=4):
    """Draws a line with 'stacked' strokes (copied from renderer.py)."""
    artists = []
    for i in range(n):
        t0 = i / (2 * n); t1 = 1 - t0
        xm0 = x0 + (x1 - x0) * t0; ym0 = y0 + (y1 - y0) * t0
        xm1 = x0 + (x1 - x0) * t1; ym1 = y0 + (y1 - y0) * t1
        lw = base_lw * (0.4 + 0.6 * (i + 1) / n)
        # zorder ensures lines layer correctly
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
            scale = limit / dist
            points[i] = wrist + (vec * scale)
    return points

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
# 3. MAIN RENDERER
# ==========================================
def main():
    target_name = "hello_world" 
    if len(sys.argv) > 1: target_name = sys.argv[1].replace(".npy", "")

    # 1. Load Data
    target_gloss = target_name.split("_")[0].upper()
    gt_id = find_ground_truth_id(target_gloss)
    if gt_id is None: print("❌ GT not found"); return
    
    print(f"🎬 RENDERING GROUND TRUTH (Clone Style): {gt_id}")
    with open(GT_PKL_PATH, "rb") as f: gt = pickle.load(f)[gt_id]
    gt = np.array(gt)

    # 2. Setup Plot (MATCHING DIMENSIONS: 640x640 @ 300 DPI)
    # This matches renderer.py: width=640, height=640, dpi=300
    width, height = 640, 640 
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=300)
    ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Label (Optional: helps distinguish, but keep small/same font)
    ax.text(0.05, 0.95, "Ground Truth Pose", color=C_TEXT, fontsize=12, fontfamily='sans-serif', weight='medium')

    current_artists = []

    def update(i):
        nonlocal current_artists
        for art in current_artists: art.remove()
        current_artists = []
        
        f = gt[i].copy()
        s_width = np.linalg.norm(f[43][:2] - f[44][:2])
        f[0:21] += (f[47] - f[0]); f[0:21] = hard_clamp_hand(f[0:21], f[47], s_width)
        f[21:42] += (f[48] - f[21]); f[21:42] = hard_clamp_hand(f[21:42], f[48], s_width)

        l_sh, r_sh = f[43][:2], f[44][:2]
        cx, cy = (l_sh[0] + r_sh[0])/2, (l_sh[1] + r_sh[1])/2
        cy_dip = cy + 0.005
        
        # Draw Shoulders & Head (Stacked)
        current_artists.extend(stacked_line(ax, l_sh[0], l_sh[1], cx, cy_dip, C_RED_ARM, 3.0))
        current_artists.extend(stacked_line(ax, r_sh[0], r_sh[1], cx, cy_dip, C_RED_ARM, 3.0))
        current_artists.extend(stacked_line(ax, cx, cy_dip, cx, cy-0.20, C_HEAD, 2.5))

        def draw_part(topo, color, s=1.0):
            for u,v in topo: 
                current_artists.extend(stacked_line(ax, f[u,0], f[u,1], f[v,0], f[v,1], color, 2.5*s))

        # Draw Arms
        draw_part(ARM_RIGHT_UPPER, C_ORANGE_DARK); draw_part(ARM_RIGHT_LOWER, C_ORANGE_LIGHT)
        draw_part(ARM_LEFT_UPPER, C_GREEN_ARM); draw_part(ARM_LEFT_LOWER, C_GREEN_ARM)
        
        # Draw Hands
        f_s = 0.8
        draw_part(LH_THUMB,C_THUMB,f_s); draw_part(LH_INDEX,C_INDEX,f_s); draw_part(LH_MIDDLE,C_MIDDLE,f_s)
        draw_part(LH_RING,C_RING,f_s); draw_part(LH_PINKY,C_PINKY,f_s)
        draw_part(RH_THUMB,C_THUMB_R,f_s); draw_part(RH_INDEX,C_INDEX_R,f_s); draw_part(RH_MIDDLE,C_MIDDLE_R,f_s)
        draw_part(RH_RING,C_RING_R,f_s); draw_part(RH_PINKY,C_PINKY_R,f_s)

    # 3. Save
    ani = FuncAnimation(fig, update, frames=len(gt), interval=40)
    out_path = os.path.join(OUTPUT_DIR, f"{target_name}_gt.mp4")
    ani.save(out_path, writer="ffmpeg", dpi=300)
    print(f"✅ Saved Clone GT: {out_path} (640x640 | Stacked Style)")

if __name__ == "__main__":
    main()
