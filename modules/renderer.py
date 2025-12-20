import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

C_RED_ARM="#FF0000"; C_GREEN_ARM="#00AA00"; C_ORANGE_DARK="#FF8C00"; C_ORANGE_LIGHT="#FFD700"
C_HEAD="#00AA00"; C_TEXT="#FF0000"
C_THUMB="#FF0000"; C_INDEX="#00AA00"; C_MIDDLE="#0000CD"; C_RING="#FF69B4"; C_PINKY="#FFD700"
C_THUMB_R="#FF0000"; C_INDEX_R="#00AA00"; C_MIDDLE_R="#0000CD"; C_RING_R="#FF69B4"; C_PINKY_R="#FFD700"

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
            scale = limit / dist
            points[i] = wrist + (vec * scale)
    return points

ARM_RIGHT_UPPER=[(43,45)]; ARM_RIGHT_LOWER=[(45,47)]; ARM_LEFT_UPPER=[(44,46)]; ARM_LEFT_LOWER=[(46,48)]
LH_THUMB=[(0,1),(1,2),(2,3),(3,4)]; LH_INDEX=[(0,5),(5,6),(6,7),(7,8)]
LH_MIDDLE=[(0,9),(9,10),(10,11),(11,12)]; LH_RING=[(0,13),(13,14),(14,15),(15,16)]
LH_PINKY=[(0,17),(17,18),(18,19),(19,20)]
RH_THUMB=[(u+21,v+21) for u,v in LH_THUMB]; RH_INDEX=[(u+21,v+21) for u,v in LH_INDEX]
RH_MIDDLE=[(u+21,v+21) for u,v in LH_MIDDLE]; RH_RING=[(u+21,v+21) for u,v in LH_RING]
RH_PINKY=[(u+21,v+21) for u,v in LH_PINKY]

def render_layered(sequences, save_path, width=640, height=640, fps=25, transition=15):
    full = [sequences[0]]
    for i in range(1, len(sequences)):
        a = sequences[i-1]; b = sequences[i]
        t = min(transition, a.shape[0], b.shape[0])
        trans = np.linspace(a[-1], b[0], num=t)
        full.extend([trans, b])
    data = np.concatenate(full, axis=0)

    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=300)
    ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.text(0.05, 0.95, "Predicted Sign Pose", color=C_TEXT, fontsize=12, fontfamily='sans-serif', weight='medium')

    current_artists = []
    def update(i):
        nonlocal current_artists
        for art in current_artists: art.remove()
        current_artists = []
        f = data[i].copy()
        s_width = np.linalg.norm(f[43][:2] - f[44][:2])
        f[0:21] += (f[47] - f[0]); f[0:21] = hard_clamp_hand(f[0:21], f[47], s_width)
        f[21:42] += (f[48] - f[21]); f[21:42] = hard_clamp_hand(f[21:42], f[48], s_width)

        l_sh, r_sh = f[43][:2], f[44][:2]
        cx, cy = (l_sh[0] + r_sh[0])/2, (l_sh[1] + r_sh[1])/2
        cy_dip = cy + 0.005
        current_artists.extend(stacked_line(ax, l_sh[0], l_sh[1], cx, cy_dip, C_RED_ARM, 3.0))
        current_artists.extend(stacked_line(ax, r_sh[0], r_sh[1], cx, cy_dip, C_RED_ARM, 3.0))
        current_artists.extend(stacked_line(ax, cx, cy_dip, cx, cy-0.20, C_HEAD, 2.5))

        def draw_part(topo, color, s=1.0):
            for u,v in topo: current_artists.extend(stacked_line(ax, f[u,0], f[u,1], f[v,0], f[v,1], color, 2.5*s))

        draw_part(ARM_RIGHT_UPPER, C_ORANGE_DARK); draw_part(ARM_RIGHT_LOWER, C_ORANGE_LIGHT)
        draw_part(ARM_LEFT_UPPER, C_GREEN_ARM); draw_part(ARM_LEFT_LOWER, C_GREEN_ARM)
        
        f_s = 0.8
        draw_part(LH_THUMB,C_THUMB,f_s); draw_part(LH_INDEX,C_INDEX,f_s); draw_part(LH_MIDDLE,C_MIDDLE,f_s)
        draw_part(LH_RING,C_RING,f_s); draw_part(LH_PINKY,C_PINKY,f_s)
        draw_part(RH_THUMB,C_THUMB_R,f_s); draw_part(RH_INDEX,C_INDEX_R,f_s); draw_part(RH_MIDDLE,C_MIDDLE_R,f_s)
        draw_part(RH_RING,C_RING_R,f_s); draw_part(RH_PINKY,C_PINKY_R,f_s)

    ani = FuncAnimation(fig, update, frames=len(data), interval=1000/fps)
    ani.save(save_path, writer="ffmpeg", dpi=300)
    plt.close(fig)
