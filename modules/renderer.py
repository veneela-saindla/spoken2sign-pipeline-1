import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

BODY  = [(42,43),(42,44),(43,44),(43,45),(45,47),(44,46),(46,48)]
ATT   = [(47,0), (48,21)]
LH    = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20)]
RH    = [(u+21, v+21) for u,v in LH]

def render_layered(sequences, save_path, width=640, height=640, fps=25, transition=15):
    full = [sequences[0]]
    for i in range(1, len(sequences)):
        a = sequences[i-1]
        b = sequences[i]
        t = min(transition, a.shape[0], b.shape[0])
        trans = np.linspace(a[-1], b[0], num=t)
        full.extend([trans, b])

    data = np.concatenate(full, axis=0)

    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(0,1)
    ax.set_ylim(1,0)

    C_BODY = "#9E9E9E"
    C_DOTS = "#1E88E5"
    C_HAND = "#F44336"

    body_lines = [ax.plot([], [], C_BODY, lw=3)[0] for _ in BODY]
    att_lines  = [ax.plot([], [], C_BODY, lw=3)[0] for _ in ATT]
    lh_lines   = [ax.plot([], [], C_HAND, lw=3)[0] for _ in LH]
    rh_lines   = [ax.plot([], [], C_HAND, lw=3)[0] for _ in RH]
    dots       = ax.scatter([], [], s=80, c=C_DOTS)

    def update(i):
        f = data[i]

        for ln, (u,v) in zip(body_lines, BODY):
            ln.set_data([f[u,0], f[v,0]], [f[u,1], f[v,1]])

        for ln, (u,v) in zip(att_lines, ATT):
            ln.set_data([f[u,0], f[v,0]], [f[u,1], f[v,1]])

        for ln, (u,v) in zip(lh_lines, LH):
            ln.set_data([f[u,0], f[v,0]], [f[u,1], f[v,1]])

        for ln, (u,v) in zip(rh_lines, RH):
            ln.set_data([f[u,0], f[v,0]], [f[u,1], f[v,1]])

        dots.set_offsets(f[42:49,:2])

    ani = FuncAnimation(fig, update, frames=len(data), interval=1000/fps)
    ani.save(save_path, writer="ffmpeg", dpi=100)
    plt.close(fig)
