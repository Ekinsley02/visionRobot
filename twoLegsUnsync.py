import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ─────────────── CONSTANTS (metres) ───────────────
radiusUpper   = 38.0  / 1000
radiusLower   = 30.0  / 1000
link1_length  = 31   / 1000        # red
link2_length  = 104  / 1000        # blue
bar_length    = 124  / 1000        # orange
attach_dist   = 48.0 / 1000        # green
joint_ratio   = 0.24               # blue-bar split

OFFSET_X      = 180.0 / 1000       # 180 mm between the two motor-2 anchors

# ─────────────── FIRST-LEG ANCHORS ────────────────
motor1_A  = np.array([0.0, 0.0])
motor2_A  = motor1_A + np.array([0.0386, -0.0268])
right_B   = motor2_A + np.array([4.0, 11.3]) / 1000

# ─────────────── SECOND-LEG ANCHORS ───────────────
motor2_A2 = motor2_A + np.array([OFFSET_X, 0.0])
motor1_A2 = motor2_A2 - (motor2_A - motor1_A)         # keep same offset
right_B2  = motor2_A2 + np.array([4.0, 11.3]) / 1000

# ─────────────── FIGURE & ARTISTS ────────────────
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.30)                      # extra room for 4 sliders
ax.set_aspect('equal')
ax.set_xlim(-0.10, OFFSET_X + 0.22)
ax.set_ylim(-0.22, 0.15)
ax.set_title('Two-Leg 2-Link System with Independent Sliders')
ax.axis('off')

# anchor dots
for p in (motor1_A, motor2_A, right_B, motor1_A2, motor2_A2, right_B2):
    ax.plot(*p, 'ko' if p is right_B or p is right_B2 else 'ko')

# leg-1 artists
l1,  = ax.plot([], [], 'r-',  lw=3)
l2,  = ax.plot([], [], 'b-',  lw=3)
l3,  = ax.plot([], [], 'g-',  lw=3)
l4,  = ax.plot([], [], color='orange', lw=3)
tip1, = ax.plot([], [], 'go')
tip2, = ax.plot([], [], 'go')

# leg-2 artists (dashed for contrast)
l1b,  = ax.plot([], [], 'r--',  lw=3)
l2b,  = ax.plot([], [], 'b--',  lw=3)
l3b,  = ax.plot([], [], 'g--',  lw=3)
l4b,  = ax.plot([], [], color='orange', ls='--', lw=3)
tip1b,= ax.plot([], [], 'go')
tip2b,= ax.plot([], [], 'go')

# ─────────────── SLIDERS ────────────────
# 1st leg (left on screen)
ax_s1  = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_s2  = plt.axes([0.15, 0.15, 0.65, 0.03])
s1  = Slider(ax_s1,  'Leg-1 M1 °', 0, 40, valinit=0)
s2  = Slider(ax_s2,  'Leg-1 M2 °', 0, 40, valinit=0)
# 2nd leg (right on screen)
ax_s1b = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_s2b = plt.axes([0.15, 0.05, 0.65, 0.03])
s1b = Slider(ax_s1b, 'Leg-2 M1 °', 0, 40, valinit=0)
s2b = Slider(ax_s2b, 'Leg-2 M2 °', 0, 40, valinit=0)

# ─────────────── KINEMATICS ───────────────
def solve_leg(m1, m2, right_anchor, th1_deg, th2_deg):
    """Return (tip1, joint1, left_joint, orange_bot, tip2, green_start)."""
    t1 = np.radians(135 + th1_deg)   # your empirical offsets
    t2 = np.radians(-45 - th2_deg)
    tip1 = m1 + radiusUpper * np.array([np.cos(t1), np.sin(t1)])
    tip2 = m2 + radiusLower * np.array([np.cos(t2), np.sin(t2)])

    # red & blue linkage
    r, d = link1_length, link2_length * (1 - joint_ratio)
    D = right_anchor - tip1
    dist = np.linalg.norm(D)
    if dist > r + d or dist < abs(r - d) or (dist == 0 and r == d):
        return None
    a = (r**2 - d**2 + dist**2) / (2 * dist)
    h = np.sqrt(abs(r**2 - a**2))
    midpoint = tip1 + a * D / dist
    perp = np.array([-D[1], D[0]]) / dist
    joint1 = midpoint - h * perp

    full_vec = (right_anchor - joint1) / (1 - joint_ratio)
    full_vec = full_vec / np.linalg.norm(full_vec) * link2_length
    left_joint = right_anchor - full_vec

    # orange bar bottom
    vec_lo = tip2 - left_joint
    d_lo = np.linalg.norm(vec_lo)
    if d_lo == 0 or d_lo > 2 * bar_length:
        return None
    a2 = d_lo / 2
    h2 = np.sqrt(bar_length**2 - a2**2)
    base = left_joint + a2 * vec_lo / d_lo
    perp2 = np.array([-vec_lo[1], vec_lo[0]]) / d_lo
    cand1, cand2 = base + h2 * perp2, base - h2 * perp2
    orange_bottom = cand1 if cand1[1] < cand2[1] else cand2

    axis_len = np.linalg.norm(orange_bottom - left_joint)
    if axis_len == 0 or attach_dist > axis_len:
        return None
    dir_orange = (orange_bottom - left_joint) / axis_len
    green_start = left_joint + dir_orange * attach_dist

    return tip1, joint1, left_joint, orange_bottom, tip2, green_start

# ─────────────── DRAW / UPDATE ───────────────
def draw_leg(artists, points, right_anchor):
    l1_, l2_, l3_, l4_, tipA_, tipB_ = artists
    tipA, J1, LJ, OB, tipB, GS = points
    l1_.set_data([tipA[0], J1[0]], [tipA[1], J1[1]])
    l2_.set_data([LJ[0],  right_anchor[0]], [LJ[1], right_anchor[1]])
    l3_.set_data([GS[0],  tipB[0]], [GS[1], tipB[1]])
    l4_.set_data([LJ[0],  OB[0]],  [LJ[1],  OB[1]])
    tipA_.set_data([tipA[0]], [tipA[1]])
    tipB_.set_data([tipB[0]], [tipB[1]])

def update(_):
    # leg-1
    pts1 = solve_leg(motor1_A, motor2_A, right_B, s1.val,  s2.val)
    if pts1: draw_leg((l1,l2,l3,l4,tip1,tip2), pts1, right_B)

    # leg-2
    pts2 = solve_leg(motor1_A2, motor2_A2, right_B2, s1b.val, s2b.val)
    if pts2: draw_leg((l1b,l2b,l3b,l4b,tip1b,tip2b), pts2, right_B2)

    fig.canvas.draw_idle()

for sl in (s1, s2, s1b, s2b):
    sl.on_changed(update)

update(None)
plt.show()
