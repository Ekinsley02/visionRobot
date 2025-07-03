#!/usr/bin/env python3
# slider_dump_points.py
import json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.text  as mtext
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

# ───────────── CONSTANTS (metres) ──────────────
radiusUpper   = 38.0 / 1000
radiusLower   = 30.0 / 1000
link1_length  = 31.0 / 1000         # red
link2_length  = 104.0 / 1000        # blue
bar_length    = 124.0 / 1000        # orange
attach_dist   = 48.0 / 1000         # green‐fix on orange
joint_ratio   = 0.24                # blue joint fraction (from right)

BODY_W        = 304.8 / 1000
TOP_OFFSET    = 31.9  / 1000
BOTTOM_OFFSET = 25.8  / 1000
FRONT_OFFSET  = 42.5  / 1000
OFFSET_X      = 180.0 / 1000        # hip spacing

# ───────────── ANCHOR POINTS ──────────────
motor1_A  = np.array([0.0, 0.0])
motor2_A  = motor1_A + np.array([0.0386, -0.0268])
right_B   = motor2_A + np.array([4.0, 11.3]) / 1000

motor2_A2 = motor2_A + np.array([OFFSET_X, 0.0])
motor1_A2 = motor2_A2 - (motor2_A - motor1_A)
right_B2  = motor2_A2 + np.array([4.0, 11.3]) / 1000

# ───────────── FIGURE SET-UP ──────────────
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.30)
ax.set_aspect('equal')
ax.set_xlim(-0.15, OFFSET_X + 0.35)
ax.set_ylim(-0.30, 0.15)
ax.set_title('Two-Leg Simulator – press  d  to dump linkage points')
ax.axis('off')

# banner reminder
banner = mtext.Annotation("Press  **d**  to dump current linkage points",
                          xy=(0.5, 1.02), xycoords='axes fraction',
                          ha='center', va='bottom', size=9,
                          bbox=dict(fc="lightgrey", ec="none", pad=3))
ax.add_artist(banner)

# body outline
body_patch = Rectangle((0, 0), 0, 0, facecolor='none',
                       edgecolor='black', linewidth=2, zorder=0)
ax.add_patch(body_patch)

# anchor dots
for p in (motor1_A, motor2_A, right_B, motor1_A2, motor2_A2, right_B2):
    ax.plot(*p, 'ko')

# rear-leg artists
l1 ,= ax.plot([], [], 'r-' , lw=3)
l2 ,= ax.plot([], [], 'b-' , lw=3)
l3 ,= ax.plot([], [], 'g-' , lw=3)
l4 ,= ax.plot([], [], color='orange', lw=3)
tip1,= ax.plot([], [], 'go')
tip2,= ax.plot([], [], 'go')

# front-leg artists
l1b ,= ax.plot([], [], 'r--', lw=3)
l2b ,= ax.plot([], [], 'b--', lw=3)
l3b ,= ax.plot([], [], 'g--', lw=3)
l4b ,= ax.plot([], [], color='orange', ls='--', lw=3)
tip1b,= ax.plot([], [], 'go')
tip2b,= ax.plot([], [], 'go')

# ───────────── SLIDERS ──────────────
ax_s1  = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_s2  = plt.axes([0.15, 0.15, 0.65, 0.03])
s1  = Slider(ax_s1,  'Rear M1°', 0, 40, valinit=0)
s2  = Slider(ax_s2,  'Rear M2°', 0, 40, valinit=0)

ax_s1b = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_s2b = plt.axes([0.15, 0.05, 0.65, 0.03])
s1b = Slider(ax_s1b, 'Front M1°', 0, 40, valinit=0)
s2b = Slider(ax_s2b, 'Front M2°', 0, 40, valinit=0)

# ───────────── KINEMATICS ──────────────
def solve_leg(m1, m2, right_anchor, th1_deg, th2_deg):
    t1 = np.radians(135 + th1_deg)
    t2 = np.radians(-45 - th2_deg)
    tip1 = m1 + radiusUpper * np.array([np.cos(t1), np.sin(t1)])
    tip2 = m2 + radiusLower * np.array([np.cos(t2), np.sin(t2)])

    r, d = link1_length, link2_length * (1 - joint_ratio)
    D = right_anchor - tip1
    dist = np.linalg.norm(D)
    if dist > r + d or dist < abs(r - d) or dist == 0 and r == d:
        return None
    a = (r**2 - d**2 + dist**2)/(2*dist)
    h = np.sqrt(abs(r**2 - a**2))
    mid = tip1 + a*D/dist
    perp = np.array([-D[1], D[0]])/dist
    joint1 = mid - h*perp

    full = (right_anchor - joint1)/(1-joint_ratio)
    full = full/np.linalg.norm(full)*link2_length
    left_joint = right_anchor - full

    vec_lo = tip2 - left_joint
    d_lo = np.linalg.norm(vec_lo)
    if d_lo == 0 or d_lo > 2*bar_length:
        return None
    a2  = d_lo/2
    h2  = np.sqrt(bar_length**2 - a2**2)
    base = left_joint + a2*vec_lo/d_lo
    perp2= np.array([-vec_lo[1], vec_lo[0]])/d_lo
    o1,o2 = base + h2*perp2, base - h2*perp2
    orange_bot = o1 if o1[1] < o2[1] else o2

    ax_len = np.linalg.norm(orange_bot-left_joint)
    if ax_len == 0 or attach_dist > ax_len:
        return None
    dir_or = (orange_bot-left_joint)/ax_len
    green_start = left_joint + dir_or*attach_dist

    return tip1, joint1, left_joint, orange_bot, tip2, green_start

# ───────────── POINT DUMP HELPER ──────────────
def collect_points():
    rear = solve_leg(motor1_A ,motor2_A ,right_B ,
                     s1.val  ,s2.val)
    front= solve_leg(motor1_A2,motor2_A2,right_B2,
                     s1b.val ,s2b.val)
    if not rear or not front:
        return None
    t1,j1,lj,ob,t2,gs = rear
    ft1,fj1,flj,fob,ft2,fgs = front
    return {
        "rear":  {"joint1":j1.tolist(),
                  "blue_orange":lj.tolist(),
                  "green_attach":gs.tolist()},
        "front": {"joint1":fj1.tolist(),
                  "blue_orange":flj.tolist(),
                  "green_attach":fgs.tolist()}
    }

# ───────────── DRAW UPDATE ──────────────
def draw_leg(arts, pts, right_anchor):
    l1_,l2_,l3_,l4_,tA_,tB_ = arts
    tipA,J1,LJ,OB,tipB,GS = pts
    l1_.set_data([tipA[0],J1[0]],[tipA[1],J1[1]])
    l2_.set_data([LJ[0],right_anchor[0]],[LJ[1],right_anchor[1]])
    l3_.set_data([GS[0],tipB[0]],[GS[1],tipB[1]])
    l4_.set_data([LJ[0],OB[0]],[LJ[1],OB[1]])
    tA_.set_data([tipA[0]],[tipA[1]])
    tB_.set_data([tipB[0]],[tipB[1]])

def update(_=None):
    body_top    = motor1_A[1] + TOP_OFFSET
    body_bottom = motor2_A[1] - BOTTOM_OFFSET
    body_h      = body_top - body_bottom
    front_edge  = motor2_A2[0] + FRONT_OFFSET
    rear_edge   = front_edge - BODY_W
    body_patch.set_xy((rear_edge, body_bottom))
    body_patch.set_width(BODY_W); body_patch.set_height(body_h)

    p1 = solve_leg(motor1_A ,motor2_A ,right_B , s1.val ,s2.val)
    p2 = solve_leg(motor1_A2,motor2_A2,right_B2, s1b.val,s2b.val)
    if p1: draw_leg((l1,l2,l3,l4,tip1,tip2), p1, right_B)
    if p2: draw_leg((l1b,l2b,l3b,l4b,tip1b,tip2b), p2, right_B2)
    fig.canvas.draw_idle()

for sl in (s1,s2,s1b,s2b): sl.on_changed(update)
update()

# ───────────── KEY HANDLER ──────────────
def on_key(event):
    if event.key.lower() == 'd':
        pts = collect_points()
        if pts is None:
            print("⚠ Linkage impossible – no dump.")
            return
        with open("robot_points.json","w") as f:
            json.dump(pts,f,indent=2)
        print("\nDumped linkage anchors to robot_points.json")
        print(json.dumps(pts,indent=2))

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
