#!/usr/bin/env python3
# simulation.py – rigid joints, coloured links, **servo‑driven distal cranks**
#
# ▸ Only the distal ends of the grey crank links are actuated (grey↔red / grey↔green).
# ▸ Blue link must behave like a rigid *strut*: its right‑hand pivot is fixed to
#   the torso and its orientation is locked, so the left joint (LJ) no longer
#   swings when the leg cranks move.
#
#   We achieve this by adding a **RotaryLimitJoint(…lo==hi)** between the torso
#   and each blue bar after building all pins – effectively freezing the
#   relative angle at its zero‑pose value while still letting the blue bar take
#   shear forces.
#
# Run locally with:  python simulation.py  (requires pymunk, pyglet)
# -----------------------------------------------------------------------------
import json, math, random, pymunk, pyglet
from   pymunk import Vec2d
from   pymunk.pyglet_util import DrawOptions

DT, SIM_TIME   = 1/240, 20.0
GRAVITY        = (0, -9.81)
STALL_TORQUE   = 1.18 * 2.4
Y_OFFSET       = 0.12
BLUE, GREY     = (0,90,240,255), (180,180,180,255)
GROUP          = 12
RAND_PERIOD    = 0.5
RAND_ANGLE_MAX = math.radians(60)
PURPLE     = (140,  0, 140, 255)    # deep purple
FOOT_R     = 25.2e-3 * 0.5          # radius in metres
FOOT_MASS  = 0.015                  # ≈ 15 g rubber cap
FOOT_FRICT = 4.0                    # high friction = sticky
FOOT_REST  = 0.05                   # low bounce
# ─ striped ground: alternating colours every 6 in ─
STRIPE_LEN  = 0.1524   # 6 in  → 0.1524 m  (was 0.3048)
STRIPE_COLS = [
    (225,225,225,255),   # very light grey
    (190,190,190,255),   # mid grey
]                       # add more tuples if you want a longer cycle

LINK_COL = {
    "red":   (255,0,0,255),
    "blue":  (0,0,255,255),
    "green": (0,180,0,255),
    "orange":(255,140,0,255),
    "upper": (120,120,120,255),
    "lower": (120,120,120,255),
}

rod_I = lambda m,L: m*L*L/12.0
v     = lambda obj: Vec2d(*obj)

spec           = json.load(open("robot_geom.json"))
links, joints  = spec["links"], spec["joints"]
torso_spec     = spec["torso"];  offs = spec["fixed_offsets"]
LEN = {n:d["length"] for n,d in links.items()}
Lred, Lblue, Lorange = (LEN[k] for k in ("rear_red","rear_blue","rear_orange"))
R_UP,R_LO   = 38e-3,30e-3
ATTACH,J_F  = 48e-3,0.24
HIP_SP,TOP,BOT,FRONT = (offs[k] for k in ("hip_spacing","top_offset","bottom_offset","front_offset"))

space              = pymunk.Space(); space.gravity = GRAVITY
space.damping      = 0.99; space.iterations = 60
bodies             = {}
shape_filter       = pymunk.ShapeFilter(group=GROUP)

# ─ torso (dynamic) ─
tor_w,tor_h  = torso_spec["size"].values()
rear_up      = Vec2d(0,Y_OFFSET)
rear_lo      = rear_up + (0.0386,-0.0268)
front_lo     = rear_lo + (HIP_SP,0)
front_up     = front_lo - (rear_lo-rear_up)
front_x      = front_lo.x + FRONT; rear_x = front_x - tor_w
centre       = Vec2d((front_x+rear_x)/2,(rear_up.y+TOP + rear_lo.y-BOT)/2)

torso = pymunk.Body(torso_spec["mass"], torso_spec["inertia_zz"])
torso.position = centre
poly = pymunk.Poly.create_box(torso,(tor_w,tor_h)); poly.color = BLUE; poly.filter = shape_filter
space.add(torso,poly); bodies["torso"] = torso

# ground
seg0,seg1 = (v(p) for p in spec["ground"]["segment"])
thick      = spec["ground"]["thickness"]   
start_x, end_x = seg0.x, seg1.x
x, idx = start_x, 0
while x < end_x:
    nxt = min(x + STRIPE_LEN, end_x)
    seg = pymunk.Segment(space.static_body, (x, seg0.y), (nxt, seg1.y), thick)
    seg.color = STRIPE_COLS[idx % len(STRIPE_COLS)]
    space.add(seg)
    x = nxt; idx += 1


# FK helper
def leg(up,lo,right,th_u,th_l):
    t1,t2 = math.radians(135+th_u), math.radians(-45-th_l)
    tip1  = up + Vec2d(R_UP*math.cos(t1), R_UP*math.sin(t1))
    tip2  = lo + Vec2d(R_LO*math.cos(t2), R_LO*math.sin(t2))
    r,d   = Lred, Lblue*(1-J_F)
    D = right - tip1; dist = D.length
    if not(0<dist<r+d and dist>abs(r-d)): return None
    a = (r*r-d*d+dist*dist)/(2*dist); h = math.sqrt(max(0,r*r-a*a))
    J1 = tip1 + D.normalized()*a - Vec2d(-D.y,D.x).normalized()*h
    LJ = right - (right-J1)/(1-J_F)
    vec = tip2-LJ
    if not(0<vec.length<2*Lorange): return None
    mid = LJ+vec*0.5; h2 = math.sqrt(max(0,Lorange**2-(vec.length*0.5)**2))
    perp = Vec2d(-vec.y,vec.x).normalized()*h2
    OB   = mid-perp if (mid-perp).y<(mid+perp).y else mid+perp
    if (OB-LJ).length<ATTACH: return None
    GS = LJ + (OB-LJ).normalized()*ATTACH
    return tip1,J1,LJ,OB,tip2,GS


def add_rubber_foot(parent_name: str, world_tip: Vec2d):
    """
    Add a purple, sticky foot that is rigidly part of the *orange-bar body*.
    The offset must be expressed in *body-local* coordinates → rotate it back.
    """
    parent = bodies[parent_name]
    # vector from bar origin to tip, then rotate into bar-local frame
    local = (world_tip - parent.position).rotated(-parent.angle)

    foot = pymunk.Circle(parent, FOOT_R, local)
    foot.friction    = FOOT_FRICT     # sticky rubber
    foot.elasticity  = FOOT_REST
    foot.color       = PURPLE
    foot.filter      = shape_filter   # same collision group
    space.add(foot)

# zero pose
zero = {j["child"]:j["zero_angle"] for j in joints if j["type"]=="revolute" and j["parent"]=="torso"}
rear_u  = math.degrees(zero["rear_crank_upper"])-135; rear_l  = -math.degrees(zero["rear_crank_lower"])-45
front_u = math.degrees(zero["front_crank_upper"])-135; front_l = -math.degrees(zero["front_crank_lower"])-45
blueR  = rear_lo + (0.004,0.0113); blueF = front_lo + (0.004,0.0113)
kinR   = leg(rear_up,rear_lo,blueR,rear_u,rear_l); kinF = leg(front_up,front_lo,blueF,front_u,front_l)
if not(kinR and kinF): raise SystemExit("FK failed – zero pose impossible")

# bar builder

def make_bar(name,A,B):
    axis=B-A; m=links[name].get("mass",0.02)
    body=pymunk.Body(m,rod_I(m,axis.length)); body.position,body.angle=A,axis.angle
    seg=pymunk.Segment(body,(0,0),(axis.length,0),0.002)
    key=name.split('_')[-1]; seg.color=LINK_COL.get(name,LINK_COL.get(key,BLUE))
    seg.friction=1.0; seg.filter=shape_filter
    space.add(body,seg); bodies[name]=body

# build bars
(t1R,J1R,LJR,OBR,t2R,GSR)=kinR; (t1F,J1F,LJF,OBF,t2F,GSF)=kinF
len_up,len_lo=LEN["rear_crank_upper"],LEN["rear_crank_lower"]
make_bar("rear_crank_upper",rear_up,rear_up+Vec2d(len_up,0).rotated(math.radians(135+rear_u)))
make_bar("rear_crank_lower",rear_lo,rear_lo+Vec2d(len_lo,0).rotated(math.radians(-45-rear_l)))
make_bar("rear_red",t1R,J1R); make_bar("rear_blue",LJR,blueR)
make_bar("rear_orange",LJR,OBR); make_bar("rear_green",t2R,GSR)
make_bar("front_crank_upper",front_up,front_up+Vec2d(len_up,0).rotated(math.radians(135+front_u)))
make_bar("front_crank_lower",front_lo,front_lo+Vec2d(len_lo,0).rotated(math.radians(-45-front_l)))
make_bar("front_red",t1F,J1F); make_bar("front_blue",LJF,blueF)
make_bar("front_orange",LJF,OBF); make_bar("front_green",t2F,GSF)
add_rubber_foot("rear_orange",  OBR)
add_rubber_foot("front_orange", OBF)

# pins & limits (from JSON) – motors skipped

def pin(a,b,W):
    pj=pymunk.PivotJoint(a,b,W); pj.max_force=8_000; pj.error_bias=0; pj.max_bias=0; return pj

for jd in joints:
    a,b=bodies[jd["parent"]],bodies[jd["child"]]; W=v(jd["anchor"])+Vec2d(0,Y_OFFSET)
    space.add(pin(a,b,W))
    if jd["type"]=="revolute":
        lo,hi=jd.get("limits",[-math.pi,math.pi]); rl=pymunk.RotaryLimitJoint(a,b,lo,hi)
        rl.max_force=8_000; rl.error_bias=0; rl.max_bias=0; space.add(rl)

# ───── anchor blue bars at *both* ends so they stay rigid w.r.t torso ─────
# We already have one pin (from JSON) at the interior “blue_orange” point.
# Add a **second pin** at the far right‑hand tip (blueR / blueF).  Two pins
# fully constrain translation + rotation, so we can drop the rotary limit.
for name, tip in (("rear_blue", blueR), ("front_blue", blueF)):
    space.add(pin(torso, bodies[name], tip))

# ───── our servo motors at distal crank joints ───── at distal crank joints
active_motors={}
DRIVE_PAIRS=[
    ("rear_crank_upper","rear_red"),
    ("rear_crank_lower","rear_green"),
    ("front_crank_upper","front_red"),
    ("front_crank_lower","front_green"),
]
for A,B in DRIVE_PAIRS:
    mot=pymunk.SimpleMotor(bodies[A],bodies[B],0); mot.max_force=STALL_TORQUE
    space.add(mot); active_motors[f"{A}->{B}"]=mot

def shuffle_motors(_):
    for m in active_motors.values():
        m.rate=random.uniform(-RAND_ANGLE_MAX,RAND_ANGLE_MAX)
pyglet.clock.schedule_interval(shuffle_motors,RAND_PERIOD)

# viewer
win=pyglet.window.Window(820,540,"assembled robot – blue locked, distal motors")
opt=DrawOptions(); opt.flip_y=True; opt.flags=pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
opt.transform=pymunk.Transform(a=320,d=320,tx=450,ty=260)
@win.event
def on_draw():
    win.clear(); space.debug_draw(opt)

def tick(_): space.step(DT)
_tid=pyglet.clock.schedule_interval(tick,DT)
pyglet.clock.schedule_once(lambda _:(pyglet.clock.unschedule(_tid),win.close()),SIM_TIME)
pyglet.app.run()
