#!/usr/bin/env python3
# pose_viewer.py – render spawn pose exactly like pose_builder (no physics)

import json, math, pyglet, pymunk
from   pymunk import Vec2d
from   pymunk.pyglet_util import DrawOptions
from   pyglet import gl                       # ← for setting clear colour

# ───────── read geometry spec ─────────
spec                = json.load(open("robot_geom.json"))
offs, links         = spec["fixed_offsets"], spec["links"]

# lengths
L = dict(red   = links["rear_red"]["length"],
         blue  = links["rear_blue"]["length"],
         orange= links["rear_orange"]["length"],
         green = links["rear_green"]["length"],
         crank_up = links["rear_crank_upper"]["length"],
         crank_lo = links["rear_crank_lower"]["length"])

# wheel radii & geometry constants (same as pose_builder)
R_UP, R_LO = 38e-3, 30e-3
ATTACH, J_FRAC = 48e-3, 0.24
HIP_SP  = offs["hip_spacing"]
BODY_W  = spec["torso"]["size"]["width"]
TOP_OFF, BOT_OFF = offs["top_offset"], offs["bottom_offset"]
FRONT_OFF = offs["front_offset"]

# base anchors
rear_up  = Vec2d(0, 0.0)
rear_lo  = rear_up + (0.0386, -0.0268)
blue_R   = rear_lo + (0.004 , 0.0113)
front_lo = rear_lo + (HIP_SP, 0)
front_up = front_lo - (rear_lo - rear_up)
blue_R2  = front_lo + (0.004 , 0.0113)

# colour map
LINK_COL = dict(red=(255,0,0,255), blue=(0,0,255,255),
                green=(0,180,0,255), orange=(255,140,0,255),
                crank_up=(120,120,120,255), crank_lo=(120,120,120,255))

# zero-angle offsets (all zero == editor defaults)
mot = dict(rear_u=0, rear_l=0, front_u=0, front_l=0)

# ---------- kinematics identical to pose_builder ----------
def leg(up,lo,right,th_u,th_l):
    t1,t2 = math.radians(135+th_u), math.radians(-45-th_l)
    tip1 = up + Vec2d(R_UP*math.cos(t1), R_UP*math.sin(t1))
    tip2 = lo + Vec2d(R_LO*math.cos(t2), R_LO*math.sin(t2))
    r,d  = L["red"], L["blue"]*(1-J_FRAC)
    D=right-tip1; dist=D.length
    if not(0<dist<r+d and dist>abs(r-d)): return None
    a = (r*r-d*d+dist*dist)/(2*dist)
    h = math.sqrt(max(0,r*r-a*a))
    J1= tip1 + D.normalized()*a - Vec2d(-D.y,D.x).normalized()*h
    LJ= right - (right-J1)/(1-J_FRAC)
    vec,d2 = tip2-LJ,(tip2-LJ).length
    if not(0<d2<2*L["orange"]): return None
    mid=LJ+vec*0.5
    h2=math.sqrt(max(0,L["orange"]**2-(d2*0.5)**2))
    perp=Vec2d(-vec.y,vec.x).normalized()*h2
    OB = mid-perp if (mid-perp).y<(mid+perp).y else mid+perp
    axis=(OB-LJ).length
    if axis==0 or ATTACH>axis: return None
    GS  = LJ + (OB-LJ).normalized()*ATTACH
    return tip1,J1,LJ,OB,tip2,GS

# ---------- build static pymunk scene ----------
space = pymunk.Space(); space.gravity = (0,0)

def add_bar(tag, A:Vec2d, B:Vec2d):
    seg = pymunk.Segment(space.static_body, A, B, 0.002)
    seg.color = LINK_COL.get(tag.split('_')[-1], (0,170,255,255))
    space.add(seg)

# torso outline
top = rear_up.y + TOP_OFF; bot = rear_lo.y - BOT_OFF
front = front_lo.x + FRONT_OFF; rear_edge = front - BODY_W
rect = [(rear_edge,bot),(front,bot),(front,top),(rear_edge,top)]
for a,b in zip(rect, rect[1:]+rect[:1]):
    seg = pymunk.Segment(space.static_body, a, b, 0.003)
    seg.color = (0,0,0,255); space.add(seg)

# anchor dots
for p in (rear_up, rear_lo, blue_R, front_up, front_lo, blue_R2):
    dot = pymunk.Circle(space.static_body, 0.003, p)
    dot.color = (0,0,0,255); space.add(dot)

# legs
kinR = leg(rear_up ,rear_lo ,blue_R ,mot["rear_u"], mot["rear_l"])
kinF = leg(front_up,front_lo,blue_R2,mot["front_u"], mot["front_l"])

def make_leg(tag, kin, up, lo, right):
    tip1,J1,LJ,OB,tip2,_ = kin
    add_bar(f"{tag}_crank_up", up, up + Vec2d(L["crank_up"],0)
            .rotated(math.radians(135+mot[f"{tag}_u"])))
    add_bar(f"{tag}_crank_lo", lo, lo + Vec2d(L["crank_lo"],0)
            .rotated(math.radians(-45-mot[f'{tag}_l'])))
    add_bar(f"{tag}_red",    tip1, J1)
    add_bar(f"{tag}_blue",   LJ,   right)
    add_bar(f"{tag}_orange", LJ,   OB)
    add_bar(f"{tag}_green",  tip2, LJ)

make_leg("rear" ,kinR, rear_up ,rear_lo ,blue_R )
make_leg("front",kinF, front_up,front_lo, blue_R2)

# ---------- viewer ----------
win = pyglet.window.Window(900, 560, "spawn pose (exact preview)")
gl.glClearColor(1.0, 1.0, 1.0, 1.0)     # white background

opt = DrawOptions(); opt.flip_y = True
opt.flags &= ~pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
opt.transform = pymunk.Transform(a=320, d=320, tx=450, ty=260)

@win.event
def on_draw():
    win.clear()
    space.debug_draw(opt)

pyglet.app.run()
