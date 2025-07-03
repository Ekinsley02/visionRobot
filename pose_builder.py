#!/usr/bin/env python3
# pose_builder.py — interactive two-leg pose editor → robot_geom.json
# 2025-07-02 (v2) :   dynamic zero-angles for cranks

import json, math, pyglet, pymunk
from   pymunk import Vec2d
from   pymunk.pyglet_util import DrawOptions
# ───────────────── CONSTANTS ─────────────────────────────────────────
R_UP, R_LO = 38e-3, 30e-3
L = dict(red=31e-3,  blue=104e-3,  orange=124e-3, green=48e-3,
         crank_up=38e-3, crank_lo=30e-3)
ATTACH, J_FRAC       = 48e-3, 0.24
HIP_SP, BODY_W       = 0.180, 0.3048
TOP_OFF, BOT_OFF     = 0.0319, 0.0258
FRONT_OFF            = 0.0425
# ───────────────── anchors ───────────────────────────────────────────
rear_up  = Vec2d(0, 0)
rear_lo  = rear_up + (0.0386, -0.0268)
blue_R   = rear_lo + (0.004,  0.0113)
front_lo = rear_lo + (HIP_SP, 0)
front_up = front_lo - (rear_lo - rear_up)
blue_R2  = front_lo + (0.004, 0.0113)
ANCHORS  = (rear_up, rear_lo, blue_R, front_up, front_lo, blue_R2)
# ───────────────── slider state ──────────────────────────────────────
mot = dict(rear_u=0, rear_l=0, front_u=0, front_l=0)
# ───────────────── preview world ─────────────────────────────────────
space = pymunk.Space(); space.gravity = (0,0)
bars: dict[str,pymunk.Body] = {}
LINK_COL = dict(red=(255,0,0,255), blue=(0,0,255,255),
                green=(0,180,0,255), orange=(255,140,0,255),
                crank_up=(120,120,120,255), crank_lo=(120,120,120,255))
# torso outline -------------------------------------------------------
def draw_torso():
    top, bot  = rear_up.y + TOP_OFF, rear_lo.y - BOT_OFF
    front     = front_lo.x + FRONT_OFF
    rear_edge = front - BODY_W
    quad = [(rear_edge,bot),(front,bot),(front,top),(rear_edge,top)]
    for a,b in zip(quad, quad[1:]+quad[:1]):
        s = pymunk.Segment(space.static_body, a, b, 0.003)
        s.color=(0,0,0,255); space.add(s)
draw_torso()
# anchor dots ---------------------------------------------------------
for p in ANCHORS:
    circ = pymunk.Circle(space.static_body, 0.003, p)
    circ.color=(0,0,0,255); space.add(circ)
# helpers -------------------------------------------------------------
def add_bar(tag,Leng,pos,ang):
    m=1e-3; I=m*Leng*Leng/12
    b=pymunk.Body(m,I,body_type=pymunk.Body.KINEMATIC)
    b.position,b.angle=pos,ang
    seg=pymunk.Segment(b,(0,0),(Leng,0),0.002)
    seg.color=LINK_COL.get(tag.split('_')[-1],(0,170,255,255))
    space.add(b,seg); bars[tag]=b
def clear_bars():
    for b in list(bars.values()):
        for s in b.shapes: space.remove(s)
        space.remove(b)
    bars.clear()
# leg kinematics (unchanged) -----------------------------------------
def leg(up,lo,right,th_u,th_l):
    t1,t2 = math.radians(135+th_u), math.radians(-45-th_l)
    tip1  = up + Vec2d(R_UP*math.cos(t1), R_UP*math.sin(t1))
    tip2  = lo + Vec2d(R_LO*math.cos(t2), R_LO*math.sin(t2))
    r,d   = L['red'], L['blue']*(1-J_FRAC)
    D=right-tip1; dist=D.length
    if not(0<dist<r+d and dist>abs(r-d)): return None
    a=(r*r-d*d+dist*dist)/(2*dist)
    h=math.sqrt(max(0,r*r-a*a))
    J1 = tip1 + D.normalized()*a - Vec2d(-D.y,D.x).normalized()*h
    LJ = right - (right-J1)/(1-J_FRAC)
    vec,d2 = tip2-LJ,(tip2-LJ).length
    if not(0<d2<2*L['orange']): return None
    mid=LJ+vec*0.5
    h2=math.sqrt(max(0,L['orange']**2-(d2*0.5)**2))
    perp=Vec2d(-vec.y,vec.x).normalized()*h2
    OB = mid-perp if (mid-perp).y<(mid+perp).y else mid+perp
    axis=(OB-LJ).length
    if axis==0 or ATTACH>axis: return None
    GS=LJ+(OB-LJ).normalized()*ATTACH
    return tip1,J1,LJ,OB,tip2,GS
# build preview -------------------------------------------------------
def rebuild():
    clear_bars()
    kinR=leg(rear_up ,rear_lo ,blue_R ,mot['rear_u'],mot['rear_l'])
    kinF=leg(front_up,front_lo,blue_R2,mot['front_u'],mot['front_l'])
    if not(kinR and kinF): return
    def mk(tag,kin,anc):
        tip1,J1,LJ,OB,tip2,_=kin
        up,lo,right=anc
        add_bar(f'{tag}_crank_up',L['crank_up'],up, math.radians(135+mot[f'{tag}_u']))
        add_bar(f'{tag}_crank_lo',L['crank_lo'],lo, math.radians(-45-mot[f'{tag}_l']))
        add_bar(f'{tag}_red',   L['red'],   tip1,(J1-tip1).angle)
        add_bar(f'{tag}_blue',  L['blue'],  LJ,  (right-LJ).angle)
        add_bar(f'{tag}_orange',L['orange'],LJ,  (OB-LJ).angle)
        add_bar(f'{tag}_green', L['green'], tip2,(LJ-tip2).angle)
    mk('rear' ,kinR,(rear_up ,rear_lo ,blue_R ))
    mk('front',kinF,(front_up,front_lo,blue_R2))
rebuild()
# ───────── pyglet UI ─────────────────────────────────────────────────
win=pyglet.window.Window(900,560,"Pose-builder – drag sliders, d = dump")
pyglet.gl.glClearColor(1,1,1,1)
draw=DrawOptions(); draw.flip_y=True
draw.transform=pymunk.Transform(a=320,d=320,tx=450,ty=260)
info=pyglet.text.Label("",x=10,y=10,anchor_x='left',anchor_y='bottom',
                       color=(0,0,0,255))
def refresh_lbl(): info.text=f"angles (deg): {mot}"
refresh_lbl()
# sliders -------------------------------------------------------------
class Slider:
    W,H,R=260,6,8
    def __init__(s,key,label,y):
        s.key,y=key,y; s.x0=40
        s.bar = pyglet.shapes.Rectangle(s.x0,y,s.W,s.H,color=(210,210,210))
        s.knob= pyglet.shapes.Circle(0,y+s.H/2,s.R,color=(0,0,255))
        s.lab = pyglet.text.Label(label,x=s.x0-10,y=y+2,anchor_x='right',
                                  anchor_y='center',color=(0,0,0,255))
        s.drag=False; s.update()
    def update(s): s.knob.x=s.x0+mot[s.key]/40*s.W
    def draw  (s): s.bar.draw(); s.knob.draw(); s.lab.draw()
    def hit(s,x,y): return (x-s.knob.x)**2+(y-s.knob.y)**2<s.R**2
    def on_mouse_press(s,x,y,btn,mods): s.drag=s.hit(x,y)
    def on_mouse_release(s,*_): s.drag=False
    def on_mouse_drag(s,x,y,dx,dy,btn,mods):
        if not s.drag: return
        x=max(s.x0,min(s.x0+s.W,x))
        mot[s.key]=round((x-s.x0)/s.W*40,1)
        s.update(); rebuild(); refresh_lbl()
sliders=[Slider('rear_u','Rear Upper', 60),
         Slider('rear_l','Rear Lower',110),
         Slider('front_u','Front Upper',160),
         Slider('front_l','Front Lower',210)]
for sld in sliders: win.push_handlers(sld)
@win.event
def on_draw():
    win.clear(); space.debug_draw(draw)
    for sld in sliders: sld.draw(); info.draw()
# keyboard ------------------------------------------------------------
sel=0
@win.event
def on_key_press(sym,_):
    global sel
    if pyglet.window.key._1<=sym<=pyglet.window.key._4:
        sel=sym-pyglet.window.key._1
    elif sym in (pyglet.window.key.LEFT,pyglet.window.key.RIGHT):
        k=list(mot.keys())[sel]
        mot[k]=max(0,min(40,mot[k]+(1 if sym==pyglet.window.key.RIGHT else -1)))
        sliders[sel].update(); rebuild(); refresh_lbl()
    elif sym==pyglet.window.key.D: dump_json()
# ───────── dump helper ───────────────────────────────────────────────
def dump_json():
    kinR=leg(rear_up ,rear_lo ,blue_R ,mot['rear_u'],mot['rear_l'])
    kinF=leg(front_up,front_lo,blue_R2,mot['front_u'],mot['front_l'])
    if not(kinR and kinF): print("⚠ impossible pose"); return
    tidy=lambda v: round(float(v),10)
    tipR_u,_,_,_,tipR_l,_ = kinR
    tipF_u,_,_,_,tipF_l,_ = kinF
    def pack(tag,kin):
        _,J1,LJ,_,_,GS=kin
        return {f"{tag}_joint1":[tidy(J1.x),tidy(J1.y)],
                f"{tag}_blue_orange":[tidy(LJ.x),tidy(LJ.y)],
                f"{tag}_green_attach":[tidy(GS.x),tidy(GS.y)]}
    pts={**pack('rear',kinR),**pack('front',kinF)}
    with open("robot_points.json","w") as f:
        json.dump({"rear":{k.split('_',1)[1]:v for k,v in pts.items() if k.startswith('rear')},
                   "front":{k.split('_',1)[1]:v for k,v in pts.items() if k.startswith('front')}},
                  f,indent=2)
    print("✓ robot_points.json written")
    # ─── live zero-angles (radians) ──────────────────────────────────
    zero_rear_u  = math.radians(135 + mot['rear_u'])
    zero_rear_l  = math.radians(-45 - mot['rear_l'])
    zero_front_u = math.radians(135 + mot['front_u'])
    zero_front_l = math.radians(-45 - mot['front_l'])
    # link table ------------------------------------------------------
    links_dict={
      "rear_crank_upper":{"length":L["crank_up"]},
      "rear_crank_lower":{"length":L["crank_lo"]},
      "rear_red":{"length":L["red"]},"rear_blue":{"length":L["blue"]},
      "rear_orange":{"length":L["orange"]},"rear_green":{"length":L["green"]},
      "front_crank_upper":{"length":L["crank_up"]},
      "front_crank_lower":{"length":L["crank_lo"]},
      "front_red":{"length":L["red"]},"front_blue":{"length":L["blue"]},
      "front_orange":{"length":L["orange"]},"front_green":{"length":L["green"]}
    }
    def pj(pa,ch,anc): return {"parent":pa,"child":ch,"type":"pin",
                               "anchor":[tidy(anc.x),tidy(anc.y)]}
    geom={
      "link_defaults":{"mass":0.50},
      "torso":{"size":{"width":BODY_W,"height":TOP_OFF+BOT_OFF},
               "mass":2.27,"inertia_zz":0.033,"friction":1.0},
      "fixed_offsets":{"top_offset":TOP_OFF,"bottom_offset":BOT_OFF,
                       "front_offset":FRONT_OFF,"hip_spacing":HIP_SP},
      "links":links_dict,
      "joints":[
        # motors with live zero-angles ------------------------------
        {"name":"rear_upper_motor","parent":"torso","child":"rear_crank_upper",
         "type":"revolute","anchor":[0,0],"limits":[-math.pi,math.pi],
         "motor":{"enabled":True,"max_force":2.5},"zero_angle":zero_rear_u},
        {"name":"rear_lower_motor","parent":"torso","child":"rear_crank_lower",
         "type":"revolute","anchor":[0.0386,-0.0268],"limits":[-math.pi,math.pi],
         "motor":{"enabled":True,"max_force":2.5},"zero_angle":zero_rear_l},
        {"name":"front_upper_motor","parent":"torso","child":"front_crank_upper",
         "type":"revolute","anchor":[HIP_SP,0],"limits":[-math.pi,math.pi],
         "motor":{"enabled":True,"max_force":2.5},"zero_angle":zero_front_u},
        {"name":"front_lower_motor","parent":"torso","child":"front_crank_lower",
         "type":"revolute","anchor":[HIP_SP+0.0386,-0.0268],
         "limits":[-math.pi,math.pi],
         "motor":{"enabled":True,"max_force":2.5},"zero_angle":zero_front_l},
        # rear passive pins -----------------------------------------
        pj("rear_crank_upper","rear_red",   tipR_u),
        pj("rear_red","rear_blue",Vec2d(*pts["rear_joint1"])),
        pj("torso","rear_blue", Vec2d(*pts["rear_blue_orange"])),
        pj("rear_blue","rear_orange",Vec2d(*pts["rear_blue_orange"])),
        pj("rear_crank_lower","rear_green", tipR_l),
        pj("rear_orange","rear_green",Vec2d(*pts["rear_green_attach"])),
        # front passive pins ----------------------------------------
        pj("front_crank_upper","front_red",   tipF_u),
        pj("front_red","front_blue",Vec2d(*pts["front_joint1"])),
        pj("torso","front_blue",Vec2d(*pts["front_blue_orange"])),
        pj("front_blue","front_orange",Vec2d(*pts["front_blue_orange"])),
        pj("front_crank_lower","front_green", tipF_l),
        pj("front_orange","front_green",Vec2d(*pts["front_green_attach"]))
      ],
      "ground":{"segment":[[-2,-0.03],[6,-0.03]],
                "thickness":0.02,"friction":1.2}
    }
    with open("robot_geom.json","w") as g: json.dump(geom,g,indent=2)
    print("✓ robot_geom.json written – run simulation.py!")
pyglet.app.run()
