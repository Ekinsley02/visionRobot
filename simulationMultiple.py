#!/usr/bin/env python3
# simulation_multi.py – 20-robot evolutionary arena
#
# ▸ Geometry & joint layout identical to single-robot reference.
# ▸ Inter-robot collisions disabled via collision-group 99.
# ▸ Fitness  = (right-ward travel) – 0.3 × |torso tilt|.
# ▸ Best-3 elitism, tiny Gaussian mutation on the champion.
#
# Requires: pymunk ≥ 6, pyglet ≥ 2
# --------------------------------------------------------------------------
import json, math, random, pathlib
import pymunk, pyglet
from   pymunk import Vec2d
from   pymunk.pyglet_util import DrawOptions

# ───────── simulation constants ─────────
DT, SIM_TIME  = 1/240, 20.0
GRAVITY       = (0, -9.81)

POP_SIZE      = 20
RAND_PERIOD   = 0.5
GENE_LEN      = int(SIM_TIME / RAND_PERIOD)
STALL_TORQUE  = 1.18 * 2.4
Y_OFFSET      = 0.12

GROUP_ID      = 99                       # same group = no inter-robot contacts
STRIPE_LEN    = 0.1524
STRIPE_COLS   = [(230,230,230,255), (160,160,160,255)]

RAND_MAX_RATE = math.radians(60)
SAVE_FILE     = pathlib.Path("best_genes.json")

# torso colours (20)
TORSO_COLS=[(255,60,60,255),(60,255,60,255),(60,60,255,255),(255,255,60,255),
            (255,60,255,255),(60,255,255,255),(255,140,60,255),(180,180,180,255),
            (255,210,40,255),(180,100,220,255),(255,100,140,255),(100,255,140,255),
            (140,100,255,255),(255,180,100,255),(100,180,255,255),(200,200,80,255),
            (255,80,200,255),(80,255,200,255),(200,80,255,255),(120,120,120,255)]

LINK_COL={"red":(255,0,0,255),"blue":(0,0,255,255),"green":(0,180,0,255),
          "orange":(255,140,0,255),"upper":(120,120,120,255),"lower":(120,120,120,255)}

PURPLE=(140,0,140,255); FOOT_R=25.2e-3*0.5; FOOT_M=0.015; FOOT_F=4.0; FOOT_E=0.05

# ───────── load CAD spec ─────────
spec=json.load(open("robot_geom.json"))
links,joints=spec["links"],spec["joints"]
torso_spec,offs=spec["torso"],spec["fixed_offsets"]

LEN={n:d["length"] for n,d in links.items()}
Lred,Lblue,Lorange=(LEN[k] for k in("rear_red","rear_blue","rear_orange"))

R_UP,R_LO=38e-3,30e-3
ATTACH,J_F=48e-3,0.24
HIP_SP,TOP,BOT,FRONT=(offs[k] for k in
                      ("hip_spacing","top_offset","bottom_offset","front_offset"))

rod_I=lambda m,L: m*L*L/12.0
v=lambda obj: Vec2d(*obj)

# ───────── genomes ─────────
def random_gene():
    return [[random.uniform(-RAND_MAX_RATE,RAND_MAX_RATE) for _ in range(4)]
            for _ in range(GENE_LEN)]

def load_population():
    try:
        pop=json.load(open(SAVE_FILE)) if SAVE_FILE.is_file() else []
        if not isinstance(pop,list): raise ValueError
    except Exception:
        pop=[]
    return (pop+[random_gene()]*POP_SIZE)[:POP_SIZE]

def save_population(pop): SAVE_FILE.write_text(json.dumps(pop[:POP_SIZE],indent=1))

# ───────── FK helper ─────────
def leg(up,lo,right,th_u,th_l):
    t1,t2=math.radians(135+th_u),math.radians(-45-th_l)
    tip1=up+Vec2d(R_UP*math.cos(t1),R_UP*math.sin(t1))
    tip2=lo+Vec2d(R_LO*math.cos(t2),R_LO*math.sin(t2))
    r,d=Lred,Lblue*(1-J_F)
    D=right-tip1; dist=D.length
    if not(0<dist<r+d and dist>abs(r-d)): return None
    a=(r*r-d*d+dist*dist)/(2*dist); h=math.sqrt(max(0,r*r-a*a))
    J1=tip1+D.normalized()*a-Vec2d(-D.y,D.x).normalized()*h
    LJ=right-(right-J1)/(1-J_F)
    vec=tip2-LJ
    if not(0<vec.length<2*Lorange): return None
    mid=LJ+vec*0.5; h2=math.sqrt(max(0,Lorange**2-(vec.length*0.5)**2))
    perp=Vec2d(-vec.y,vec.x).normalized()*h2
    OB=mid-perp if (mid-perp).y<(mid+perp).y else mid+perp
    if (OB-LJ).length<ATTACH: return None
    GS=LJ+(OB-LJ).normalized()*ATTACH
    return tip1,J1,LJ,OB,tip2,GS

# ───────── physics factory ─────────
def new_space():
    s=pymunk.Space(); s.gravity=GRAVITY; s.damping=0.99; s.iterations=60
    return s

# ───────── Robot ─────────
class Robot:
    def __init__(self,idx,gene,space):
        self.idx, self.gene, self.space = idx, gene, space
        self.motors=[]; self.frame=0
        self._build()
        self.x0=self.torso.position.x; self.a0=self.torso.angle

    # ───────────────── construction ────────────────────────────
    def _build(self):
        filt=pymunk.ShapeFilter(group=GROUP_ID)
        bodies={}

        # torso --------------------------------------------------
        tor_w,tor_h=torso_spec["size"].values()
        rear_up=Vec2d(0,Y_OFFSET)
        rear_lo=rear_up+(0.0386,-0.0268)
        front_lo=rear_lo+(HIP_SP,0)
        front_up=front_lo-(rear_lo-rear_up)
        front_x=front_lo.x+FRONT; rear_x=front_x-tor_w
        centre=Vec2d((front_x+rear_x)/2,(rear_up.y+TOP+rear_lo.y-BOT)/2)

        self.torso=pymunk.Body(torso_spec["mass"],torso_spec["inertia_zz"])
        self.torso.position=centre
        poly=pymunk.Poly.create_box(self.torso,(tor_w,tor_h))
        poly.color=TORSO_COLS[self.idx]; poly.filter=filt
        self.space.add(self.torso,poly)

        # zero-pose angles --------------------------------------
        z={j["child"]:j["zero_angle"] for j in joints
           if j["type"]=="revolute" and j["parent"]=="torso"}
        ru,rl=math.degrees(z["rear_crank_upper"])-135,-math.degrees(z["rear_crank_lower"])-45
        fu,fl=math.degrees(z["front_crank_upper"])-135,-math.degrees(z["front_crank_lower"])-45

        blueR=rear_lo+(0.004,0.0113); blueF=front_lo+(0.004,0.0113)
        kR=leg(rear_up,rear_lo,blueR,ru,rl); kF=leg(front_up,front_lo,blueF,fu,fl)
        if not(kR and kF): raise RuntimeError("FK failed")
        (t1R,J1R,LJR,OBR,t2R,GSR)=kR
        (t1F,J1F,LJF,OBF,t2F,GSF)=kF
        len_up,len_lo=LEN["rear_crank_upper"],LEN["rear_crank_lower"]

        # helper ─ rigid bar ------------------------------------
        def add_bar(name,A,B):
            axis=B-A; m=links[name].get("mass",0.02)
            body=pymunk.Body(m,rod_I(m,axis.length))
            body.position,body.angle=A,axis.angle
            seg=pymunk.Segment(body,(0,0),(axis.length,0),0.002)
            seg.color=LINK_COL.get(name.split('_')[-1],(200,200,200,255))
            seg.filter=filt
            self.space.add(body,seg); bodies[name]=body

        # → bars -------------------------------------------------
        add_bar("rear_crank_upper",rear_up,rear_up+Vec2d(len_up,0).rotated(math.radians(135+ru)))
        add_bar("rear_crank_lower",rear_lo,rear_lo+Vec2d(len_lo,0).rotated(math.radians(-45-rl)))
        add_bar("rear_red",t1R,J1R); add_bar("rear_blue",LJR,blueR)
        add_bar("rear_orange",LJR,OBR); add_bar("rear_green",t2R,GSR)

        add_bar("front_crank_upper",front_up,front_up+Vec2d(len_up,0).rotated(math.radians(135+fu)))
        add_bar("front_crank_lower",front_lo,front_lo+Vec2d(len_lo,0).rotated(math.radians(-45-fl)))
        add_bar("front_red",t1F,J1F); add_bar("front_blue",LJF,blueF)
        add_bar("front_orange",LJF,OBF); add_bar("front_green",t2F,GSF)

        # feet ---------------------------------------------------
        self._add_foot(bodies["rear_orange"],OBR,filt)
        self._add_foot(bodies["front_orange"],OBF,filt)

        # joint helpers -----------------------------------------
        def soft_pivot(A,B,W):
            pj=pymunk.PivotJoint(A,B,W); pj.filter=filt
            pj.max_force=8_000; pj.error_bias=0; pj.max_bias=0
            self.space.add(pj); return pj

        def rigid_weld(A,B,W):
            """pivot + frozen relative angle"""
            pj=pymunk.PivotJoint(A,B,W);  pj.filter=filt;  pj.max_force=1e12
            self.space.add(pj)
            a0=B.angle-A.angle
            rl=pymunk.RotaryLimitJoint(A,B,a0,a0); rl.filter=filt; rl.max_force=1e12
            self.space.add(rl)

        # any link-pair inside this set will get a rigid_weld
        rigid_pairs={frozenset(("rear_blue","rear_orange")),
                     frozenset(("front_blue","front_orange")),
                     frozenset(("rear_green","rear_orange")),
                     frozenset(("front_green","front_orange"))}

        # soft pins / limits from JSON ---------------------------
        for jd in joints:
            key=frozenset((jd["parent"],jd["child"]))
            if key in rigid_pairs:
                continue                       # handled later

            a=self.torso if jd["parent"]=="torso" else bodies[jd["parent"]]
            b=bodies[jd["child"]]
            W=v(jd["anchor"])+Vec2d(0,Y_OFFSET)
            soft_pivot(a,b,W)

            if jd["type"]=="revolute":
                lo,hi=jd.get("limits",[-math.pi,math.pi])
                rl=pymunk.RotaryLimitJoint(a,b,lo,hi); rl.filter=filt
                rl.max_force=8_000; rl.error_bias=0; rl.max_bias=0
                self.space.add(rl)

        # rigid welds -------------------------------------------
        rigid_weld(bodies["rear_blue"],  bodies["rear_orange"],  LJR)
        rigid_weld(bodies["front_blue"], bodies["front_orange"], LJF)
        rigid_weld(bodies["rear_green"], bodies["rear_orange"],  GSR)
        rigid_weld(bodies["front_green"],bodies["front_orange"], GSF)

        # blue tip pinned to torso (second anchor) --------------
        rigid_weld(self.torso, bodies["rear_blue"],  blueR)
        rigid_weld(self.torso, bodies["front_blue"], blueF)

        # freeze blue-torso angle (prevents tiny drift) ---------
        for bname in ("rear_blue","front_blue"):
            bar=bodies[bname]
            a0=bar.angle-self.torso.angle
            lock=pymunk.RotaryLimitJoint(self.torso,bar,a0,a0)
            lock.filter=filt; lock.max_force=1e12
            self.space.add(lock)

        # motors -------------------------------------------------
        for A,B in [("rear_crank_upper","rear_red"),
                    ("rear_crank_lower","rear_green"),
                    ("front_crank_upper","front_red"),
                    ("front_crank_lower","front_green")]:
            m=pymunk.SimpleMotor(bodies[A],bodies[B],0); m.filter=filt
            m.max_force=STALL_TORQUE
            self.space.add(m); self.motors.append(m)

    # ───────── helpers ─────────
    def _add_foot(self,bar_body,world_tip,filt):
        local=(world_tip-bar_body.position).rotated(-bar_body.angle)
        foot=pymunk.Circle(bar_body,FOOT_R,local)
        foot.mass=FOOT_M; foot.friction=FOOT_F; foot.elasticity=FOOT_E
        foot.color=PURPLE; foot.filter=filt
        self.space.add(foot)

    def drive(self,_dt):
        i=min(self.frame,GENE_LEN-1)
        for mot,rate in zip(self.motors,self.gene[i]): mot.rate=rate
        self.frame+=1

    def fitness(self):
        return (self.torso.position.x-self.x0) - 0.3*abs(self.torso.angle-self.a0)

# ───────── ground stripes ─────────
def build_ground(space):
    seg0,seg1=(v(p) for p in spec["ground"]["segment"])
    thick=spec["ground"]["thickness"]
    x=seg0.x; idx=0
    while x<seg1.x-1e-6:
        nx=min(x+STRIPE_LEN,seg1.x)
        seg=pymunk.Segment(space.static_body,(x,seg0.y),(nx,seg1.y),thick)
        seg.color=STRIPE_COLS[idx&1]; space.add(seg)
        x=nx; idx+=1

# ───────── evolution helpers ─────────
def run_generation(space,robots):
    for r in robots: pyglet.clock.schedule_interval(r.drive,RAND_PERIOD)
    pyglet.clock.schedule_interval(lambda _dt: space.step(DT),DT)
    pyglet.clock.schedule_once(lambda _dt: pyglet.app.exit(),SIM_TIME)
    pyglet.app.run()
    for r in robots: pyglet.clock.unschedule(r.drive)

# ───────── main loop ─────────
if __name__=="__main__":
    genes=load_population(); gen=0
    while True:
        space=new_space(); build_ground(space)
        robots=[Robot(i,genes[i],space) for i in range(POP_SIZE)]

        win=pyglet.window.Window(1020,600,f"generation {gen}")
        opt=DrawOptions(); opt.flip_y=True
        opt.flags=pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        opt.transform=pymunk.Transform(a=320,d=320,tx=450,ty=260)

        @win.event
        def on_draw(): win.clear() or space.debug_draw(opt)

        run_generation(space,robots); win.close()

        scored=sorted(((r.fitness(),r) for r in robots),reverse=True)
        print(f"\n=== generation {gen} – best distances (m) ===",
              ", ".join(f"{d:.3f}" for d,_ in scored[:3]))

        new=[r.gene for _,r in scored[:3]]
        while len(new)<POP_SIZE:
            base=[row[:] for row in new[0]]
            r,c=random.randrange(GENE_LEN),random.randrange(4)
            base[r][c]+=random.gauss(0,0.15)
            new.append(base)

        genes=new; save_population(genes); gen+=1
