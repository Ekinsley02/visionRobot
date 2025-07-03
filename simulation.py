#!/usr/bin/env python3
"""Two‑leg linkage simulator

Builds a full Pymunk model from ``robot_geom.json`` and animates it in a
Pyglet window, driving the two hip cranks through a ±20° random sweep.
The key point in this revision is that **all anchors from the JSON file
are treated as *local* coordinates relative to the rear‑upper hip
pivot**.  They are translated to world space with one common ``BASE``
vector before any bodies or joints are created, so every pin actually
lands on the correct spot.
"""

import json
import math
import random

import pyglet
import pymunk
from pymunk import Vec2d
from pymunk.pyglet_util import DrawOptions

# ──────────────────────────────────────────
#  Simulation constants
# ──────────────────────────────────────────
DT, SIM_TIME = 1 / 240, 10.0            # time‑step and demo length [s]
GRAVITY = (0, -9.81)                    # m s⁻²
STALL_TORQUE = 1.18 * 1.20              # Nm, gearbox * fudge
Y_OFFSET = 0.15                         # world‑space height of rear upper hip
BLUE, GREY = (0, 90, 240, 255), (180, 180, 180, 255)
GROUP = 12                              # Pymunk shape‑filter group
KP, TARGET_T = 6.0, 1.0                # servo P‑gain and random target period
RANGE_RAD = math.radians(20)            # ±20° random sweep for the cranks


def rod_I(m: float, L: float) -> float:
    """Moment of inertia of a slender rod about its centre."""
    return m * L * L / 12.0


# ──────────────────────────────────────────
#  Load robot spec
# ──────────────────────────────────────────
with open("robot_geom.json", "r", encoding="utf‑8") as fh:
    spec = json.load(fh)

links = spec["links"]
joints = spec["joints"]
torso_spec = spec["torso"]
offsets = spec["fixed_offsets"]

# Rear‑upper hip pivot (the origin used in CAD for all anchor numbers)
BASE = Vec2d(0, Y_OFFSET)

def w(pt):  # noqa: E741  (short but clear here)
    """Convert *local* anchor (from JSON) → world coordinates."""
    return BASE + Vec2d(*pt)


# ──────────────────────────────────────────
#  Pymunk world
# ──────────────────────────────────────────
space = pymunk.Space()
space.gravity = GRAVITY
space.damping = 0.99
space.iterations = 60
flt = pymunk.ShapeFilter(group=GROUP)

bodies: dict[str, pymunk.Body] = {}


# ──────────────────────────────────────────
#  Torso
# ──────────────────────────────────────────
# Size dictionary might not preserve order, grab explicitly

tw = torso_spec["size"]["width"]
th = torso_spec["size"]["height"]

ru = BASE                                        # rear‑upper anchor
rl = ru + (0.0386, -0.0268)                      # rear‑lower anchor (CAD)
fl = rl + (offsets["hip_spacing"], 0.0)          # front‑lower anchor
front_x = fl.x + offsets["front_offset"]
rear_x = front_x - tw
cy = (ru.y + offsets["top_offset"] + rl.y - offsets["bottom_offset"]) / 2

# Body
torso = pymunk.Body(torso_spec["mass"], torso_spec["inertia_zz"])
torso.position = Vec2d((front_x + rear_x) / 2, cy)
poly = pymunk.Poly.create_box(torso, (tw, th))
poly.color = BLUE
poly.filter = flt
space.add(torso, poly)
bodies["torso"] = torso

# Ground segment
(g0_local, g1_local) = spec["ground"]["segment"]
gseg = pymunk.Segment(space.static_body, Vec2d(*g0_local), Vec2d(*g1_local),
                      spec["ground"]["thickness"])
gseg.color = GREY
space.add(gseg)


# ──────────────────────────────────────────
#  Collect anchor sets and zero angles
# ──────────────────────────────────────────
anchors: dict[str, list[Vec2d]] = {}
zero_angle: dict[str, float] = {}

for jd in joints:
    P = w(jd["anchor"])
    anchors.setdefault(jd["parent"], []).append(P)
    anchors.setdefault(jd["child"], []).append(P)

    if jd["type"] == "revolute" and jd["parent"] == "torso":
        zero_angle[jd["child"]] = jd["zero_angle"]

# Deduplicate anchor lists
for plist in anchors.values():
    uniq: list[Vec2d] = []
    for pt in plist:
        if all((pt - q).length > 1e-5 for q in uniq):
            uniq.append(pt)
    plist[:] = uniq


# ──────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────

def longest_pair(pts: list[Vec2d]) -> tuple[Vec2d, Vec2d]:
    """Return the two points in *pts* that are farthest apart."""
    best_a, best_b, best_d = pts[0], pts[0], 0.0
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            d = (b - a).length
            if d > best_d:
                best_a, best_b, best_d = a, b, d
    return best_a, best_b


def build_passive(pts: list[Vec2d], mass: float, nominal: float):
    """Create a free rod that spans the longest distance in *pts*."""
    A, B = longest_pair(pts)
    axis = B - A
    length = axis.length or nominal * 0.25  # fall back if all anchors overlap
    body = pymunk.Body(mass, rod_I(mass, length))
    body.position = A
    body.angle = axis.angle
    seg = pymunk.Segment(body, (0, 0), (length, 0), 0.002)
    return body, seg


# ──────────────────────────────────────────
#  Build every link body
# ──────────────────────────────────────────
for name, info in links.items():
    mass = info.get("mass", 0.015)
    L0 = info["length"]
    pts = anchors.get(name, [])

    if name in zero_angle:  # motorised crank on torso
        pivot_local = next(j["anchor"] for j in joints
                            if j["child"] == name and j["parent"] == "torso")
        pivot = w(pivot_local)

        body = pymunk.Body(mass, rod_I(mass, L0))
        body.position = pivot
        body.angle = zero_angle[name]
        seg = pymunk.Segment(body, (0, 0), (L0, 0), 0.002)

    elif pts:              # passive connecting rod
        body, seg = build_passive(pts, mass, L0)

    else:                  # orphan link – warn and skip
        print("⚠  skipping link (no anchors):", name)
        continue

    seg.color = BLUE
    seg.filter = flt
    seg.friction = 1.0
    space.add(body, seg)
    bodies[name] = body


# ──────────────────────────────────────────
#  Joints and (optional) motors
# ──────────────────────────────────────────

def stiff_pivot(a: pymunk.Body, b: pymunk.Body, P: Vec2d) -> pymunk.Constraint:
    pj = pymunk.PivotJoint(a, b, P)
    pj.max_force = 8e3
    pj.error_bias = 0.0
    pj.max_bias = 0.0
    return pj


servos: list[dict] = []

for jd in joints:
    parent = bodies[jd["parent"]]
    child = bodies[jd["child"]]
    world_anchor = w(jd["anchor"])

    space.add(stiff_pivot(parent, child, world_anchor))

    if jd["type"] == "revolute":
        lo, hi = jd.get("limits", [-math.pi, math.pi])
        rl = pymunk.RotaryLimitJoint(parent, child, lo, hi)
        rl.max_force = 8e3
        rl.error_bias = 0.0
        rl.max_bias = 0.0
        space.add(rl)

        if jd.get("motor", {}).get("enabled", False):
            mot = pymunk.SimpleMotor(parent, child, 0.0)
            mot.max_force = STALL_TORQUE
            space.add(mot)

            # Only drive the two upper (hip) motors
            if jd["name"] in ("rear_upper_motor", "front_upper_motor"):
                home = child.angle - parent.angle
                servos.append({
                    "motor": mot,
                    "parent": parent,
                    "child": child,
                    "home": home,
                    "target": home,
                })


# ──────────────────────────────────────────
#  Simple servo controller
# ──────────────────────────────────────────

def new_targets(_):
    """Pick fresh random targets for each servo every TARGET_T seconds."""
    for s in servos:
        s["target"] = s["home"] + random.uniform(-RANGE_RAD, RANGE_RAD)


def drive_servos():
    for s in servos:
        rel = s["child"].angle - s["parent"].angle
        err = (s["target"] - rel + math.pi) % (2.0 * math.pi) - math.pi
        s["motor"].rate = KP * err


new_targets(0)
pyglet.clock.schedule_interval(new_targets, TARGET_T)


# ──────────────────────────────────────────
#  Viewer
# ──────────────────────────────────────────
win = pyglet.window.Window(820, 540, "assembled robot")
opt = DrawOptions()
opt.flip_y = True
opt.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
opt.transform = pymunk.Transform(a=320, d=320, tx=450, ty=260)


@win.event
def on_draw():
    win.clear()
    space.debug_draw(opt)


# ──────────────────────────────────────────
#  Main simulation loop
# ──────────────────────────────────────────

def tick(_):
    drive_servos()
    space.step(DT)


tid = pyglet.clock.schedule_interval(tick, DT)
pyglet.clock.schedule_once(lambda _:
                           (pyglet.clock.unschedule(tid), win.close()),
                           SIM_TIME)
pyglet.app.run()
