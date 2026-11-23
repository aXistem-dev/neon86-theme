#!/usr/bin/env python3
"""Generate the original Neon '86 wallpapers (3840x2160 PNG masters).

Everything is procedural: gradients, a striped sun, a perspective grid,
palm silhouettes, an art-deco skyline, neon shapes, water reflection, haze
and grain. No external artwork, photos, fonts or logos are used.

Requires Pillow and numpy.
    python3 generate-wallpapers.py OUT_DIR [--seed 1986] [--scale 1.0]
Writes OUT_DIR/sunset-grid.png and OUT_DIR/ocean-drive.png (OUT_DIR must be new).
"""
import argparse
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W0, H0 = 3840, 2160


def hexrgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def vgrad(h, stops):
    """stops: [(pos 0..1, '#rrggbb'), ...] -> (h, 3) float array, smoothstep blend."""
    ys = np.linspace(0, 1, h, dtype=np.float32)
    out = np.zeros((h, 3), np.float32)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        m = (ys >= p0) & (ys <= p1)
        t = (ys[m] - p0) / max(p1 - p0, 1e-6)
        t = t * t * (3 - 2 * t)
        out[m] = hexrgb(c0) * (1 - t)[:, None] + hexrgb(c1) * t[:, None]
    out[ys < stops[0][0]] = hexrgb(stops[0][1])
    out[ys > stops[-1][0]] = hexrgb(stops[-1][1])
    return out


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB')


def arr(img):
    return np.asarray(img, dtype=np.float32)


def blur_layer(img, radius):
    """Cheap wide blur: blur at quarter size and upscale."""
    w, h = img.size
    small = img.resize((max(1, w // 4), max(1, h // 4)), Image.Resampling.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(max(1, radius / 4)))
    return small.resize((w, h), Image.Resampling.BICUBIC)


def add_glow(base, layer_rgb, radii=((6, 1.0), (40, 0.8), (140, 0.5))):
    """base: float HxWx3; layer_rgb: PIL RGB emissive layer (black = nothing)."""
    out = base + arr(layer_rgb)
    for r, s in radii:
        out += arr(layer_rgb.filter(ImageFilter.GaussianBlur(r)) if r < 20 else blur_layer(layer_rgb, r)) * s
    return out


def composite(base, mask_l, color):
    m = arr(mask_l)[..., None] / 255.0
    return base * (1 - m) + hexrgb(color) * m


def grain(base, rng, amount=3.5):
    n = rng.normal(0, amount, base.shape[:2]).astype(np.float32)[..., None]
    return base + n


def vignette(base, strength=0.35):
    h, w = base.shape[:2]
    y, x = np.ogrid[0:h, 0:w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    return base * (1 - strength * d ** 2.2)[..., None]


# ---------------------------------------------------------------- palms

def bezier(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)


def draw_palm(draw, base, height, lean, rng, s=1.0, fronds=11):
    """Silhouette palm into an 'L' ImageDraw (fill 255). base=(x,y) ground point."""
    bx, by = base
    top = (bx + lean * height, by - height)
    ctrl = (bx + lean * height * 0.15, by - height * 0.55)
    spine = bezier((bx, by), ctrl, top, 60)
    w0, w1 = 26 * s, 12 * s
    left, right = [], []
    for i in range(len(spine)):
        t = i / (len(spine) - 1)
        p = spine[i]
        q = spine[min(i + 1, len(spine) - 1)] - spine[max(i - 1, 0)]
        nrm = np.array([-q[1], q[0]]) / (np.hypot(*q) + 1e-6)
        # Slight ring texture: width wobble.
        wv = (w0 * (1 - t) + w1 * t) * (1 + 0.06 * math.sin(i * 1.7))
        left.append(tuple(p + nrm * wv / 2))
        right.append(tuple(p - nrm * wv / 2))
    draw.polygon(left + right[::-1], fill=255)
    cx, cy = top
    draw.ellipse((cx - 24 * s, cy - 18 * s, cx + 24 * s, cy + 22 * s), fill=255)
    for k in range(fronds):
        ang = -math.pi / 2 + (k / (fronds - 1) - 0.5) * math.pi * 1.95 + rng.uniform(-0.1, 0.1)
        length = height * rng.uniform(0.3, 0.42)
        droop = length * rng.uniform(0.3, 0.55) * (0.4 + 0.6 * abs(math.cos(ang)))
        end = (cx + math.cos(ang) * length, cy + math.sin(ang) * length * 0.5 + droop)
        mid = (cx + math.cos(ang) * length * 0.55, cy + math.sin(ang) * length * 0.55 - length * 0.16)
        pts = bezier((cx, cy), mid, end, 34)
        maxw = length * rng.uniform(0.14, 0.18)
        left, right = [], []
        for i in range(len(pts)):
            t = i / (len(pts) - 1)
            p = pts[i]
            q = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
            d = q / (np.hypot(*q) + 1e-6)
            nrm = np.array([-d[1], d[0]])
            w = maxw * math.sin(math.pi * min(1.0, t * 1.08)) ** 0.75 + 2 * s
            # Serrated leaflet tips swept toward the frond end and gravity.
            serr = 1.0 if i % 2 else 0.5
            for side, acc in ((1, left), (-1, right)):
                hang = np.array([0.0, 0.35]) * w * serr
                acc.append(tuple(p + nrm * side * w * serr + d * w * 0.35 * serr + hang))
        draw.polygon([tuple(pts[0])] + left + [tuple(pts[-1])] + right[::-1], fill=255)
        for i in range(int((len(pts) - 1) * 0.82)):
            t = i / (len(pts) - 1)
            draw.line([tuple(pts[i]), tuple(pts[i + 1])], fill=255, width=max(2, int(8 * s * (1 - t) + 2)))


# ---------------------------------------------------------------- sunset grid

def sunset_grid(W, H, rng):
    hz = int(H * 0.6)
    sky = vgrad(hz, [(0.0, '#07061a'), (0.22, '#1a0b3d'), (0.5, '#4b0f63'), (0.72, '#a3206f'), (0.88, '#ff4f6d'), (1.0, '#ff9a45')])
    img = np.zeros((H, W, 3), np.float32)
    img[:hz] = sky[:, None, :]
    ground = vgrad(H - hz, [(0.0, '#1c0730'), (0.15, '#0d0520'), (1.0, '#04030c')])
    img[hz:] = ground[:, None, :]

    # Stars in the upper sky.
    stars = Image.new('RGB', (W, H))
    sd = ImageDraw.Draw(stars)
    for _ in range(int(900 * W / 3840)):
        x, y = rng.uniform(0, W), rng.uniform(0, hz * 0.55) ** 1.0
        b = rng.uniform(60, 220) * (1 - y / (hz * 0.6))
        r = rng.choice([1, 1, 1, 2]) * W / 3840
        sd.ellipse((x - r, y - r, x + r, y + r), fill=(int(b), int(b * 0.9), int(b)))
    img = add_glow(img, stars, radii=((3, 0.6),))

    # Striped sun.
    R = int(H * 0.27)
    cx, cy = W // 2, hz - int(R * 0.42)
    sun = np.zeros((H, W, 3), np.float32)
    yy, xx = np.ogrid[0:H, 0:W]
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= R * R
    t = np.clip((yy - (cy - R)) / (2 * R), 0, 1)
    col = (hexrgb('#ffe066') * (1 - t)[..., None] + hexrgb('#ff2e88') * t[..., None])
    col = np.broadcast_to(col, (H, W, 3))
    # Stripes: gaps grow toward the bottom.
    rel = (yy - cy) / R  # -1..1
    stripe = np.ones_like(rel, dtype=bool)
    for i in range(9):
        y0 = -0.16 + i * 0.075
        thick = 0.012 + i * 0.0075
        stripe &= ~((rel > y0) & (rel < y0 + thick))
    mask = inside & stripe & (yy < hz)
    sun[mask] = col[mask] * 0.92
    sun_img = to_img(sun)
    img = img * (1 - mask[..., None] * 1.0)
    img = add_glow(img, sun_img, radii=((60, 0.18), (200, 0.32), (420, 0.28)))

    # Distant mountains with a neon rim.
    mtn = Image.new('L', (W, H))
    md = ImageDraw.Draw(mtn)
    rim = Image.new('RGB', (W, H))
    rd = ImageDraw.Draw(rim)
    for layer, (amp, base_off, color, rimc) in enumerate([(0.09, 0.0, '#2a0b45', (90, 40, 160)), (0.055, 0.0, '#14062a', (255, 60, 170))]):
        pts = []
        x = 0
        y_base = hz - base_off * H
        while x <= W + 40:
            n = (math.sin(x / W * 9.1 + layer * 3) * 0.5 + math.sin(x / W * 23.7 + layer) * 0.3 + rng.uniform(-0.15, 0.15))
            edge = abs(x - W / 2) / (W / 2)
            hgt = amp * H * (0.35 + 0.65 * edge) * (0.6 + 0.4 * n)
            pts.append((x, y_base - max(0, hgt)))
            x += W / 90
        poly = [(0, hz)] + pts + [(W, hz)]
        md.polygon(poly, fill=255)
        img = composite(img, mtn, color)
        rd.line(pts, fill=rimc, width=max(2, int(3 * W / 3840)))
        mtn = Image.new('L', (W, H))
        md = ImageDraw.Draw(mtn)
    img = add_glow(img, rim, radii=((4, 0.7), (30, 0.4)))

    # Perspective grid.
    grid = Image.new('RGB', (W, H))
    gd = ImageDraw.Draw(grid)
    lw = max(2, int(4 * W / 3840))
    depth = H - hz
    for i in range(1, 26):
        z = i / 25.0
        y = hz + depth * (z ** 2.4)
        fade = min(1, z * 2.2)
        c = (int(255 * fade), int(40 * fade), int(170 * fade))
        gd.line([(0, y), (W, y)], fill=c, width=lw)
    vx = W / 2
    for k in range(-34, 35):
        xb = vx + k * W * 0.075
        c = (int(60), int(200), int(255)) if k % 6 == 0 else (230, 50, 170)
        gd.line([(vx + (xb - vx) * 0.02, hz), (xb, H)], fill=c, width=lw)
    # Fade the grid near the horizon.
    gmask = np.zeros((H, W, 1), np.float32)
    gmask[hz:, :, 0] = np.clip(np.linspace(0, 1, depth)[:, None] * 3.0, 0, 1) ** 1.4
    grid = to_img(arr(grid) * gmask)
    img = add_glow(img, grid, radii=((5, 0.9), (28, 0.6), (110, 0.35)))

    # Horizon haze band.
    haze = np.zeros((H, W, 3), np.float32)
    band = np.exp(-((np.arange(H) - hz) / (H * 0.025)) ** 2)[:, None, None]
    haze += band * hexrgb('#ff6aa0') * 0.55
    img += haze

    # Palms, framing left and right.
    palm = Image.new('L', (W, H))
    pd = ImageDraw.Draw(palm)
    s = W / 3840
    draw_palm(pd, (W * 0.06, H + 10), H * 0.78, 0.16, rng, s * 1.2, fronds=12)
    draw_palm(pd, (W * 0.16, H + 10), H * 0.46, -0.1, rng, s * 0.9, fronds=10)
    draw_palm(pd, (W * 0.94, H + 10), H * 0.82, -0.18, rng, s * 1.25, fronds=12)
    palm = palm.filter(ImageFilter.GaussianBlur(1.0 * s))
    img = composite(img, palm, '#06030d')

    img = vignette(img, 0.38)
    img = grain(img, np.random.default_rng(rng.randrange(1 << 30)), 3.2)
    return to_img(img)


# ---------------------------------------------------------------- ocean drive

def deco_building(draw, glow_draw, x, w, top, ground, rng, s, palette):
    """Art-deco hotel: stepped crown, rounded corner, fins, neon bands. Returns window rects."""
    body = palette['body']
    if rng.random() < 0.5:
        draw.rounded_rectangle((x, top, x + w, ground + w * 0.2), radius=w * 0.18, fill=body)
    else:
        draw.rectangle((x, top, x + w, ground), fill=body)
    # Stepped crown.
    steps = rng.choice([2, 3])
    cw, ct = w, top
    for i in range(steps):
        cw = cw * 0.68
        h = (ground - top) * rng.uniform(0.04, 0.07)
        cx = x + (w - cw) / 2
        draw.rectangle((cx, ct - h, cx + cw, ct), fill=body)
        ct -= h
    # Central fin / spire.
    fin_w = max(6 * s, w * 0.06)
    fx = x + w / 2 - fin_w / 2
    draw.rectangle((fx, ct - (ground - top) * 0.12, fx + fin_w, ct), fill=body)
    # Neon outline of the crown and vertical bands.
    nc = palette['neon']
    lw = max(2, int(4 * s))
    glow_draw.line([(x + w * 0.04, top), (x + w * 0.96, top)], fill=nc, width=lw)
    if rng.random() < 0.45:
        side = x + w * rng.choice([0.03, 0.97])
        glow_draw.line([(side, top + (ground - top) * 0.02), (side, ground - (ground - top) * 0.25)], fill=nc, width=lw)
    # Spire outline only above the roof line.
    sh = (ground - top) * 0.12
    glow_draw.line([(fx, ct), (fx, ct - sh), (fx + fin_w, ct - sh), (fx + fin_w, ct)], fill=palette['neon2'], width=lw)
    for band in range(rng.choice([0, 1, 2])):
        by = top + (ground - top) * (0.1 + band * 0.07)
        glow_draw.line([(x + w * 0.08, by), (x + w * 0.92, by)], fill=palette['neon2'] if band else nc, width=lw)
    # Abstract signs (no text): ring, chevron stack or vertical blade sign.
    kind = rng.choice(['ring', 'chevron', 'blade', None, None])
    r = w * 0.1
    sx = x + w * rng.choice([0.2, 0.8])
    sy = top + (ground - top) * rng.uniform(0.22, 0.4)
    if kind == 'ring':
        glow_draw.ellipse((sx - r, sy - r, sx + r, sy + r), outline=nc, width=lw)
        glow_draw.ellipse((sx - r * 0.55, sy - r * 0.55, sx + r * 0.55, sy + r * 0.55), outline=palette['neon2'], width=lw)
    elif kind == 'chevron':
        for c in range(3):
            yy = sy + c * r * 0.6
            glow_draw.line([(sx - r * 0.8, yy), (sx, yy + r * 0.4), (sx + r * 0.8, yy)], fill=nc, width=lw)
    elif kind == 'blade':
        glow_draw.rounded_rectangle((sx - r * 0.3, sy - r * 1.6, sx + r * 0.3, sy + r * 1.6), radius=r * 0.3, outline=nc, width=lw)
        for c in range(4):
            yy = sy - r * 1.1 + c * r * 0.72
            glow_draw.line([(sx - r * 0.12, yy), (sx + r * 0.12, yy)], fill=palette['neon2'], width=lw)
    # Windows: small, dim, in deco horizontal bands; some rows fully lit strips.
    wins = []
    pitch = 30 * s
    rows = int((ground - top) / pitch)
    cols = max(4, int(w / (30 * s)))
    for r_ in range(4, rows - 1):
        wy = top + r_ * pitch + 8 * s
        if rng.random() < 0.08:
            col = rng.choice(palette['windows'])
            wins.append((x + w * 0.1, wy + 3 * s, x + w * 0.9, wy + 7 * s, tuple(int(c * 0.6) for c in col)))
            continue
        for c in range(cols):
            if rng.random() < 0.28:
                wx = x + w * 0.1 + c * (w * 0.8 / cols)
                col = rng.choice(palette['windows'])
                k = rng.uniform(0.35, 0.75)
                wins.append((wx, wy, wx + w * 0.8 / cols * 0.5, wy + 10 * s, tuple(int(v * k) for v in col)))
    return wins


def ocean_drive(W, H, rng):
    s = W / 3840
    shore = int(H * 0.66)
    sky = vgrad(shore, [(0.0, '#050716'), (0.35, '#120a35'), (0.68, '#3a0f5c'), (0.9, '#8a1f78'), (1.0, '#c23a7a')])
    img = np.zeros((H, W, 3), np.float32)
    img[:shore] = sky[:, None, :]

    # Moon with soft halo.
    moon = Image.new('RGB', (W, H))
    md = ImageDraw.Draw(moon)
    mx, my, mr = W * 0.73, H * 0.17, H * 0.055
    md.ellipse((mx - mr, my - mr, mx + mr, my + mr), fill=(255, 214, 236))
    img = add_glow(img * 1.0, moon, radii=((8, 0.4), (80, 0.5), (300, 0.35)))

    # Stars.
    stars = Image.new('RGB', (W, H))
    sd = ImageDraw.Draw(stars)
    for _ in range(int(500 * s)):
        x, y = rng.uniform(0, W), rng.uniform(0, shore * 0.5)
        b = rng.uniform(50, 180)
        sd.point((x, y), fill=(int(b), int(b), int(b)))
    img += arr(stars)

    # Far skyline (towers, dim windows).
    far = Image.new('L', (W, H))
    fd = ImageDraw.Draw(far)
    farwin = Image.new('RGB', (W, H))
    fwd = ImageDraw.Draw(farwin)
    x = -20
    while x < W:
        w = rng.uniform(60, 180) * s
        h = rng.uniform(0.12, 0.34) * H
        top = shore - h
        fd.rectangle((x, top, x + w, shore), fill=255)
        if rng.random() < 0.3:
            fd.rectangle((x + w * 0.4, top - h * 0.12, x + w * 0.6, top), fill=255)
        for wy in np.arange(top + 10 * s, shore - 6 * s, 16 * s):
            for wx in np.arange(x + 6 * s, x + w - 8 * s, 14 * s):
                if rng.random() < 0.12:
                    c = rng.choice([(120, 90, 160), (90, 140, 170), (160, 100, 140)])
                    fwd.rectangle((wx, wy, wx + 5 * s, wy + 6 * s), fill=c)
        x += w + rng.uniform(-10, 30) * s
    img = composite(img, far, '#1b0d3a')
    img += arr(farwin) * 0.8

    # Haze over far skyline.
    band = np.exp(-((np.arange(H) - shore * 0.93) / (H * 0.11)) ** 2)[:, None, None]
    img += band * hexrgb('#ff4fa0') * 0.34

    # Near art-deco hotels along the drive.
    near = Image.new('L', (W, H))
    nd = ImageDraw.Draw(near)
    neon = Image.new('RGB', (W, H))
    gd = ImageDraw.Draw(neon)
    palettes = [
        dict(body=255, neon=(255, 46, 136), neon2=(40, 220, 255), windows=[(255, 190, 120), (120, 240, 255), (255, 120, 200)]),
        dict(body=255, neon=(40, 220, 255), neon2=(255, 70, 160), windows=[(255, 200, 140), (180, 120, 255)]),
        dict(body=255, neon=(255, 140, 60), neon2=(255, 46, 136), windows=[(255, 210, 150), (120, 240, 255)]),
    ]
    windows = []
    x = W * 0.04
    i = 0
    heights = [0.18, 0.31, 0.24, 0.38, 0.21, 0.28, 0.34, 0.2, 0.26]
    while x < W * 0.97:
        w = rng.uniform(260, 560) * s
        h = (heights[i % len(heights)] + rng.uniform(-0.03, 0.03)) * H
        windows += deco_building(nd, gd, x, w, shore - h, shore, rng, s, palettes[i % 3])
        x += w + rng.uniform(40, 160) * s
        i += 1
    # Facades: dusky violet catching the sky glow at the top, darker at street level.
    facade = vgrad(H, [(0.0, '#2c1450'), (0.35, '#2a1248'), (0.55, '#1a0c33'), (shore / H, '#0b0617'), (1.0, '#0b0617')])[:, None, :]
    m = arr(near)[..., None] / 255.0
    img = img * (1 - m) + facade * m
    # Vertical pilaster highlights inside the facades (neon spill).
    pil = Image.new('RGB', (W, H))
    pdw = ImageDraw.Draw(pil)
    for px in np.arange(0, W, 22 * s):
        pdw.line([(px, shore * 0.5), (px, shore)], fill=(26, 12, 40), width=max(1, int(2 * s)))
    img += arr(pil) * m * 0.9
    win = Image.new('RGB', (W, H))
    wd = ImageDraw.Draw(win)
    for (a, b, c, d, col) in windows:
        wd.rectangle((a, b, c, d), fill=col)
    img = add_glow(img, win, radii=((3, 0.5), (14, 0.25)))
    img = add_glow(img, neon, radii=((4, 1.0), (22, 0.8), (90, 0.45)))

    # Palms on the promenade.
    palm = Image.new('L', (W, H))
    pd = ImageDraw.Draw(palm)
    for px in np.arange(W * 0.02, W, W * 0.16):
        jitter = rng.uniform(-0.03, 0.03) * W
        draw_palm(pd, (px + jitter, shore + 6 * s), H * rng.uniform(0.3, 0.42), rng.uniform(-0.12, 0.12), rng, s * 0.75, fronds=10)
    img = composite(img, palm, '#05030c')

    # Promenade strip with a cyan neon rail.
    rail = Image.new('RGB', (W, H))
    ImageDraw.Draw(rail).line([(0, shore + 4 * s), (W, shore + 4 * s)], fill=(40, 220, 255), width=max(2, int(5 * s)))
    img[shore: shore + int(10 * s)] = hexrgb('#0a0516')
    img = add_glow(img, rail, radii=((5, 0.9), (30, 0.5)))

    # Ocean: reflection of everything above the shore, rippled and darkened.
    above = np.clip(img[:shore], 0, 255)
    refl_h = H - shore
    src = above[::-1][: refl_h]
    if src.shape[0] < refl_h:
        src = np.concatenate([src, np.repeat(src[-1:], refl_h - src.shape[0], 0)], 0)
    rows = np.arange(refl_h)
    out = np.empty_like(src)
    for r in rows:
        amp = (3 + 18 * (r / refl_h)) * s
        shift = int(amp * math.sin(r * 0.09 / s + math.sin(r * 0.013 / s) * 2))
        out[r] = np.roll(src[r], shift, axis=0)
    refl = to_img(out).filter(ImageFilter.GaussianBlur(4 * s))
    # Vertical smear: water stretches lights downward.
    refl = refl.resize((W, max(1, refl_h // 6)), Image.Resampling.BILINEAR).resize((W, refl_h), Image.Resampling.BICUBIC)
    refl = Image.blend(refl, to_img(out).filter(ImageFilter.GaussianBlur(2 * s)), 0.35)
    refl = arr(refl)
    depth = np.linspace(0, 1, refl_h)[:, None, None]
    water = vgrad(refl_h, [(0.0, '#1a0a33'), (1.0, '#03030a')])[:, None, :]
    refl = refl * (0.62 - 0.45 * depth) + water * (0.5 + 0.5 * depth)
    # Horizontal glints.
    gl = Image.new('RGB', (W, refl_h))
    gld = ImageDraw.Draw(gl)
    for _ in range(int(900 * s)):
        y = rng.uniform(0, refl_h) ** 1.0
        x = rng.uniform(0, W)
        ln = rng.uniform(20, 140) * s * (1 - y / refl_h * 0.6)
        col = rng.choice([(255, 80, 170), (60, 220, 255), (255, 170, 90), (255, 220, 240)])
        a = rng.uniform(0.15, 0.5) * (1 - y / refl_h)
        gld.line([(x, y), (x + ln, y)], fill=tuple(int(c * a) for c in col), width=max(1, int(2 * s)))
    refl += arr(gl)
    # Moon path on the water.
    mpath = np.exp(-((np.arange(W) - mx) / (W * 0.03)) ** 2)[None, :, None] * (1 - depth) ** 0.7
    refl += mpath * hexrgb('#ff9fd0') * 0.35
    img[shore:] = refl

    img = vignette(img, 0.4)
    img = grain(img, np.random.default_rng(rng.randrange(1 << 30)), 3.0)
    return to_img(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=1986)
    ap.add_argument('--scale', type=float, default=1.0, help='e.g. 0.25 for a fast draft')
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit('Choose a new output directory')
    os.makedirs(a.out)
    W, H = int(W0 * a.scale), int(H0 * a.scale)
    sunset_grid(W, H, random.Random(a.seed)).save(os.path.join(a.out, 'sunset-grid.png'))
    print('wrote sunset-grid.png')
    ocean_drive(W, H, random.Random(a.seed + 1)).save(os.path.join(a.out, 'ocean-drive.png'))
    print('wrote ocean-drive.png')


if __name__ == '__main__':
    main()
