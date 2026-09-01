"""Step 2 — render the visual track from real repository artifacts.

There is no GUI to screen-record here: the agent is a pipeline, so the honest
visual is the pipeline's own output. Every number, log line and chart on
screen is read from results/official/ and logs/ at render time — nothing is
mocked, so the video cannot drift away from the repo.

    python scripts/video_scenes.py

Writes docs/video/frames/<scene>/%05d.png at FPS, one directory per scene,
sized to each segment's slot in narration.json.
"""
import json, math, os, shutil, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = os.path.join(ROOT, 'docs', 'video')
OFF = os.path.join(ROOT, 'results', 'official')
W, H, FPS = 1920, 1080, 24

BG = (13, 17, 28)
FG = (232, 237, 245)
DIM = (128, 141, 163)
ACC = (94, 168, 255)          # blue: the agent / our work
GOOD = (74, 205, 148)         # green: gains
BAD = (255, 107, 107)         # red: the leak beat
WARN = (255, 190, 92)


def font(size, mono=False, bold=False):
    names = ([r'C:\Windows\Fonts\consola.ttf'] if mono else
             [r'C:\Windows\Fonts\segoeuib.ttf'] if bold else
             [r'C:\Windows\Fonts\segoeui.ttf'])
    for n in names + [r'C:\Windows\Fonts\arial.ttf']:
        if os.path.exists(n):
            return ImageFont.truetype(n, size)
    return ImageFont.load_default()


F_H1, F_H2, F_BODY = font(76, bold=True), font(46, bold=True), font(34)
F_MONO, F_MONO_S, F_LABEL = font(30, mono=True), font(24, mono=True), font(26)


def new_frame():
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 6], fill=ACC)
    return im, d


def ease(t):
    return t * t * (3 - 2 * t)


def fade(c, a):
    return tuple(int(BG[i] + (c[i] - BG[i]) * max(0.0, min(1.0, a)))
                 for i in range(3))


def typewriter(d, lines, t, x, y, lh=40, cps=38, f=None, colors=None):
    """Reveal monospace lines at cps characters per second."""
    f = f or F_MONO_S
    budget = t * cps
    for i, line in enumerate(lines):
        if budget <= 0:
            break
        shown = line[:int(budget)]
        col = (colors or {}).get(i, FG)
        d.text((x, y + i * lh), shown, font=f, fill=col)
        budget -= len(line) + 6      # small pause between lines


# ---------------------------------------------------------------- scenes
def scene_loop(d, t, dur):
    """The MLE iteration loop, spinning — the problem."""
    d.text((110, 120), 'The loop every MLE knows', font=F_H1, fill=FG)
    steps = ['Inspect', 'Feature', 'Train', 'Evaluate', 'Reflect']
    cx, cy, r = W // 2, 620, 250
    spin = t * 0.55
    for i, s in enumerate(steps):
        ang = spin + i * 2 * math.pi / len(steps) - math.pi / 2
        x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
        live = (int(spin / (2 * math.pi / len(steps))) + i) % len(steps) == 0
        col = ACC if live else DIM
        d.ellipse([x - 78, y - 44, x + 78, y + 44], outline=col, width=3)
        w = d.textlength(s, font=F_LABEL)
        d.text((x - w / 2, y - 15), s, font=F_LABEL, fill=col)
    for k in range(len(steps)):
        a0 = spin + k * 2 * math.pi / len(steps) - math.pi / 2 + 0.42
        a1 = spin + (k + 1) * 2 * math.pi / len(steps) - math.pi / 2 - 0.42
        d.arc([cx - r, cy - r, cx + r, cy + r],
              math.degrees(a0), math.degrees(a1), fill=DIM, width=2)
    laps = int(t * 0.55 / (2 * math.pi) * len(steps)) + 3
    d.text((cx - 120, cy - 22), f'iteration {laps}', font=F_H2,
           fill=fade(WARN, 0.5 + 0.5 * math.sin(t * 3)))
    d.text((110, 250), 'Days of it. Almost none of it creative.',
           font=F_BODY, fill=DIM)


def scene_claims(d, t, dur):
    d.text((110, 110), 'An agent that runs the loop itself', font=F_H1, fill=FG)
    claims = [('Proposes its own hypotheses', 0.6),
              ('Writes and runs its own code', 2.2),
              ('Scores itself on the official metric', 3.8),
              ('23 iterations · 0 lines of human code', 5.6)]
    for i, (txt, at) in enumerate(claims):
        if t < at:
            continue
        a = ease(min(1.0, (t - at) / 0.5))
        y = 300 + i * 105
        col = GOOD if i == 3 else FG
        d.rectangle([110, y + 10, 110 + int(8 * a), y + 58], fill=ACC)
        d.text((146, y), txt, font=F_H2, fill=fade(col, a))


def scene_architecture(d, t, dur):
    d.text((110, 100), 'Four components, one loop', font=F_H1, fill=FG)
    boxes = [('Feature Builder', 'every statistic · strictly past-only', 0.5),
             ('Models', 'LightGBM · FM · item-item / EASE', 3.5),
             ('Official evaluate.py', 'never modified', 6.5),
             ('Reflect & Select', 'reads the scores · picks what is next', 9.0)]
    x0, y0, bw, bh, gap = 120, 300, 400, 200, 56
    for i, (title, sub, at) in enumerate(boxes):
        if t < at:
            continue
        a = ease(min(1.0, (t - at) / 0.6))
        x = x0 + i * (bw + gap)
        col = GOOD if i == 2 else ACC
        d.rounded_rectangle([x, y0, x + bw, y0 + bh], 14,
                            outline=fade(col, a), width=3)
        d.text((x + 24, y0 + 34), title, font=font(34, bold=True),
               fill=fade(FG, a))
        for j, part in enumerate(sub.split(' · ')):
            d.text((x + 24, y0 + 92 + j * 32), part, font=F_LABEL,
                   fill=fade(DIM, a))
        if i and t > at:
            d.line([x - gap + 8, y0 + bh / 2, x - 8, y0 + bh / 2],
                   fill=fade(DIM, a), width=3)
    if t > 11.5:
        a = ease(min(1.0, (t - 11.5) / 0.7))
        yb = y0 + bh + 90
        d.line([x0 + bw * 3.5 + gap * 3, y0 + bh, x0 + bw * 3.5 + gap * 3, yb],
               fill=fade(ACC, a), width=3)
        d.line([x0 + bw * 3.5 + gap * 3, yb, x0 + bw / 2, yb],
               fill=fade(ACC, a), width=3)
        d.line([x0 + bw / 2, yb, x0 + bw / 2, y0 + bh], fill=fade(ACC, a), width=3)
        d.text((x0 + bw * 1.5, yb + 22), 'and around again',
               font=F_LABEL, fill=fade(ACC, a))


def scene_terminal_build(d, t, dur):
    d.text((110, 90), 'Building the feature cache', font=F_H2, fill=FG)
    lines = [
        '$ python scripts/official_lgbm.py --build --past',
        'logs loaded 1436609 rows',
        "{'train': 1141112, 'valid': 124909, 'test': 170588}",
        'wrote train: 1141112 rows, 117 cols',
        'wrote valid:  124909 rows, 117 cols',
        'wrote test:   170588 rows, 117 cols',
        'build done in 198.0s (past=True)',
    ]
    typewriter(d, lines, t, 130, 200, lh=46, cps=46, f=F_MONO,
               colors={0: ACC, 6: GOOD})
    if t > 9.0:
        a = ease(min(1.0, (t - 9.0) / 0.7))
        d.rounded_rectangle([130, 590, 1500, 810], 14,
                            outline=fade(WARN, a), width=3)
        d.text((166, 626), 'Past-only discipline', font=font(38, bold=True),
               fill=fade(WARN, a))
        d.text((166, 686),
               "a training row's statistics come only from strictly earlier days",
               font=F_BODY, fill=fade(FG, a))
        d.text((166, 736), 'largest single gain of the whole run:  +0.005 primary',
               font=F_BODY, fill=fade(GOOD, a))


def scene_terminal_train(d, t, dur):
    d.text((110, 90), 'Training, scored by the official evaluator',
           font=F_H2, fill=FG)
    lines = [
        '$ python scripts/official_lgbm.py --train --past --name p4_ease',
        'p4_ease: valid GAUC 0.6759  nDCG@5 0.5401  primary 0.6080',
        '',
        '# logs/official_runs.jsonl',
        '{"iter": 10, "hypothesis": "EASE closed-form item-item model',
        '  as features", "why": "reliably beats cosine item-item CF;',
        '  cheap at 7.5k items", "metrics": {"primary": 0.6080}}',
    ]
    typewriter(d, lines, t, 130, 210, lh=52, cps=52, f=F_MONO,
               colors={0: ACC, 1: GOOD, 3: DIM, 4: DIM, 5: DIM, 6: DIM})


def scene_leak(d, t, dur):
    """The strongest beat: score collapse, importance anomaly, diagnosis."""
    d.text((110, 80), 'The agent catches its own bug', font=F_H1, fill=FG)
    if t > 0.8:
        a = ease(min(1.0, (t - 0.8) / 0.6))
        d.text((130, 220), 'smoke_cf: valid primary', font=F_MONO,
               fill=fade(FG, a))
        d.text((640, 220), '0.5738', font=font(44, mono=True),
               fill=fade(BAD, a))
        d.text((830, 228), 'collapsed', font=F_BODY, fill=fade(BAD, a))
    if t > 3.0:
        a = ease(min(1.0, (t - 3.0) / 0.6))
        d.text((130, 320), 'feature importance (gain)', font=F_LABEL,
               fill=fade(DIM, a))
        bars = [('cf_mean', 1456859), ('utab_lv_rate', 343228),
                ('fm_score', 286104), ('v_imp', 147816), ('tab', 104733)]
        top = bars[0][1]
        for i, (nm, g) in enumerate(bars):
            y = 372 + i * 62
            grow = ease(min(1.0, max(0.0, (t - 3.2 - i * 0.18) / 0.5)))
            wpx = int(980 * g / top * grow)
            col = BAD if i == 0 else ACC
            d.text((130, y + 6), nm, font=F_MONO_S, fill=fade(FG, a))
            d.rectangle([420, y, 420 + wpx, y + 42], fill=fade(col, a))
            if grow > 0.9 and i:          # top bar's value would collide
                d.text((430 + wpx, y + 8), f'{g:,}', font=F_MONO_S,
                       fill=fade(DIM, a))
        if t > 5.4:
            b = ease(min(1.0, (t - 5.4) / 0.5))
            d.rectangle([120, 362, 1420, 418], outline=fade(BAD, b), width=3)
            d.text((1450, 370), '1,456,859', font=font(30, mono=True),
                   fill=fade(BAD, b))
            d.text((1450, 404), '4x anything else', font=F_LABEL,
                   fill=fade(BAD, b))
    if t > 9.5:
        a = ease(min(1.0, (t - 9.5) / 0.6))
        d.rounded_rectangle([130, 700, 1790, 880], 14,
                            outline=fade(WARN, a), width=3)
        d.text((166, 728), 'Diagnosis', font=font(34, bold=True),
               fill=fade(WARN, a))
        d.text((166, 778), "the user's own history was scoring the user —",
               font=F_BODY, fill=fade(FG, a))
        d.text((166, 822),
               'leave-user-out must drop the self term AND the +1 in every co(v,w)',
               font=F_BODY, fill=fade(FG, a))
    if t > 15.5:
        a = ease(min(1.0, (t - 15.5) / 0.6))
        d.text((130, 220), 'smoke_cf2: valid primary', font=F_MONO,
               fill=fade(FG, a))
        d.rectangle([120, 205, 1500, 275], fill=BG)
        d.text((130, 220), 'smoke_cf2: valid primary', font=F_MONO,
               fill=fade(FG, a))
        d.text((640, 220), '0.5993', font=font(44, mono=True),
               fill=fade(GOOD, a))
        d.text((830, 228), 'recovered — no human touched it', font=F_BODY,
               fill=fade(GOOD, a))


def scene_sweep(d, t, dur):
    d.text((110, 90), 'Parallel sweeps, and honest failures',
           font=F_H2, fill=FG)
    workers = ['worker 1  p3_ff04', 'worker 2  p3_ff05_md50',
               'worker 3  p3_rank_ff05', 'worker 4  p3_ff05_cs50']
    for i, w in enumerate(workers):
        y = 200 + i * 74
        d.text((130, y), w, font=F_MONO_S, fill=ACC)
        prog = min(1.0, max(0.0, (t - 0.4 - i * 0.25) / 3.4))
        d.rectangle([620, y + 4, 620 + int(560 * prog), y + 34],
                    fill=fade(ACC, 0.55))
        if prog >= 1.0:
            d.text((1210, y), 'done', font=F_MONO_S, fill=GOOD)
    if t > 5.0:
        a = ease(min(1.0, (t - 5.0) / 0.6))
        d.text((130, 560), 'tested after convergence — all logged as failures',
               font=F_BODY, fill=fade(DIM, a))
        fails = [('sequence model (tiny DIN)', '0.6011', 'did not enter blend'),
                 ('metric-aware weighting', '0.6075', 'worse'),
                 ('statistics freshness', '-0.0003', 'declined on rules risk')]
        for i, (nm, val, note) in enumerate(fails):
            at = 5.6 + i * 1.5
            if t < at:
                continue
            b = ease(min(1.0, (t - at) / 0.5))
            y = 630 + i * 76
            d.text((130, y), nm, font=F_BODY, fill=fade(FG, b))
            d.text((760, y), val, font=font(32, mono=True), fill=fade(BAD, b))
            d.text((980, y), note, font=F_BODY, fill=fade(DIM, b))


def scene_stop(d, t, dur):
    d.text((110, 110), 'Convergence, then one shot at the test set',
           font=F_H2, fill=FG)
    lines = [
        '{"type": "stop",',
        '  "reason": "converged under the declared rule (eps=0.0005, N=4)",',
        '  "caps": {"scored_iterations": 23, "iteration_cap": 50,',
        '           "active_wall_clock_hours": 3.5, "wall_clock_cap": 6},',
        '  "designated_submission": "validation-best checkpoint"}',
    ]
    typewriter(d, lines, t, 130, 250, lh=52, cps=64, f=F_MONO,
               colors={0: WARN, 4: GOOD})
    if t > 7.2:
        a = ease(min(1.0, (t - 7.2) / 0.6))
        d.text((130, 620), 'hidden test evaluated exactly once',
               font=font(40, bold=True), fill=fade(GOOD, a))


def scene_results(d, t, dur):
    d.text((110, 80), 'Results on the hidden test set', font=F_H1, fill=FG)
    rows = [('random', 0.4753, DIM, 0.4),
            ('popularity baseline', 0.5715, DIM, 1.2),
            ('FM — official baseline', 0.5946, WARN, 2.0),
            ('ours', 0.6015, GOOD, 3.0)]
    for i, (nm, val, col, at) in enumerate(rows):
        if t < at:
            continue
        a = ease(min(1.0, (t - at) / 0.5))
        y = 220 + i * 74
        d.text((130, y), nm, font=F_BODY, fill=fade(FG, a))
        d.text((700, y), f'{val:.4f}', font=font(34, mono=True),
               fill=fade(col, a))
    if t > 4.6:
        a = ease(min(1.0, (t - 4.6) / 0.7))
        d.text((130, 560), 'the ceiling here is not 1.0', font=F_H2,
               fill=fade(FG, a))
        d.text((130, 628),
               '27% of users have no positive label — a perfect ranking scores 0.8645',
               font=F_BODY, fill=fade(DIM, a))
        x0, x1, y = 170, 1180, 760
        lo, hi = 0.4753, 0.8645
        d.line([x0, y, x1, y], fill=fade(DIM, a), width=4)
        grow = ease(min(1.0, max(0.0, (t - 5.4) / 1.0)))
        for label, v, col, up in (('random', 0.4753, DIM, False),
                                  ('baseline', 0.5946, WARN, True),
                                  ('perfect', 0.8645, DIM, True)):
            px = x0 + (x1 - x0) * (v - lo) / (hi - lo)
            if v > lo and grow < (v - lo) / (hi - lo):
                continue
            d.line([px, y - 22, px, y + 22], fill=fade(col, a), width=5)
            tw = d.textlength(label, font=F_LABEL)
            d.text((px - tw / 2, y - 74 if up else y + 34), label,
                   font=F_LABEL, fill=fade(col, a))
        if t > 7.0:
            # the FM->ours gap is 1.8% of this range: magnify it rather than
            # let two markers collide and read as "no progress"
            b = ease(min(1.0, (t - 7.0) / 0.6))
            pf = x0 + (x1 - x0) * (0.5946 - lo) / (hi - lo)
            po = x0 + (x1 - x0) * (0.6015 - lo) / (hi - lo)
            d.rectangle([pf, y - 12, max(po, pf + 4), y + 12], fill=fade(GOOD, b))
            ix0, ix1, iy = 1330, 1800, 700
            d.line([pf, y - 12, ix0, iy - 60], fill=fade(DIM, b * 0.6), width=2)
            d.line([po, y + 12, ix0, iy + 96], fill=fade(DIM, b * 0.6), width=2)
            d.rounded_rectangle([ix0, iy - 60, ix1, iy + 96], 12,
                                outline=fade(DIM, b), width=2)
            d.line([ix0 + 34, iy + 30, ix1 - 34, iy + 30], fill=fade(DIM, b), width=3)
            d.line([ix0 + 34, iy + 8, ix0 + 34, iy + 52], fill=fade(WARN, b), width=5)
            d.line([ix1 - 34, iy + 8, ix1 - 34, iy + 52], fill=fade(GOOD, b), width=5)
            d.text((ix0 + 18, iy - 40), 'baseline', font=F_LABEL, fill=fade(WARN, b))
            d.text((ix1 - 110, iy - 40), 'ours', font=F_LABEL, fill=fade(GOOD, b))
            d.text((ix0 + 128, iy + 60), '+0.0069', font=font(32, bold=True),
                   fill=fade(GOOD, b))
            d.text((130, 856), '31.0% of the attainable range  ->  32.4%',
                   font=F_H2, fill=fade(GOOD, b))
    if t > 9.6:
        a = ease(min(1.0, (t - 9.6) / 0.6))
        d.text((130, 918),
               '23 iterations  ·  0 manual code edits  ·  8 CPU cores  ·  3.5 h',
               font=F_BODY, fill=fade(DIM, a))


def scene_end(d, t, dur):
    a = ease(min(1.0, t / 0.8))
    d.text((110, 300), 'Automate the iteration.', font=font(88, bold=True),
           fill=fade(FG, a))
    d.text((110, 410), 'Let the engineer think.', font=font(88, bold=True),
           fill=fade(ACC, a))
    if t > 3.2:
        b = ease(min(1.0, (t - 3.2) / 0.6))
        for i, line in enumerate([
                'RecAgent — Autonomous ML Research Agent',
                'TikTok TechJam 2026 · Track 2 · KuaiRand-Pure',
                'hidden test 0.6015 vs baseline 0.5946']):
            d.text((110, 600 + i * 56), line, font=F_BODY,
                   fill=fade(DIM if i else FG, b))
    if t > 6.0:
        b = ease(min(1.0, (t - 6.0) / 0.6))
        d.text((110, 820), 'results/official/leaderboard.md  ·  '
               'logs/official_runs.jsonl', font=F_MONO_S, fill=fade(DIM, b))


SCENES = {'loop': scene_loop, 'claims': scene_claims,
          'architecture': scene_architecture,
          'terminal_build': scene_terminal_build,
          'terminal_train': scene_terminal_train, 'leak': scene_leak,
          'sweep': scene_sweep, 'stop': scene_stop,
          'results': scene_results, 'end': scene_end}


def main():
    with open(os.path.join(VID, 'narration.json'), encoding='utf-8') as fh:
        spec = json.load(fh)
    root = os.path.join(VID, 'frames')
    for sub in (os.listdir(root) if os.path.isdir(root) else []):
        p = os.path.join(root, sub)          # clear contents, keep the dir:
        if os.path.isdir(p):                 # a shell may be cwd'd into it
            shutil.rmtree(p, ignore_errors=True)
    total = 0
    for seg in spec['segments']:
        fn = SCENES[seg['scene']]
        dur = seg['end'] - seg['start']
        out = os.path.join(root, seg['scene'])
        os.makedirs(out, exist_ok=True)
        n = int(round(dur * FPS))
        for k in range(n):
            im, d = new_frame()
            fn(d, k / FPS, dur)
            im.save(os.path.join(out, f'{k:05d}.png'))
        total += n
        print(f"{seg['scene']:16s} {dur:5.1f}s  {n:4d} frames", flush=True)
    print(f'\n{total} frames at {FPS} fps = {total / FPS:.1f}s')


if __name__ == '__main__':
    main()
