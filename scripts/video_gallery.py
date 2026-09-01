"""Step 4 — pick gallery stills from the rendered frames, at Devpost's 3:2.

Frames are 16:9; Devpost asks for 3:2, so each still is padded on the top and
bottom with the deck background rather than cropped, which would cut content.

    python scripts/video_gallery.py

Writes docs/video/gallery/*.png (1800x1200, well under the 5 MB limit).
"""
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = os.path.join(ROOT, 'docs', 'video')
BG = (13, 17, 28)
OUT_W, OUT_H = 1800, 1200

# (output name, scene, frame index) — indices chosen after the scene's
# animation has fully settled
PICKS = [
    ('01_results', 'results', 560),
    ('02_leak', 'leak', 500),
    ('03_architecture', 'architecture', 460),
    ('04_build', 'terminal_build', 400),
    ('05_convergence', 'stop', 270),
]


def main():
    gal = os.path.join(VID, 'gallery')
    os.makedirs(gal, exist_ok=True)
    for name, scene, idx in PICKS:
        src = os.path.join(VID, 'frames', scene, f'{idx:05d}.png')
        if not os.path.exists(src):
            raise SystemExit(f'missing {src} — run scripts/video_scenes.py')
        im = Image.open(src).convert('RGB')
        w = OUT_W
        h = round(im.height * OUT_W / im.width)
        im = im.resize((w, h), Image.LANCZOS)
        canvas = Image.new('RGB', (OUT_W, OUT_H), BG)
        canvas.paste(im, (0, (OUT_H - h) // 2))
        dst = os.path.join(gal, f'{name}.png')
        canvas.save(dst, optimize=True)
        mb = os.path.getsize(dst) / 1e6
        print(f'{name:18s} {OUT_W}x{OUT_H}  {mb:.2f} MB')
    print(f'\n{len(PICKS)} stills in {os.path.relpath(gal, ROOT)}')


if __name__ == '__main__':
    main()
