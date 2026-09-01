"""Step 3 — assemble frames + voiceover + burned-in subtitles into the MP4.

    python scripts/video_assemble.py

Hard assertion at the end: if the finished file is not 180 seconds (within
tolerance) it is deleted and the script exits non-zero, so a broken cut can
never be handed in by accident.

Writes docs/video/recagent_demo.mp4 (H.264 + AAC, 1920x1080, 24 fps).
"""
import json, os, subprocess, sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = os.path.join(ROOT, 'docs', 'video')
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 24
TOL = 0.6            # seconds of slack allowed against the declared runtime


def run(args):
    p = subprocess.run(args, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if p.returncode:
        print(p.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f'ffmpeg failed: {" ".join(args[:6])} ...')
    return p


def duration(path):
    p = subprocess.run([FFMPEG, '-i', path, '-f', 'null', '-'],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    for line in reversed(p.stderr.splitlines()):
        if 'time=' in line:
            h, m, s = line.split('time=')[1].split(' ')[0].split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f'no duration for {path}')


def main():
    with open(os.path.join(VID, 'narration.json'), encoding='utf-8') as fh:
        spec = json.load(fh)
    target = float(spec['total_seconds'])

    # 1. concatenate the per-scene frame directories, in narration order
    concat = os.path.join(VID, 'frames', 'concat.txt')
    silent = os.path.join(VID, 'video_track.mp4')
    parts = []
    for i, seg in enumerate(spec['segments']):
        sd = os.path.join(VID, 'frames', seg['scene'])
        n = len([f for f in os.listdir(sd) if f.endswith('.png')])
        want = int(round((seg['end'] - seg['start']) * FPS))
        if n != want:
            raise SystemExit(f"scene {seg['scene']}: {n} frames, expected "
                             f"{want} — rerun scripts/video_scenes.py")
        part = os.path.join(VID, 'frames', f'part_{i:02d}.mp4')
        run([FFMPEG, '-y', '-framerate', str(FPS),
             '-i', os.path.join(sd, '%05d.png'),
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', part])
        parts.append(part)
    with open(concat, 'w', encoding='utf-8') as fh:
        for p in parts:
            fh.write(f"file '{p.replace(os.sep, '/')}'\n")
    run([FFMPEG, '-y', '-f', 'concat', '-safe', '0', '-i', concat,
         '-c', 'copy', silent])
    print(f'video track {duration(silent):.1f}s', flush=True)

    # 2. burn subtitles, then mux the voiceover
    srt = os.path.join(VID, 'voiceover.srt').replace('\\', '/')
    srt = srt.replace(':', '\\:')          # ffmpeg filter-arg escaping
    # The subtitles filter lays out in a 384x288 script space unless told
    # otherwise, so sizes here are ~3.75x smaller than the 1080p pixels they
    # become: FontSize 13 renders ~49 px, MarginV 15 sits ~56 px off the floor.
    style = ('FontName=Segoe UI Semibold,FontSize=13,PrimaryColour=&H00F5EDE8,'
             'OutlineColour=&H00140F0A,BackColour=&H90140F0A,BorderStyle=3,'
             'Outline=3,Shadow=0,Alignment=2,MarginV=15')
    out = os.path.join(VID, 'recagent_demo.mp4')
    run([FFMPEG, '-y', '-i', silent, '-i', os.path.join(VID, 'voiceover.mp3'),
         '-vf', f"subtitles='{srt}':force_style='{style}'",
         '-map', '0:v', '-map', '1:a',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
         '-c:a', 'aac', '-b:a', '192k', '-shortest', out])

    got = duration(out)
    size = os.path.getsize(out) / 1e6
    print(f'\n{os.path.relpath(out, ROOT)}  {got:.1f}s  {size:.1f} MB')
    if abs(got - target) > TOL:
        os.remove(out)
        raise SystemExit(f'REJECTED: {got:.1f}s is outside {target:.0f}s '
                         f'+/-{TOL}s. File deleted, nothing to submit.')
    print(f'OK: within {TOL}s of the declared {target:.0f}s runtime')


if __name__ == '__main__':
    main()
