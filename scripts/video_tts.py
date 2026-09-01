"""Step 1 — narration to audio. Timing measured here drives every later step.

Reads docs/video/narration.json, synthesizes each segment with edge-tts, and
auto-calibrates the speaking rate until the audio fits its slot. Word-boundary
events from the engine give millisecond-accurate subtitle timings, so the
subtitles match the audio that was actually produced rather than an estimate.

    python scripts/video_tts.py

Writes docs/video/: seg_*.mp3, voiceover.mp3, voiceover.srt, timing.json
"""
import asyncio, json, os, subprocess

import edge_tts
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = os.path.join(ROOT, 'docs', 'video')
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TAIL = 0.45          # breath left at the end of each slot
TOL = 0.35           # accept when within this many seconds of target
MAX_SPEEDUP = 18     # beyond this the delivery starts to sound rushed


def duration(path):
    out = subprocess.run([FFMPEG, '-i', path, '-f', 'null', '-'],
                         capture_output=True, text=True, encoding='utf-8',
                         errors='replace')
    for line in reversed(out.stderr.splitlines()):
        if 'time=' in line:
            h, m, s = line.split('time=')[1].split(' ')[0].split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f'no duration for {path}')


async def synth(text, voice, rate, path):
    comm = edge_tts.Communicate(text, voice, rate=rate,
                                boundary='WordBoundary')
    marks = []
    with open(path, 'wb') as fh:
        async for ch in comm.stream():
            if ch['type'] == 'audio':
                fh.write(ch['data'])
            elif ch['type'] == 'WordBoundary':
                marks.append((ch['offset'] / 1e7, ch['duration'] / 1e7,
                              ch['text']))
    return marks


def srt_time(t):
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round(s % 1 * 1000)):03d}'


def group_lines(marks, base, text, max_chars=58, max_gap=0.6):
    """Pack words into caption lines with the ORIGINAL punctuation.

    Word-boundary events carry bare words, so captions built from them read as
    fragments. The engine emits one mark per whitespace token, so when the
    counts agree we substitute the punctuated tokens and prefer to break where
    a sentence actually ends.
    """
    tokens = text.split()
    use_src = len(tokens) == len(marks)
    lines, cur, start, last = [], [], None, None

    def flush():
        nonlocal cur, start
        if cur:
            lines.append((base + start, base + last, ' '.join(cur)))
            cur, start = [], None

    for i, (off, dur, word) in enumerate(marks):
        w = tokens[i] if use_src else word
        if cur and (len(' '.join(cur)) + 1 + len(w) > max_chars
                    or off - last > max_gap):
            flush()
        if start is None:
            start = off
        cur.append(w)
        last = off + dur
        # a finished sentence is a natural caption break once the line has body
        if use_src and w.endswith(('.', '!', '?')) and len(' '.join(cur)) > 24:
            flush()
    flush()
    return lines


async def main():
    with open(os.path.join(VID, 'narration.json'), encoding='utf-8') as fh:
        spec = json.load(fh)
    voice = spec['voice']
    os.makedirs(VID, exist_ok=True)
    report, lines, parts = [], [], []

    for i, seg in enumerate(spec['segments'], 1):
        slot = seg['end'] - seg['start']
        target = slot - TAIL
        path = os.path.join(VID, f"seg_{i:02d}_{seg['name']}.mp3")
        # Never slow speech below its natural pace to fill a slot — that reads
        # as draggy. Short copy simply leaves a pause before the next beat.
        # Only speed up, and only as much as the slot actually demands.
        marks = await synth(seg['text'], voice, '+0%', path)
        natural, rate = duration(path), 0
        for _ in range(6):
            if natural <= target + TOL:
                want = 0
            else:
                want = min(MAX_SPEEDUP, round((natural / target - 1) * 100))
            if want == rate:
                break
            rate = want
            marks = await synth(seg['text'], voice, f'{rate:+d}%', path)
            got = duration(path)
            if got <= target + TOL:
                break
            natural = got * (1 + rate / 100)
        got = duration(path)
        if not marks:
            raise RuntimeError(f"no word boundaries for '{seg['name']}' — "
                               "subtitles would be empty")
        wpm = round(len(seg['text'].split()) / (got / 60))
        report.append({'segment': seg['name'], 'scene': seg['scene'],
                       'start': seg['start'], 'slot_s': round(slot, 2),
                       'audio_s': round(got, 2), 'rate': f'{rate:+d}%',
                       'wpm': wpm, 'words': len(marks)})
        over = ' OVERRUN' if got > slot else ''
        print(f"{seg['name']:13s} slot {slot:5.1f}s  audio {got:5.1f}s  "
              f"rate {rate:+3d}%  {wpm:3d} wpm  {len(marks)} words{over}",
              flush=True)
        lines += group_lines(marks, seg['start'], seg['text'])
        parts.append((seg['start'], path))

    inputs, filters = [], []
    for idx, (start, path) in enumerate(parts):
        inputs += ['-i', path]
        ms = int(start * 1000)
        filters.append(f'[{idx}:a]adelay={ms}|{ms}[a{idx}]')
    graph = (';'.join(filters) + ';'
             + ''.join(f'[a{i}]' for i in range(len(parts)))
             + f'amix=inputs={len(parts)}:normalize=0[out]')
    full = os.path.join(VID, 'voiceover.mp3')
    subprocess.run([FFMPEG, '-y', *inputs, '-filter_complex', graph,
                    '-map', '[out]', '-t', str(spec['total_seconds']),
                    '-b:a', '192k', full], capture_output=True, check=True)

    with open(os.path.join(VID, 'voiceover.srt'), 'w', encoding='utf-8') as fh:
        for n, (s, e, txt) in enumerate(lines, 1):
            fh.write(f'{n}\n{srt_time(s)} --> {srt_time(e)}\n{txt}\n\n')
    with open(os.path.join(VID, 'timing.json'), 'w', encoding='utf-8') as fh:
        json.dump({'voice': voice, 'total_s': round(duration(full), 2),
                   'segments': report, 'subtitle_lines': len(lines)},
                  fh, indent=2)
    print(f'\ntrack {duration(full):.1f}s · {len(lines)} subtitle lines · '
          f'{sum(r["words"] for r in report)} words timed')


if __name__ == '__main__':
    asyncio.run(main())
