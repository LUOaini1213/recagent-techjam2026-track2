# Demo video — built by code, not by screen recording

`recagent_demo.mp4` — 1920×1080, 24 fps, 3:00, English voiceover with
burned-in subtitles. Nobody held a screen recorder; four commands produce it.

```bash
python scripts/video_tts.py        # 1. narration -> audio + timing + SRT
python scripts/video_scenes.py     # 2. real artifacts -> 4320 PNG frames
python scripts/video_assemble.py   # 3. frames + audio + subs -> MP4
```

To change the video, edit **`docs/video/narration.json`** and rerun those
three. Nothing else needs touching — the text is the only authored input.

## Why it is built this way

**1 — Narration first, timing drives everything.** Each segment is synthesized
with Microsoft's `en-US-AndrewMultilingualNeural` neural voice, and its real
duration is measured from the rendered audio. Those durations decide how long
every scene runs, so the picture can never drift from the voice.

The speaking rate auto-calibrates to fit each slot, with one rule: **never
slow speech below its natural pace to fill time.** Copy that runs short simply
leaves a pause; only overlong copy is sped up, capped at +18% so it never
sounds rushed. The run prints the result per segment and flags `OVERRUN`:

```
demo_leak     slot  24.0s  audio  22.6s  rate  +0%  143 wpm  54 words
results       slot  25.0s  audio  24.4s  rate +14%  177 wpm  72 words
```

**2 — Subtitles come from the engine, not from a guess.** `edge-tts` is run
with `boundary='WordBoundary'`, so every word carries an offset and duration.
Captions are cut on those real timings and on sentence ends. Because word
events arrive without punctuation, the caption text is re-joined from the
original narration tokens — otherwise captions read as fragments
(*"the data Build a feature Train Evaluate"* instead of
*"Look at the data. Build a feature."*).

**3 — The picture is rendered from real repository data.** There is no GUI to
record: the agent is a pipeline, so the honest visual is the pipeline's own
output. Every number on screen is a real one from `results/official/` and
`logs/official_runs.jsonl` — the feature-importance bars in the leak scene are
the actual gains that exposed the bug (`cf_mean` 1,456,859 against 343,228),
and the result figures are the real ones. If the repo's numbers change, the
video changes with them.

**4 — Assembly with a hard gate.** ffmpeg (portable, via `imageio-ffmpeg` — no
system install) concatenates the scenes, burns the subtitles, and muxes the
voiceover. The last thing `video_assemble.py` does is assert the runtime:

```python
if abs(got - target) > TOL:
    os.remove(out)
    raise SystemExit(f'REJECTED: {got:.1f}s is outside {target:.0f}s ...')
```

A cut that misses 3:00 is **deleted**, not shipped.

## Two deliberate design calls

- **The magnifier inset in the Results scene.** Baseline 0.5946 and ours
  0.6015 are only 1.8% apart on the 0.4753 → 0.8645 range, so plotting both on
  one axis puts the markers on top of each other and reads as "no progress".
  The inset magnifies exactly that gap and labels it `+0.0069`, while the main
  axis still carries the honest message: the ceiling is 0.8645, not 1.0.
- **The leak beat gets 24 seconds**, the longest segment in the film. A score
  that goes up is unremarkable; an agent that watches its own score collapse,
  reads feature importances, names the leak, and fixes itself is the thing
  worth showing.

## Files

| File | What it is |
|---|---|
| `narration.json` | the only authored input — text, slots, scene per segment |
| `recagent_demo.mp4` | the finished film |
| `voiceover.mp3` / `.srt` | audio track and subtitles, usable separately |
| `seg_*.mp3` | per-segment audio, for editing scene by scene |
| `timing.json` | measured duration, rate and words per segment |
| `frames/<scene>/` | rendered PNG frames |
