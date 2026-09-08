"""Render logs/official_runs.jsonl as a table a reader can scan in a minute.

    python scripts/official_runlog_md.py            # writes logs/RUN_LOG.md
    python scripts/official_runlog_md.py --stdout   # print instead

The JSONL is the deliverable (compiled from results/official/*.json by
official_runlog.py); this file is a view of it and carries no number that is
not in the JSONL.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'logs', 'official_runs.jsonl')
MD = os.path.join(ROOT, 'logs', 'RUN_LOG.md')

OUTCOME_TAGS = ['ERROR', 'RECOVERY', 'CONFIRMED', 'NEGATIVE', 'NEUTRAL', 'DECLINED', 'ABANDONED']


def load(path=LOG):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def outcome(row):
    text = ' '.join(str(row.get(k, '')) for k in ('change', 'hypothesis', 'why', 'metrics_note'))
    tags = [t for t in OUTCOME_TAGS if t in text.upper()]
    return ' / '.join(tags) if tags else ''


def change_text(row):
    """The change column; an iteration without a scored metric carries its
    result in `metrics_note` (blend explorations, the freshness measurement)."""
    change = row.get('change', '')
    note = row.get('metrics_note')
    return f'{change} — {note}' if note else change


def primary_of(row):
    m = row.get('metrics')
    if isinstance(m, dict) and 'primary' in m:
        return m['primary']
    v = row.get('valid')
    if isinstance(v, dict) and 'primary' in v:
        return v['primary']
    if isinstance(row.get('valid_primary'), (int, float)):
        return row['valid_primary']
    return None


def wall_seconds(row):
    subs = row.get('sub_runs') or []
    secs = [s.get('wall_seconds') for s in subs if isinstance(s.get('wall_seconds'), (int, float))]
    return sum(secs) if secs else None


def esc(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def short(text, n=110):
    text = esc(text)
    return text if len(text) <= n else text[: n - 1] + '…'


def render(rows):
    header = next((r for r in rows if r.get('type') == 'header'), {})
    footer = next((r for r in rows if r.get('type') == 'footer'), {})
    stop = next((r for r in rows if r.get('type') == 'stop'), {})
    out = ['# Run log — one line per scored iteration', '']
    out.append(f"Rendered from `logs/official_runs.jsonl` by `scripts/official_runlog_md.py`; "
               f"the JSONL is the deliverable, this is a view of it.")
    out.append('')
    if header:
        rule = header.get('convergence_rule', {})
        out += [
            f"- **Benchmark / protocol:** {esc(header.get('benchmark', ''))} — {esc(header.get('protocol', ''))}",
            f"- **Convergence rule:** ε = {rule.get('epsilon')}, N = {rule.get('N')}, min iterations {rule.get('min_iterations')}",
            f"- **Hard caps:** {esc(json.dumps(header.get('hard_caps', {})))}",
            f"- **Hardware:** {esc(header.get('hardware', ''))}",
            f"- **Agent:** {esc(header.get('agent', ''))}",
            f"- **Human interventions:** {header.get('human_interventions')} — {esc(header.get('human_interventions_note', ''))}",
            '',
        ]
    out += ['| iter | hypothesis | what changed | valid primary | Δ vs best so far | runs | wall (s) | outcome |',
            '|---:|---|---|---:|---:|---:|---:|---|']
    best = None
    n_iter = 0
    for r in rows:
        if r.get('type') != 'iteration':
            continue
        n_iter += 1
        p = primary_of(r)
        if p is None:
            delta = ''
            p_txt = '—'
        else:
            delta = '' if best is None else f'{p - best:+.4f}'
            p_txt = f'{p:.4f}'
            best = p if best is None or p > best else best
        secs = wall_seconds(r)
        out.append(
            f"| {r.get('iter')} | {short(r.get('hypothesis', ''), 80)} | {short(change_text(r), 160)} "
            f"| {p_txt} | {delta} | {len(r.get('sub_runs') or [])} | {'' if secs is None else f'{secs:.0f}'} "
            f"| {outcome(r)} |")
    out.append('')
    out.append(f'{n_iter} scored iterations. `Δ vs best so far` is against the best validation '
               f'primary of all earlier iterations; a negative value is a hypothesis that was '
               f'measured and not adopted.')
    out.append('')
    out.append('## Events between iterations')
    out.append('')
    for r in rows:
        t = r.get('type')
        if t in ('header', 'footer', 'iteration'):
            continue
        what = r.get('what') or r.get('reason') or ''
        detail = r.get('finding') or r.get('why') or r.get('note') or ''
        results = r.get('results')
        line = f"- **{t}** — {esc(what)}"
        if detail:
            line += f" {esc(detail)}"
        if results:
            line += f" `{esc(json.dumps(results))}`"
        if r.get('caps'):
            line += f" caps: `{esc(json.dumps(r['caps']))}`"
        out.append(line)
    out.append('')
    if stop:
        out.append(f"**Stop:** {esc(stop.get('reason', ''))}")
        out.append('')
    if footer:
        ru = footer.get('resource_use', {})
        out += ['## Resource use (footer)', '',
                f"- active wall-clock hours: {ru.get('active_wall_clock_hours')}",
                f"- GPU hours: {ru.get('gpu_hours')}",
                f"- LLM tokens: {esc(ru.get('llm_tokens_estimate', ''))}",
                f"- training compute: {esc(ru.get('training_compute', ''))}",
                f"- human interventions, total: {footer.get('human_interventions_total')} — {esc(footer.get('human_interventions_detail', ''))}",
                '']
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stdout', action='store_true')
    a = ap.parse_args()
    text = render(load())
    if a.stdout:
        sys.stdout.write(text)
        return 0
    with open(MD, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    print(f'wrote {MD}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
