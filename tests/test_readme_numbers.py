"""Every number the README's first screen and Result section quotes, derived
from the committed artifact it comes from and looked up in the README text.

Change a digit in the README, or a value in the artifact, and one of these
fails. The artifacts themselves are pinned elsewhere: final.json by
`official_final.py --check`, ceiling.json by `official_ceiling.py --check`,
the run log by `official_runlog.py --check`. bootstrap.json cannot be
recomputed offline (it needs the FM reproduction's test predictions, which are
not committed), so here it is tied to final.json and to the README only.
"""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OFF = ROOT / 'results' / 'official'


def _text(rel):
    return (ROOT / rel).read_text(encoding='utf-8')


def _json(rel):
    return json.loads(_text(rel))


# Whitespace (line wrapping) is normalized so a re-flowed paragraph still matches.
README = ' '.join(_text('README.md').split())
FINAL = _json('results/official/final.json')
CEIL = _json('results/official/ceiling.json')
BOOT = _json('results/official/bootstrap.json')
BASE = _json('kuairand-starter-kit/baseline_scores.json')['scores']
RUNLOG = [json.loads(line) for line in _text('logs/official_runs.jsonl').splitlines() if line.strip()]


def quoted(number, text=README):
    """`number` appears as a whole token: no digit or decimal point glued to either side."""
    pat = r'(?<![\d.])' + re.escape(number) + r'(?![\d])(?!\.\d)'
    return re.search(pat, text) is not None


def first_screen():
    """README up to the first '## ' heading: what was built, found, and the limits."""
    return README.split(' ## ', 1)[0]


def _r4(x):
    # Half-up on the decimal value the artifact stores (0.66965 -> 0.6697),
    # which is how the README rounds; float formatting would give 0.6696.
    return Decimal(str(x)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


def f4(x):
    return f'{_r4(x)}'


def signed4(x):
    r = _r4(x)
    return f'+{r}' if r >= 0 else f'{r}'


# ---- Result table -------------------------------------------------------

def test_result_table_rows():
    """Each row of the Result table, cell for cell."""
    def cells(v, t):
        return f'| {f4(v)} | {f4(t)} |'
    rows = [
        '| random (uniform noise, mean of 5 seeds — measured here) '
        + cells(CEIL['valid']['random_primary_mean_of_5_seeds'], CEIL['test']['random_primary_mean_of_5_seeds']),
        '| pop (published) '
        + cells(BASE['item_popularity']['valid']['primary'], BASE['item_popularity']['test']['primary']),
        '| FM starter kit (published) '
        + cells(BASE['fm_official']['valid']['primary'], BASE['fm_official']['test']['primary']),
        f"| **RecAgent final blend** | **{f4(FINAL['metrics']['valid']['primary'])}** "
        f"| **{f4(FINAL['metrics']['test']['primary'])}** |",
    ]
    for row in rows:
        assert row in README, f'Result table row does not match the artifacts: {row}'
    # final.json and the starter kit agree on the published FM baseline
    for split in ('valid', 'test'):
        assert FINAL['baseline']['fm_published'][split] == BASE['fm_official'][split]['primary']


def test_first_screen_numbers():
    fs = first_screen()
    t = FINAL['metrics']['test']
    d = BOOT['delta_final_minus_fm_repro']['primary']
    assert (f"scores **{f4(t['primary'])}** test primary against the organizers' FM "
            f"baseline at {f4(BASE['fm_official']['test']['primary'])} "
            f"({signed4(FINAL['delta_test_primary_vs_published_fm'])}; paired bootstrap "
            f"95% CI [{signed4(d['ci95'][0])}, {signed4(d['ci95'][1])}]") in fs


def test_headline_delta_and_components():
    t = FINAL['metrics']['test']
    assert (f"**{signed4(FINAL['delta_test_primary_vs_published_fm'])} test primary over the FM "
            f"baseline** (test GAUC {f4(t['GAUC'])}, nDCG@5 {f4(t['nDCG@5'])})") in README
    # the delta really is final - published FM
    assert abs(t['primary'] - BASE['fm_official']['test']['primary']
               - FINAL['delta_test_primary_vs_published_fm']) < 5e-7


def test_attainable_range_numbers():
    t = CEIL['test']
    assert (f"a label-perfect ranking scores {f4(t['label_perfect']['primary'])} on this benchmark "
            f"({round(t['share_no_positive'] * 100)}% of users have no positive label)") in README
    assert f"on the {f4(t['random_primary_mean_of_5_seeds'])} → {f4(t['label_perfect']['primary'])} attainable range" in README
    pos = t['position_in_range']
    fm_key = f"fm_published_{f4(BASE['fm_official']['test']['primary'])}"
    final_key = f"final_{f4(FINAL['metrics']['test']['primary'])}"
    assert set(pos) == {fm_key, final_key}
    assert (f"the baseline sits at **{pos[fm_key] * 100:.1f}%** and this submission at "
            f"**{pos[final_key] * 100:.1f}%**") in README


def test_interim_reveal_number():
    """The 08-31 interim test reveal the Result section names (ERRATA 3)."""
    interim = [e for e in RUNLOG if e['type'] == 'test_eval'
               and 'Interim' in e['what']]
    assert len(interim) == 1
    assert sum(1 for e in RUNLOG if e['type'] == 'test_eval') == 2   # "revealed twice"
    blend = interim[0]['results']['blend_test_primary']
    assert f'the {f4(blend)} interim blend' in README


def test_label_row_counts():
    for split in ('valid', 'test'):
        assert f"{CEIL[split]['rows']:,}" in README


def test_bootstrap_numbers():
    d = BOOT['delta_final_minus_fm_repro']['primary']
    pe = BOOT['point_estimate']
    # bootstrap.json scored the same submission final.json holds
    assert pe['final_primary'] == FINAL['metrics']['test']['primary']
    assert pe['fm_published_primary'] == BASE['fm_official']['test']['primary']
    assert (f"final − FM at **{signed4(d['mean'])}** primary against our FM reproduction "
            f"({f4(pe['fm_repro_primary'])})") in README
    # every confidence interval the README prints is this one
    brackets = re.findall(r'\[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]', README)
    assert brackets, 'no confidence interval found in the README'
    assert set(brackets) == {(signed4(d['ci95'][0]), signed4(d['ci95'][1]))}, brackets
    reps, users = re.search(r'(\d+) resamples of (\d+) test users', BOOT['method']).groups()
    assert f'{int(reps):,} resamples of the {int(users):,} test users' in README
    assert int(users) == CEIL['test']['users']
    assert d['p_delta_le_0'] == 0.0 and f'0 of {int(reps):,} resamples' in README


# ---- budget and run counts ----------------------------------------------

def test_scored_iterations_and_hours():
    iters = sum(1 for e in RUNLOG if e['type'] == 'iteration')
    assert f'| scored iterations | **{iters}** |' in README
    assert f'{iters} scored iterations' in first_screen()
    hours = RUNLOG[-1]['resource_use']['active_wall_clock_hours']
    assert f'| active wall-clock | **~{hours:g} h** |' in README


def test_training_run_count():
    runs = [p.stem for p in OFF.glob('*.json')
            if 'best_iter' in json.loads(p.read_text(encoding='utf-8'))]
    in_run = [r for r in runs if r != 'p9_no_vstat']   # post-deadline diagnostic
    assert len(in_run) == 56
    assert f'{len(in_run)} training runs' in first_screen()
    assert f'{len(in_run)} individual training runs' in README
    assert '40+' not in README


# ---- the vstat limitation -----------------------------------------------

def test_vstat_limitation_numbers():
    with_v = _json('results/official/p4_ease.json')['valid']['primary']
    without = _json('results/official/p9_no_vstat.json')['valid']['primary']
    fm = BASE['fm_official']['valid']['primary']
    assert 'vstat' not in _json('results/official/p9_no_vstat.json')['features'].split(',')
    assert 'vstat' in _json('results/official/p4_ease.json')['features'].split(',')
    lim = README.split('## Limitations', 1)[1].split(' ## ', 1)[0]
    assert f'from {f4(with_v)} to **{f4(without)}**' in lim
    assert f'−{with_v - without:.4f}' in lim
    assert f'+{with_v - fm:.4f}' in lim and quoted(f4(fm), lim)
    assert without > fm, 'the Limitations bullet says the model still clears FM'
    assert f'{with_v - without:.4f}' in first_screen()


# ---- wording guards -----------------------------------------------------

SCORED_ONCE_FILES = ['README.md', 'results/official/leaderboard.md',
                     'docs/devpost_description.md', 'docs/devpost_form_fields.md']


@pytest.mark.parametrize('rel', SCORED_ONCE_FILES)
def test_scored_once_always_carries_the_two_reveals(rel):
    """ERRATA item 3: the designated checkpoint was scored once, the split
    was revealed twice. Every paragraph that says 'once' about test scoring
    must also say the second half."""
    paragraphs = [' '.join(p.split()) for p in re.split(r'\n\s*\n', _text(rel))]
    bad = [p for p in paragraphs
           if re.search(r'scored\s+(\*\*)?once|test\s+(set\s+|split\s+)?exactly\s+once|'
                        r'scored on the hidden test\s+(exactly\s+)?once', p)
           and not re.search(r'twice|interim|before that', p)]
    assert not bad, f'{rel}: "scored once" without the two-reveal qualifier: {bad[0][:200]!r}'


def test_first_screen_states_limits():
    fs = first_screen()
    assert 'twice' in fs and 'ERRATA' in fs
    assert 'self-reported' in fs
    assert 'vstat' in fs
    assert 'without a human writing or editing' not in README
