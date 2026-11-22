"""Screens over the behaviours other teams built: Wai and Jai's flags (github.com/waiprotein/mech-royale-bot, MIT,
a giddygiddyboy fork; vendored in bots/wai), plus our allfoes finding. Their shipped defaults are the control;
every factor's low level is their default (off), the high level the flag's documented/suggested value where the
code gives one, otherwise a mid-range value. Folded Plackett-Burman (48 runs, resolution IV) + control.

Set P (practice-mode flags, squad variant jpl=0.5) and set F (free-for-all flags, field A).
  wai_shard.py --set P|F --shard k --nshards 20 --seeds A-B --jobs J
  waidoe.py analyze P|F <results.jsonl...>
"""
import json, sys

import numpy as np
from scipy import stats

import doe

SETS = {
    "P": {   # practice / squad-mode behaviours
        "allfoes": 1, "squsedrops": 1, "sqhpw": 0.21, "sqjink": 1, "sqbackopen": 1, "sqlrlaser": 150,
        "lcdpunish": 1.5, "coverreload": 1, "rescan": 1, "sqspeed": 1, "regenlock": 1, "sqhunt": 1,
        "rearavoid": 40, "lightpad": 5, "earlyspace": 40,
    },
    "F": {   # free-for-all behaviours (FFA-only or all-mode early-game)
        "facesound": 1, "gunonlyout": 1, "lightlos": 1, "keeplaser": 1, "cubestop": 1, "earlyretreat": 60,
        "cubepush": 60, "ffadrops": 150, "regenhide": 0.5, "watchlast": 5, "laserlos": 30, "chargeengage": 180,
        "lrlaser": 160, "backopen": 2, "precharge": 1.5, "tpjoin": 20, "homelos": 10, "earlyspacekeep": 150,
        "arm": 1, "armfirst": 1, "armboost": 1, "duelflee": 0.35, "scan": 6,
    },
}


def runs(name):
    F = list(SETS[name])
    D = doe.design(len(F))
    assert doe.verify(D) == "ok"
    out = [{f: SETS[name][f] for f, x in zip(F, row) if x > 0} or None for row in D]
    out.append(None)   # control: their shipped defaults
    return F, D, out


def analyze(name, files):
    F, D, R = runs(name)
    rows = [json.loads(l) for f in files for l in open(f)]
    key = lambda p: json.dumps(p or {}, sort_keys=True)
    want = {key(R[i]): i for i in range(len(D))}
    des = [r for r in rows if key(r["p"]) in want]
    for r in des:
        r["run"] = want[key(r["p"])]
    ctrl = [r for r in rows if not r["p"]]
    by = {}
    for r in des:
        by.setdefault(r["seed"], []).append(r)
    full = {s for s, v in by.items() if len({x["run"] for x in v}) == len(D)}
    des = [r for r in des if r["seed"] in full]
    print(f"[{name}] design matches {len(des)} on {len(full)} complete seeds; late {sum(r['late'] for r in rows)}; crashes {sum(r['tb'] for r in rows)}")
    sm = {}
    for r in des:
        sm.setdefault(r["seed"], []).append(r["score"])
    sm = {s: np.mean(v) for s, v in sm.items()}
    y = np.array([r["score"] - sm[r["seed"]] for r in des])
    X = np.column_stack([np.ones(len(des)), np.array([D[r["run"]] for r in des], float)])
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X); V = XtXi @ (X.T * (e / (1 - h)) ** 2) @ X @ XtXi
    eff, se = 2 * b[1:], 2 * np.sqrt(np.diag(V))[1:]
    p = 2 * stats.norm.sf(np.abs(eff / se)); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    print(f"design mean {np.mean([r['score'] for r in des]):.3f}; control (their shipped) mean {np.mean([r['score'] for r in ctrl]) if ctrl else float('nan'):.3f} (n={len(ctrl)})")
    for i in np.argsort(-eff):
        tag = ("ADOPT" if eff[i] > 0 else "REJECT") if q[i] < 0.05 else ""
        print(f"  {F[i]:15s} on={SETS[name][F[i]]!s:6s} {eff[i]:+7.3f} ± {1.96 * se[i]:.3f}  q {q[i]:.4f}  {tag}")
    best = {F[i]: SETS[name][F[i]] for i in range(n) if q[i] < 0.05 and eff[i] > 0}
    print("significant positive flags:", best)


if __name__ == "__main__":
    analyze(sys.argv[2], sys.argv[3:])
