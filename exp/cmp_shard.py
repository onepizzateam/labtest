"""Head-to-head comparison of whole bots on shared seeds (one shard of it).

LAB_CONFIGS = JSON list of [bot, params] with bot "ours" (bin/mechbot; params = MECH_SQ dict) or "wai"
(bin/waibot; params = MECH_FLAGS dict). LAB_VARIANT = "jpl=0.5" (practice) or "ffa=A".
"""
import argparse, json, os, time
from collections import defaultdict

import harness

BOTS = {"ours": os.environ["LAB_BOT"], "wai": os.environ["LAB_WAIBOT"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=20)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    cfgs = json.loads(os.environ["LAB_CONFIGS"])
    variant = os.environ.get("LAB_VARIANT", "ffa=A")
    lo, hi = map(int, a.seeds.split("-"))
    pairs = [(i, s) for s in range(lo, hi + 1) for i in range(len(cfgs))]
    mine = defaultdict(list)
    for k, (i, s) in enumerate(pairs):
        if k % a.nshards == a.shard:
            mine[i].append(s)
    t0 = time.time()
    for i, ss in sorted(mine.items()):
        bot, params = cfgs[i]
        os.environ["LAB_FLAGMODE"] = "1" if bot == "wai" else "0"
        for c in range(0, len(ss), 8):
            if bot == "wai":
                got = harness.play(BOTS[bot], ss[c:c + 8], params, None, variant, jobs=a.jobs)
            else:
                got = harness.play(BOTS[bot], ss[c:c + 8], None, params, variant, jobs=a.jobs)
            print(f"cfg {i} ({bot}): +{len(got)} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("SHARD-DONE", flush=True)


if __name__ == "__main__":
    main()
