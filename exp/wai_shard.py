"""One shard of a waidoe screen (set P = practice, set F = free-for-all on field A)."""
import argparse, json, os, time
from collections import defaultdict

os.environ["LAB_FLAGMODE"] = "1"
import harness
import waidoe

GO = os.environ["LAB_WAIBOT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=20)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--deadline-min", type=float, default=320)
    a = ap.parse_args()
    variant = "jpl=0.5" if a.set == "P" else "ffa=" + os.environ.get("LAB_FIELD_NAME", "A")
    lo, hi = map(int, a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    _, _, R = waidoe.runs(a.set)
    if os.environ.get("LAB_CONFIGS"):
        R = json.loads(os.environ["LAB_CONFIGS"])
    pairs = [(i, s) for s in seeds for i in range(len(R))]
    mine = defaultdict(list)
    for k, (i, s) in enumerate(pairs):
        if k % a.nshards == a.shard:
            mine[i].append(s)
    t0 = time.time()
    print(f"set {a.set} shard {a.shard}: {sum(len(v) for v in mine.values())} matches", flush=True)
    for i, ss in sorted(mine.items()):
        for c in range(0, len(ss), 8):
            if (time.time() - t0) / 60 > a.deadline_min:
                print("deadline reached", flush=True)
                return
            got = harness.play(GO, ss[c:c + 8], R[i], None, variant, jobs=a.jobs)
            print(f"run {i:2d}: +{len(got)} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("SHARD-DONE", flush=True)


if __name__ == "__main__":
    main()
