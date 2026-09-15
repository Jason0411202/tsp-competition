#!/usr/bin/env python3
"""Instructor pipeline: build instances with proven bounds and reference tours.

    python3 tools/pipeline.py build --manifest tools/manifest/public.json \
            --out instances --ref-seconds 600 [--only uni10k] [--jobs 3]
    python3 tools/pipeline.py check [--dir instances]

`build`, per instance:
  1. tools.gen                    -> <name>.tsp, partial <name>.meta.json
  2. tools/ref/solvers/refsolve   -> <name>.tour (kept in tools/ref/tours/,
                                     gitignored) and its length, ref_len
  3. tools/bound/bound --ub ref   -> Held-Karp bound; lower_bound = ceil
  4. ./foundation                 -> foundation_len, the baseline row on the
                                     scoreboard
  5. sanity: lower_bound <= foundation_len, ref_len <= foundation_len,
     lower_bound <= ref_len, the reference tour re-verifies with grade.py's
     own checker.  Any violation FAILS THE BUILD.
  6. merge into <name>.meta.json

`check` re-verifies every committed instance: the reference tour still has
the stored length, the stored bound is not above it, and the bound tool
reproduces the stored bound to within 0.05% (a much larger difference means
the instance file changed).
"""

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from grade import read_points, verify_tour  # noqa: E402

BOUND = HERE / "bound" / "bound"
REFSOLVE = HERE / "ref" / "solvers" / "refsolve"
FOUNDATION = ROOT / "foundation"
TOURS = HERE / "ref" / "tours"


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"command failed (exit {r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


def build_one(spec, manifest, out_dir, ref_seconds, bound_iters, reuse_ref=False):
    name = spec["name"]
    tsp = out_dir / f"{name}.tsp"
    meta_path = out_dir / f"{name}.meta.json"
    log = lambda *a: print(f"[{name}]", *a, flush=True)

    # 1. generate
    run([sys.executable, "-m", "tools.gen", "--manifest", manifest,
         "--instance", name, "--out", out_dir], cwd=ROOT)
    meta = json.loads(meta_path.read_text())
    xs, ys = read_points(tsp)

    # 2. reference tour
    TOURS.mkdir(parents=True, exist_ok=True)
    ref_tour = TOURS / f"{name}.tour"
    secs = ref_seconds if not name.startswith("dev_") else max(10, ref_seconds // 10)
    if reuse_ref and ref_tour.exists():
        log("reusing existing reference tour")
        ref_len, why = verify_tour(ref_tour, xs, ys)
        ref_len_reported = ref_len
    else:
        log(f"refsolve {secs}s ...")
        r = run([REFSOLVE, tsp, ref_tour, secs, 1])
        ref_len_reported = int(r.stdout.strip().split()[-1])
        ref_len, why = verify_tour(ref_tour, xs, ys)
    if ref_len is None:
        raise SystemExit(f"{name}: reference tour invalid: {why}")
    if ref_len != ref_len_reported:
        raise SystemExit(f"{name}: refsolve says {ref_len_reported}, checker says {ref_len}")
    log(f"ref_len {ref_len}")

    # 3. lower bound
    log("bound ...")
    r = run([BOUND, tsp, "--ub", ref_len, "--iters", bound_iters, "--json"])
    b = json.loads(r.stdout.strip().splitlines()[-1])
    lb = int(b["lower_bound"])
    log(f"lower_bound {lb}  (HK {b['held_karp']:.1f}, ref/LB = {ref_len / lb:.4f})")

    # 4. foundation
    f_tour = TOURS / f"{name}.foundation.tour"
    run([FOUNDATION, tsp, f_tour, spec["time_limit_s"]])
    f_len, why = verify_tour(f_tour, xs, ys)
    if f_len is None:
        raise SystemExit(f"{name}: foundation tour invalid: {why}")
    log(f"foundation_len {f_len}  (ratio {f_len / lb:.4f})")

    # 5. sanity
    if not (lb <= ref_len <= f_len):
        raise SystemExit(f"{name}: bound/ref/foundation not ordered: {lb} {ref_len} {f_len}")
    if ref_len / lb > 1.25:
        raise SystemExit(f"{name}: reference is {ref_len / lb:.3f}x the bound -- "
                         "either the bound is loose or refsolve is broken")

    # 6. merge
    meta.update(
        lower_bound=lb,
        held_karp=b["held_karp"],
        one_tree_plain=b["one_tree_plain"],
        mst=b["mst"],
        bound_iters=b["iters"],
        bound_k=b["k"],
        ref_len=ref_len,
        ref_seconds=secs,
        ref_ratio=round(ref_len / lb, 5),
        foundation_len=f_len,
        foundation_ratio=round(f_len / lb, 5),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log("meta written")
    return name, meta


def cmd_build(args):
    manifest = json.loads(args.manifest.read_text())
    specs = [s for s in manifest["instances"] if not args.only or s["name"] in args.only]
    for exe in (BOUND, REFSOLVE, FOUNDATION):
        if not exe.exists():
            raise SystemExit(f"missing {exe}: run  make foundation tools tools/ref/solvers/refsolve")
    # The bound tool is multi-threaded and refsolve is single-threaded, so a
    # couple of instances in flight keeps the machine busy without thrashing.
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(build_one, s, args.manifest, args.out, args.ref_seconds, args.bound_iters, args.reuse_ref)
                for s in specs]
        results = [f.result() for f in futs]
    print()
    print(f"{'instance':<12}{'n':>8}{'lower_bound':>13}{'ref_len':>12}{'ref/LB':>9}"
          f"{'foundation':>12}{'found/LB':>10}")
    for name, m in results:
        print(f"{name:<12}{m['n']:>8}{m['lower_bound']:>13}{m['ref_len']:>12}"
              f"{m['ref_ratio']:>9.4f}{m['foundation_len']:>12}{m['foundation_ratio']:>10.4f}")


def cmd_check(args):
    bad = 0
    for meta_path in sorted(args.dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        name = meta["name"]
        tsp = args.dir / f"{name}.tsp"
        xs, ys = read_points(tsp)
        ok = True
        if len(xs) != meta["n"]:
            print(f"{name}: n mismatch"); ok = False
        ref_tour = TOURS / f"{name}.tour"
        if ref_tour.exists():
            L, why = verify_tour(ref_tour, xs, ys)
            if L != meta["ref_len"]:
                print(f"{name}: reference tour length {L} != stored {meta['ref_len']} ({why})"); ok = False
        if meta["lower_bound"] > meta["ref_len"]:
            print(f"{name}: lower_bound above ref_len"); ok = False
        if not args.fast:
            r = run([BOUND, tsp, "--ub", meta["ref_len"], "--iters", meta.get("bound_iters", 400), "--json"])
            b = json.loads(r.stdout.strip().splitlines()[-1])
            rel = abs(b["held_karp"] - meta["held_karp"]) / meta["held_karp"]
            if rel > 5e-4:
                print(f"{name}: recomputed bound {b['held_karp']:.1f} vs stored {meta['held_karp']:.1f}"); ok = False
        print(f"{'ok   ' if ok else 'BAD  '}{name}")
        bad += not ok
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--manifest", type=Path, default=HERE / "manifest" / "public.json")
    b.add_argument("--out", type=Path, default=ROOT / "instances")
    b.add_argument("--only", nargs="*")
    b.add_argument("--ref-seconds", type=int, default=600)
    b.add_argument("--bound-iters", type=int, default=6000)
    b.add_argument("--reuse-ref", action="store_true", help="keep an existing reference tour instead of re-running refsolve")
    b.add_argument("--jobs", type=int, default=2)
    c = sub.add_parser("check")
    c.add_argument("--dir", type=Path, default=ROOT / "instances")
    c.add_argument("--fast", action="store_true", help="skip recomputing the bounds")
    args = ap.parse_args()
    return cmd_build(args) if args.cmd == "build" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
