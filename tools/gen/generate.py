#!/usr/bin/env python3
"""Instance generator for the TSP Approximation Competition.

    python3 -m tools.gen --manifest tools/manifest/public.json --out instances
    python3 -m tools.gen --manifest tools/manifest/public.json \
            --instance uni10k --seed 12345 --salt mine --out mydata

Every family is a point process in the integer square [0, RANGE]^2. Points
are de-duplicated (two cities at the same coordinates would have distance 0,
which is legal but pointless). Coordinates are written as plain text:

    N
    x y
    ...

A `<name>.meta.json` is written beside each `.tsp` with the parameters. The
lower bound and the reference tour are added later by tools/pipeline.py.

Families
--------
uniform     independent uniform points.  The classic random Euclidean TSP.
clustered   K Gaussian clusters with very different spreads and sizes, plus a
            thin uniform background.  Density varies by three orders of
            magnitude, so anything that buckets the plane uniformly suffers.
lattice     a jittered square lattice.  Nearly every city has four
            equidistant neighbours, so local search sees huge plateaus, and
            the optimum is almost known: about N * spacing.
roads       cities along random polylines ("streets"), each polyline a
            random walk with a persistent heading, plus junction clusters.
            Long, thin structures: the tour must run along a street and come
            back, and a wrong turn costs the whole street twice.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

RANGE = 1_000_000


def _rng(seed, salt, name):
    h = hashlib.sha256(f"{seed}|{salt}|{name}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _dedupe_fill(pts, n, rng, fresh):
    """Keep the first occurrence of each coordinate; top up with `fresh()`."""
    seen = set()
    out = []
    for p in pts:
        if p not in seen and 0 <= p[0] <= RANGE and 0 <= p[1] <= RANGE:
            seen.add(p)
            out.append(p)
    while len(out) < n:
        p = fresh()
        if p not in seen and 0 <= p[0] <= RANGE and 0 <= p[1] <= RANGE:
            seen.add(p)
            out.append(p)
    rng.shuffle(out)
    return out[:n]


def gen_uniform(n, rng):
    def fresh():
        return (rng.randrange(RANGE + 1), rng.randrange(RANGE + 1))
    return _dedupe_fill([], n, rng, fresh)


def gen_clustered(n, rng, clusters=None):
    k = clusters or max(8, int(round(math.sqrt(n) / 4)))
    # Cluster sizes: heavy-tailed, so a few clusters hold most of the cities.
    weights = [rng.paretovariate(1.2) for _ in range(k)]
    tot = sum(weights)
    background = int(0.08 * n)
    sizes = [int((n - background) * w / tot) for w in weights]
    centres = [(rng.uniform(0.05, 0.95) * RANGE, rng.uniform(0.05, 0.95) * RANGE)
               for _ in range(k)]
    # Spreads span three orders of magnitude: tight knots to broad clouds.
    spreads = [RANGE * 10 ** rng.uniform(-3.0, -0.9) for _ in range(k)]
    pts = []
    for (cx, cy), s, m in zip(centres, spreads, sizes):
        # Anisotropic: squash one axis so clusters are ellipses, not discs.
        ratio = rng.uniform(0.3, 1.0)
        ang = rng.uniform(0, math.pi)
        ca, sa = math.cos(ang), math.sin(ang)
        for _ in range(m):
            gx, gy = rng.gauss(0, s), rng.gauss(0, s * ratio)
            x = cx + gx * ca - gy * sa
            y = cy + gx * sa + gy * ca
            pts.append((int(round(x)), int(round(y))))
    for _ in range(background):
        pts.append((rng.randrange(RANGE + 1), rng.randrange(RANGE + 1)))

    def fresh():
        return (rng.randrange(RANGE + 1), rng.randrange(RANGE + 1))
    return _dedupe_fill(pts, n, rng, fresh)


def gen_lattice(n, rng, jitter=0.08):
    side = int(math.ceil(math.sqrt(n)))
    spacing = RANGE // (side + 1)
    cells = [(i, j) for i in range(side) for j in range(side)]
    rng.shuffle(cells)
    pts = []
    for i, j in cells[:n]:
        x = (i + 1) * spacing + rng.gauss(0, jitter * spacing)
        y = (j + 1) * spacing + rng.gauss(0, jitter * spacing)
        pts.append((int(round(x)), int(round(y))))

    def fresh():
        return (rng.randrange(RANGE + 1), rng.randrange(RANGE + 1))
    return _dedupe_fill(pts, n, rng, fresh)


def gen_roads(n, rng, streets=None):
    k = streets or max(20, n // 400)
    per_street = n // k
    pts = []
    for _ in range(k):
        x, y = rng.uniform(0.05, 0.95) * RANGE, rng.uniform(0.05, 0.95) * RANGE
        heading = rng.uniform(0, 2 * math.pi)
        step = RANGE * rng.uniform(0.0008, 0.003)
        width = step * rng.uniform(0.05, 0.3)
        m = max(10, int(per_street * rng.uniform(0.4, 1.6)))
        for _ in range(m):
            heading += rng.gauss(0, 0.12)
            # Occasional sharp bend.
            if rng.random() < 0.02:
                heading += rng.choice([-1, 1]) * math.pi / 2
            x += step * math.cos(heading)
            y += step * math.sin(heading)
            if not (0.02 * RANGE < x < 0.98 * RANGE):
                heading = math.pi - heading
                x = min(max(x, 0.02 * RANGE), 0.98 * RANGE)
            if not (0.02 * RANGE < y < 0.98 * RANGE):
                heading = -heading
                y = min(max(y, 0.02 * RANGE), 0.98 * RANGE)
            px = x + rng.gauss(0, width)
            py = y + rng.gauss(0, width)
            pts.append((int(round(px)), int(round(py))))
        # A junction: a small dense knot at the end of the street.
        if rng.random() < 0.5:
            for _ in range(max(3, per_street // 15)):
                pts.append((int(round(x + rng.gauss(0, step))),
                            int(round(y + rng.gauss(0, step)))))

    def fresh():
        return (rng.randrange(RANGE + 1), rng.randrange(RANGE + 1))
    return _dedupe_fill(pts, n, rng, fresh)


FAMILIES = {
    "uniform": gen_uniform,
    "clustered": gen_clustered,
    "lattice": gen_lattice,
    "roads": gen_roads,
}


def write_instance(out_dir, name, pts, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsp = out_dir / f"{name}.tsp"
    with open(tsp, "w") as f:
        f.write(f"{len(pts)}\n")
        for x, y in pts:
            f.write(f"{x} {y}\n")
    meta = dict(meta)
    meta.update(name=name, n=len(pts), coord_range=RANGE,
                distance="EUC_2D: floor(sqrt(dx^2+dy^2)+0.5)")
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return tsp


def generate(spec, seed, salt, out_dir):
    name = spec["name"]
    rng = _rng(seed, salt, name)
    fam = FAMILIES[spec["family"]]
    kwargs = {k: v for k, v in spec.get("params", {}).items()}
    pts = fam(spec["n"], rng, **kwargs)
    meta = {
        "family": spec["family"],
        "category": spec["category"],
        "time_limit_s": spec["time_limit_s"],
        "seed": seed,
        "salt": salt,
        "params": spec.get("params", {}),
        "description": spec.get("description", ""),
    }
    return write_instance(out_dir, name, pts, meta)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--instance", help="only this instance from the manifest")
    ap.add_argument("--seed", type=int, help="override the manifest seed")
    ap.add_argument("--salt", default=None, help="override the manifest salt")
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    seed = args.seed if args.seed is not None else manifest["seed"]
    salt = args.salt if args.salt is not None else manifest.get("salt", "")
    for spec in manifest["instances"]:
        if args.instance and spec["name"] != args.instance:
            continue
        p = generate(spec, seed, salt, args.out)
        print(f"wrote {p}  (n={spec['n']}, {spec['family']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
