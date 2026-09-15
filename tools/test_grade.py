#!/usr/bin/env python3
"""Regression tests for grade.py.

    python3 tools/test_grade.py

  1. the foundation produces a valid tour and scores between 1.0 and 2.0
  2. an invalid tour (repeated city, missing city, junk) scores FAIL_RATIO
     and still counts towards the geometric mean
  3. the time limit, the 4 GB cap and the single-thread rule are enforced
  4. one bad instance does not abort the run
  5. the JSON carries the source digest and the per-instance bookkeeping
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRADE = ROOT / "grade.py"
FOUNDATION = ROOT / "foundation"
INST = ("RANDOM", ROOT / "instances/dev_uni200.tsp", 5)
INST2 = ("STRUCTURED", ROOT / "instances/dev_lat4k.tsp", 5)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def run_grade(tmp, solver, extra=(), instances=None):
    ilist = Path(tmp) / "i.txt"
    rows = instances or [INST]
    ilist.write_text("".join(f"{c} {p} {t}\n" for c, p, t in rows))
    out = Path(tmp) / "r.json"
    out.unlink(missing_ok=True)
    r = subprocess.run(
        [sys.executable, str(GRADE), "--solver", str(solver),
         "--instances", str(ilist), "--json", str(out), *extra],
        cwd=ROOT, capture_output=True, text=True)
    if not out.exists():
        print(r.stdout, r.stderr)
        raise SystemExit(f"grade.py produced no result for {solver}")
    return json.loads(out.read_text()), r


def stub(tmp, name, body):
    p = Path(tmp) / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(0o755)
    return p


def main():
    if not FOUNDATION.exists():
        raise SystemExit("build the foundation first: make foundation")
    meta = json.loads((ROOT / "instances/dev_uni200.meta.json").read_text())

    with tempfile.TemporaryDirectory() as tmp:
        print("\n[1] foundation as the solver")
        res, _ = run_grade(tmp, FOUNDATION)
        inst = res["instances"][0]
        check("valid tour", inst["note"] == "ok", f"note={inst['note']}")
        check("ratio in (1, 2]", 1.0 < inst["ratio"] <= 2.0, f"ratio={inst['ratio']:.4f}")
        check("length matches the pipeline's record",
              inst["tour_len"] == meta["foundation_len"],
              f"{inst['tour_len']} vs {meta['foundation_len']}")
        check("ratio = tour_len / lower_bound",
              abs(inst["ratio"] - inst["tour_len"] / inst["lower_bound"]) < 1e-12)
        check("overall geomean equals the single ratio",
              abs(res["overall_geomean"] - inst["ratio"]) < 1e-12)

        print("\n[1b] the source digest binds solver.cpp to the result")
        src = Path(tmp) / "mine.cpp"
        src.write_text("int main(){return 0;}\n")
        res, _ = run_grade(tmp, FOUNDATION, extra=["--source", str(src)])
        check("records the source digest",
              res["solver_source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest())
        res, _ = run_grade(tmp, FOUNDATION, extra=["--source", str(Path(tmp) / "absent.cpp")])
        check("absent source gives null, not a crash", res["solver_source_sha256"] is None)

        print("\n[2] invalid tours")
        dup = stub(tmp, "dup.sh",
                   f'"{FOUNDATION}" "$1" "$2" "$3"\n'
                   'python3 - "$2" <<\'PY\'\n'
                   'import sys\n'
                   'p=sys.argv[1]; L=open(p).read().split()\n'
                   'L[1]=L[0]\n'
                   'open(p,"w").write("\\n".join(L)+"\\n")\n'
                   'PY\n')
        res, _ = run_grade(tmp, dup)
        inst = res["instances"][0]
        check("repeated city is INVALID", inst["note"] == "INVALID", f"note={inst['note']} {inst.get('invalid_reason')}")
        check("scores FAIL_RATIO", inst["ratio"] == res["fail_ratio"] == 2.0)
        check("counts towards the geomean", abs(res["overall_geomean"] - 2.0) < 1e-12)

        short = stub(tmp, "short.sh", f'"{FOUNDATION}" "$1" "$2" "$3"\nsed -i.bak \'$d\' "$2"\n')
        res, _ = run_grade(tmp, short)
        check("missing city is INVALID", res["instances"][0]["note"] == "INVALID")

        junk = stub(tmp, "junk.sh", 'echo hello > "$2"\n')
        res, _ = run_grade(tmp, junk)
        check("junk output is INVALID", res["instances"][0]["note"] == "INVALID")

        none = stub(tmp, "none.sh", 'exit 0\n')
        res, _ = run_grade(tmp, none)
        check("no output file is NOOUTPUT", res["instances"][0]["note"] == "NOOUTPUT")

        print("\n[3] limits")
        res, _ = run_grade(tmp, stub(tmp, "slow.sh", f'"{FOUNDATION}" "$1" "$2" "$3"\nsleep 30\n'),
                           instances=[("RANDOM", INST[1], 2)])
        check("time limit is enforced (valid tour written too late)",
              res["instances"][0]["note"] == "TIMEOUT", f"note={res['instances'][0]['note']}")
        check("a timed-out instance scores FAIL_RATIO", res["instances"][0]["ratio"] == 2.0)

        hog = stub(tmp, "hog.sh",
                   'exec python3 -c "import os; n=5*1024*1024*1024; b=bytearray(n); '
                   'm=memoryview(b); c=os.urandom(1<<20)\n'
                   'for i in range(0, n, 1<<20): m[i:i+(1<<20)]=c\n'
                   'print(n)"\n')
        res, _ = run_grade(tmp, hog, instances=[("RANDOM", INST[1], 60)])
        check("4 GB cap is enforced", res["instances"][0]["note"] in ("MEMORY", "CRASH"),
              f"note={res['instances'][0]['note']}")

        threads = stub(tmp, "threads.sh",
                       'python3 -c "import time; t=time.time()\nwhile time.time()-t<2: pass" &\n'
                       'python3 -c "import time; t=time.time()\nwhile time.time()-t<2: pass" &\n'
                       'python3 -c "import time; t=time.time()\nwhile time.time()-t<2: pass" &\n'
                       f'"{FOUNDATION}" "$1" "$2" "$3"\nwait\n')
        res, _ = run_grade(tmp, threads)
        check("parallel work is detected", res["instances"][0]["note"] == "THREADS",
              f"note={res['instances'][0]['note']}")

        print("\n[4] one bad instance does not abort the run")
        res, _ = run_grade(tmp, stub(tmp, "crash.sh", "exit 3\n"), instances=[INST, INST2])
        check("both instances reported", len(res["instances"]) == 2)
        check("both CRASH", all(i["note"] == "CRASH" for i in res["instances"]))
        check("category means are separate",
              res["random_geomean"] == 2.0 and res["structured_geomean"] == 2.0)

        print("\n[5] two categories aggregate correctly")
        res, _ = run_grade(tmp, FOUNDATION, instances=[INST, INST2])
        r1, r2 = res["instances"][0]["ratio"], res["instances"][1]["ratio"]
        check("random = first, structured = second",
              abs(res["random_geomean"] - r1) < 1e-12 and abs(res["structured_geomean"] - r2) < 1e-12)
        check("overall is the geometric mean",
              abs(res["overall_geomean"] - (r1 * r2) ** 0.5) < 1e-12)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
