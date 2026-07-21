#!/usr/bin/env python3
"""Join the sym_all symbolic sweep against the recorded sTiles symbolic baseline.

Baseline = symbolic.csv:sTiles  (auto-mode @ 40 threads, the paper headline c40).
Sweep    = symbolic_speedup_all_<jobid>.csv produced by symbolic_speedup_all.slurm
           (columns: matrix,n,nnz,symbolic_parallel_s,symbolic_serial_s,speedup,winner_match)

Usage:
    python3 compare_symbolic_sweep.py <sweep.csv> [symbolic.csv]
"""
import csv, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sweep_path = sys.argv[1]
base_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "symbolic.csv")

# recorded baseline: matrix -> sTiles c40
base = {}
with open(base_path) as f:
    for row in csv.DictReader(f):
        try:
            base[row["matrix"]] = float(row["sTiles"])
        except (ValueError, KeyError):
            pass

rows, sum_par, sum_ser, sum_base = [], 0.0, 0.0, 0.0
with open(sweep_path) as f:
    for r in csv.DictReader(f):
        m = r["matrix"]
        try:
            par = float(r["symbolic_parallel_s"]); ser = float(r["symbolic_serial_s"])
        except ValueError:
            continue
        rec = base.get(m)
        layer = ser / par if par > 0 else float("nan")          # parallel-layer speedup
        vs_rec = rec / par if (rec and par > 0) else float("nan")  # parallel vs recorded c40
        rows.append((m, rec, par, ser, layer, vs_rec, r.get("winner_match", "")))
        sum_par += par; sum_ser += ser
        if rec:
            sum_base += rec

w = max((len(r[0]) for r in rows), default=6)
print(f"{'matrix':<{w}}  {'recorded_c40':>12}  {'sweep_par':>10}  {'sweep_ser':>10}  "
      f"{'layer_spd':>9}  {'par/rec':>8}  win")
for m, rec, par, ser, layer, vs_rec, win in sorted(rows, key=lambda x: -(x[2] or 0)):
    rec_s = f"{rec:.6f}" if rec else "--"
    print(f"{m:<{w}}  {rec_s:>12}  {par:>10.6f}  {ser:>10.6f}  "
          f"{layer:>9.2f}  {vs_rec:>8.2f}  {win}")

print("\n" + "=" * 60)
print(f"TOTAL over {len(rows)} matrices:")
print(f"  recorded c40 baseline : {sum_base:10.1f} s")
print(f"  sweep parallel        : {sum_par:10.1f} s")
print(f"  sweep serial          : {sum_ser:10.1f} s")
if sum_par > 0:
    print(f"  parallel-layer speedup (serial/parallel): {sum_ser/sum_par:.2f}x")
if sum_base > 0 and sum_par > 0:
    print(f"  parallel vs recorded c40 (recorded/parallel): {sum_base/sum_par:.2f}x")
