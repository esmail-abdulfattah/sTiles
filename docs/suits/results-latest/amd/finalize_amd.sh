#!/bin/bash
# Copy the AMD (EPYC) result CSVs from ibex into this paper folder, stripping bcsstk12
# (duplicate of bcsstk11) so the AMD set matches the paper's 88 matrices.
#
# Competitors (PARDISO/MUMPS/CHOLMOD/PaStiX/symPACK) + analyze are final.
# For sТiles, prefer the BOUND sweep (OMP_PLACES=cores fix, the honest config) once it
# has completed; until then fall back to the unbound sweep as a labelled placeholder.
#
# Re-run this after the bound sweep finishes to swap in the final sТiles numbers:
#   bash finalize_amd.sh
set -u
SRC=/home/abdulfe/ibex/bench/results-latest
DST=/home/abdulfe/Dropbox/sTiles/Semisparse___sTiles/results-latest/amd
strip12() { awk -F, 'NR==1 || $1!="bcsstk12"' "$1" > "$2"; }   # drop the bcsstk12 dup row

# competitors + analyze (final)
for s in pardiso mumps cholmod pastix sympack; do
  [ -f "$SRC/${s}_amd.csv" ] && strip12 "$SRC/${s}_amd.csv" "$DST/${s}_amd.csv"
done
[ -f "$SRC/analyze_amd.csv" ] && strip12 "$SRC/analyze_amd.csv" "$DST/analyze_amd.csv"

# sТiles: bound sweep if complete (>=89 rows incl bcsstk12), else unbound placeholder
B=$SRC/stiles_auto_amd_bound.csv
if [ -f "$B" ] && [ "$(($(wc -l <"$B")-1))" -ge 89 ]; then
  strip12 "$B" "$DST/stiles_auto_amd.csv"; NOTE="BOUND (OMP_PLACES=cores, final)"
else
  strip12 "$SRC/stiles_auto_amd.csv" "$DST/stiles_auto_amd.csv"
  bn=$([ -f "$B" ] && echo "$(($(wc -l <"$B")-1))/89" || echo "not started")
  NOTE="UNBOUND placeholder -- rerun me when bound sweep done (now: $bn)"
fi

echo "=== Dropbox amd/ updated ==="
for f in "$DST"/*_amd.csv "$DST"/analyze_amd.csv; do
  [ -f "$f" ] && printf "  %-22s %3d rows   bcsstk12:%s\n" "$(basename "$f")" "$(($(wc -l <"$f")-1))" "$(grep -c '^bcsstk12,' "$f")"
done
echo "  sТiles source: $NOTE"
