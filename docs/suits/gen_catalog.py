#!/usr/bin/env python3
"""
Generate data/catalog.js for the new sTiles matrix suite.

Single source of truth = the MatrixMarket files under MTX_ROOT, organized
into group1..group12 subfolders. We read only the header of each .mtx
(streamed, bounded) to extract n and nnz, then emit one catalog entry per
matrix. Benchmark numbers are intentionally absent on this fresh site and
get added later as runs land.

nnz from a "symmetric" MatrixMarket file is the stored lower triangle
(including the diagonal), so density is taken relative to n*(n+1)/2.
"""
import os, glob, re, sys

MTX_ROOT = "/home/abdulfe/rinladownload/mtx"
OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "catalog.js")
GROUPS   = [f"group{i}" for i in range(1, 13)]

# Matrices present in the .mtx tree but intentionally excluded from the suite.
# The lme4 crossed mixed-effects (MixedModel) class is omitted from the paper's
# suite table, so it is dropped from the catalog too. net549851 and pid6922_Q
# were never benchmarked (catalog stubs) and are not part of the suite.
EXCLUDE = {"lme4_crossed_n42k", "lme4_crossed_n94k", "lme4_crossed_n146k",
           "inla_graph_net549851", "inla_graph_pid6922_Q",
           "bcsstk12"}  # exact duplicate of bcsstk11 (same n,nnz,values)

# Application / domain per matrix. Domains for the standard matrices come from
# the previous catalog; the INLA latent-Gaussian, lme4 mixed-effects, and the
# group10 structural matrices are assigned here. Edit when matrices are added.
APPLICATION = {
    "1138_bus": "Power Network",
    "Emilia_923": "Structural Engineering",
    "Fault_639": "Structural Engineering",
    "G2_circuit": "Circuit Simulation",
    "af_0_k101": "Structural Engineering",
    "af_shell3": "Structural Engineering",
    "apache1": "Structural Engineering",
    "apache2": "Structural Engineering",
    "audikw_1": "Structural Engineering",
    "bcsstk08": "Structural Engineering",
    "bcsstk09": "Structural Engineering",
    "bcsstk10": "Structural Engineering",
    "bcsstk11": "Structural Engineering",
    "bcsstk13": "Structural Engineering",
    "bcsstk14": "Structural Engineering",
    "bcsstk15": "Structural Engineering",
    "bcsstk16": "Structural Engineering",
    "bcsstk17": "Structural Engineering",
    "bcsstk18": "Structural Engineering",
    "bmw3_2": "Structural Engineering",
    "bmw7st_1": "Structural Engineering",
    "bone010": "Model Reduction",
    "boneS01": "Structural Engineering",
    "boneS10": "Structural Engineering",
    "bundle1": "Computer Graphics/Vision",
    "consph": "2D/3D Problem",
    "crankseg_1": "Structural Engineering",
    "crankseg_2": "Structural Engineering",
    "ct20stif": "Structural Engineering",
    "ecology2": "Computational Science",
    "gyro_k": "Model Reduction",
    "gyro_m": "Model Reduction",
    "hood": "Structural Engineering",
    "inla_graph_83o4NNNo": "Bayesian Statistics",
    "inla_graph_8rtKSK": "Bayesian Statistics",
    "inla_graph_animal1": "Bayesian Statistics",
    "inla_graph_animal2": "Bayesian Statistics",
    "inla_graph_ayaLRw": "Bayesian Statistics",
    "inla_graph_bern_spd": "Bayesian Statistics",
    "inla_graph_diff": "Bayesian Statistics",
    "inla_graph_ferris": "Bayesian Statistics",
    "inla_graph_lgm_10010_bw2": "Bayesian Statistics",
    "inla_graph_lgm_100200_bw1": "Bayesian Statistics",
    "inla_graph_lgm_100200_bw2": "Bayesian Statistics",
    "inla_graph_lgm_48600_bw2": "Bayesian Statistics",
    "inla_graph_lgm_50000_bw15000": "Bayesian Statistics",
    "inla_graph_lgm_50400_bw2": "Bayesian Statistics",
    "inla_graph_lidense": "Bayesian Statistics",
    "inla_graph_net1628760": "Bayesian Statistics",
    "inla_graph_net549851": "Bayesian Statistics",
    "inla_graph_net814381": "Bayesian Statistics",
    "inla_graph_pedigree": "Bayesian Statistics",
    "inla_graph_pid6922_Q": "Bayesian Statistics",
    "inla_graph_sem_n100000": "Bayesian Statistics",
    "inla_graph_sem_n2000": "Bayesian Statistics",
    "inla_graph_sem_n20000": "Bayesian Statistics",
    "inla_graph_sem_n5000": "Bayesian Statistics",
    "inla_graph_sh7Pgi": "Bayesian Statistics",
    "inla_graph_spacetime": "Bayesian Statistics",
    "inla_graph_stcov": "Bayesian Statistics",
    "inla_graph_yU0G1u": "Bayesian Statistics",
    "inline_1": "Structural Engineering",
    "ldoor": "Structural Engineering",
    "lme4_crossed_n146k": "Bayesian Statistics",
    "lme4_crossed_n42k": "Bayesian Statistics",
    "lme4_crossed_n94k": "Bayesian Statistics",
    "m_t1": "Structural Engineering",
    "msc10848": "Structural Engineering",
    "msc23052": "Structural Engineering",
    "nasa2910": "Structural Engineering",
    "nasa4704": "Structural Engineering",
    "nasasrb": "Structural Engineering",
    "nd12k": "2D/3D Problem",
    "nd3k": "2D/3D Problem",
    "nd6k": "2D/3D Problem",
    "offshore": "Structural Engineering",
    "oilpan": "Structural Engineering",
    "parabolic_fem": "Computational Fluid Dynamics",
    "pwtk": "Structural Engineering",
    "s3dkq4m2": "Structural Engineering",
    "s3dkt3m2": "Structural Engineering",
    "ship_001": "Structural Engineering",
    "ship_003": "Structural Engineering",
    "shipsec1": "Structural Engineering",
    "shipsec5": "Structural Engineering",
    "shipsec8": "Structural Engineering",
    "smt": "Structural Engineering",
    "thermal1": "Thermal Problem",
    "thermomech_TC": "Thermal Problem",
    "thermomech_TK": "Thermal Problem",
    "thermomech_dM": "Thermal Problem",
    "thread": "Structural Engineering",
    "tmt_sym": "Electromagnetics",
}


def read_header(path):
    """Return (n, nnz, symmetric:bool, pattern:bool) reading only the header."""
    symmetric = pattern = False
    with open(path, "r", errors="replace") as fh:
        first = fh.readline()
        low = first.lower()
        symmetric = "symmetric" in low
        pattern   = "pattern" in low
        # skip comment lines, find the size line
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = re.split(r"\s+", s)
            n   = int(parts[0])
            nnz = int(parts[2]) if len(parts) >= 3 else 0
            return n, nnz, symmetric, pattern
    raise ValueError(f"no size line found in {path}")


def main():
    entries = []
    seen = {}
    for g in GROUPS:
        gdir = os.path.join(MTX_ROOT, g)
        if not os.path.isdir(gdir):
            continue
        files = sorted(glob.glob(os.path.join(gdir, "*.mtx")))
        for f in files:
            name = os.path.basename(f)[:-4]
            if name in EXCLUDE:
                continue
            try:
                n, nnz, sym, pat = read_header(f)
            except Exception as e:
                print(f"  SKIP {g}/{name}: {e}", file=sys.stderr)
                continue
            if name in seen:
                print(f"  WARN duplicate matrix name '{name}' "
                      f"({seen[name]} and {g}); plot files will collide",
                      file=sys.stderr)
            seen[name] = g
            tri = n * (n + 1) / 2.0          # lower-triangle incl. diagonal
            density = (nnz / tri * 100.0) if tri > 0 else 0.0
            entries.append(dict(name=name, group=g, dim=n, nnz=nnz,
                                density=density,
                                domain=APPLICATION.get(name, "General")))
        print(f"  {g}: {len(files)} matrices", file=sys.stderr)

    # sort by group number then dimension ascending
    def gnum(e):
        return int(e["group"].replace("group", ""))
    entries.sort(key=lambda e: (gnum(e), e["dim"]))

    with open(OUT, "w") as out:
        out.write("// Single source of truth for the matrix catalog.\n")
        out.write("// Auto-generated by gen_catalog.py from the MatrixMarket files\n")
        out.write(f"// under {MTX_ROOT} (group1..group12). Do not edit by hand;\n")
        out.write("// re-run gen_catalog.py to refresh.\n")
        out.write(f"// {len(entries)} matrices.\n")
        out.write("var MATRIX_CATALOG = [\n")
        for e in entries:
            out.write(
                '    {{ name: "{name}", group: "{group}", dim: {dim}, '
                'nnz: {nnz}, density: {dens:.4f}, domain: "{domain}" }},\n'.format(
                    name=e["name"], group=e["group"], dim=e["dim"],
                    nnz=e["nnz"], dens=e["density"], domain=e["domain"]))
        out.write("];\n")

    print(f"\nWrote {OUT}  ({len(entries)} matrices)", file=sys.stderr)


if __name__ == "__main__":
    main()
