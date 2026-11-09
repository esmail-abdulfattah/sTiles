# Analyze the sparsity pattern once, then factorize many matrices that
# share it. This is the loop of a hyperparameter search or an optimizer.
#
#   julia 04_reuse_analysis.jl
using sTiles, LinearAlgebra, SparseArrays, Printf

n = 100_000
T = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))

build_Q(kappa) = T + kappa * I           # same non-zero pattern for every kappa

# Step 1, once: ordering and tile layout. They depend only on WHERE the
# non-zeros are, not on their values.
F = sTiles.analyze(build_Q(1.0); cores = 2)
@printf("analysis:  %.3f s  (paid once)\n", sTiles.analyze_time(F))

# Step 2, per value set: numeric Cholesky only.
for kappa in (0.1, 1.0, 10.0)
    sTiles.update!(F, build_Q(kappa))
    @printf("kappa = %5.1f   log|Q| = %12.4f   cholesky: %.4f s\n",
            kappa, logdet(F), sTiles.chol_time(F))
end

close(F)
