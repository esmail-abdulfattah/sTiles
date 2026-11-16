using Pkg
Pkg.develop(path = "julia")
using Libdl, SparseArrays, LinearAlgebra

step(msg) = (println(msg); flush(stdout))

step("1 loading sTiles")
using sTiles
step("2 library: " * sTiles.library_path())
for p in Libdl.dllist()
    occursin(r"gomp|stdc\+\+|gcc_s|gfortran|winpthread|hwloc|openblas|stiles"i, p) && step("   dll " * p)
end
step("3 version: " * sTiles.version())
g(n) = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
F = sTiles.cholesky(g(60))
step("4 one factor, cores=1: " * string(logdet(F) ≈ logdet(Matrix(g(60)))))
close(F)
F = sTiles.cholesky(g(2000); cores = 2)
step("5 one factor, cores=2: " * string(logdet(F) ≈ logdet(Matrix(g(2000)))))
G = sTiles.cholesky(2g(500); cores = 2)
step("6 two factors: " * string(logdet(G) ≈ logdet(Matrix(2g(500)))))
close(F); close(G)
H = sTiles.cholesky(g(20); mode = :sparse)
step("7 sparse n=20: " * string(logdet(H) ≈ logdet(Matrix(g(20)))))
close(H)
step("OK")
