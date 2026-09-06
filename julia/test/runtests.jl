# Cross-checks against dense LinearAlgebra on small SPD matrices, plus the
# lifecycle and error paths. Run from the repo so binaries/ is found, or with
# STILES_LIB set:   julia --project=. -e 'using Pkg; Pkg.test()'
using Test
using sTiles
using LinearAlgebra
using SparseArrays
using Random

# A banded SPD test matrix: diagonal 4, off-diagonals -1 (a 1-D GMRF precision).
band_spd(n) = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))

# A random sparse SPD matrix with a given density (A'A + n*I).
function random_spd(n; density = 0.05, rng = MersenneTwister(1))
    A = sprandn(rng, n, n, density)
    return sparse(A' * A + n * I)
end

# Only one factor may be live at a time, so every test releases its factor
# even when an assertion throws; otherwise one failure cascades into every
# later testset.
closing(f, F) = try f(F) finally close(F) end

@testset "sTiles.jl" begin
    @test isfile(sTiles.library_path())
    @test sTiles.version() isa String

    @testset "cholesky / logdet / solve" begin
        n = 60
        Q = band_spd(n)
        Qd = Matrix(Q)
        closing(sTiles.cholesky(Q)) do F
            @test F isa sTiles.Factor
            @test size(F) == (n, n)
            @test sTiles.isfactored(F)
            @test isopen(F)
            @test nnz(F) >= n
            @test logdet(F) ≈ logdet(Qd) rtol = 1e-12
            @test det(F) ≈ det(Qd) rtol = 1e-10
            @test logabsdet(F)[2] == 1.0

            b = collect(1.0:n)
            x = F \ b
            @test x ≈ Qd \ b rtol = 1e-12
            @test norm(Q * x - b) / norm(b) < 1e-12
            @test b == collect(1.0:n)                 # untouched

            B = randn(MersenneTwister(2), n, 3)
            X = solve(F, B)
            @test X ≈ Qd \ B rtol = 1e-12
            Y = copy(B)
            @test solve!(F, Y) === Y
            @test Y ≈ X

            Z = similar(B)
            @test ldiv!(Z, F, B) === Z
            @test Z ≈ X
            W = copy(B)
            @test ldiv!(F, W) === W
            @test W ≈ X
            z = similar(b)
            @test ldiv!(z, F, b) ≈ x

            # forward then backward must equal the full solve
            y = solve(F, b; system = :L)
            @test solve(F, y; system = :Lt) ≈ x rtol = 1e-12
            @test_throws ArgumentError solve(F, b; system = :foo)
            @test_throws DimensionMismatch solve(F, ones(n + 1))

            s = sTiles.summary(F)
            @test s.n == n && s.nnz == nnz(tril(Q)) && s.factored
            @test s.chol_time isa Float64
            @test s.selinv_time === nothing
            @test sTiles.chol_time(F) >= 0
            @test sTiles.analyze_time(F) >= 0
            @test occursin("factorized", sprint(show, F))
            @test occursin("logdet", sprint(show, MIME("text/plain"), F))

            @test sort(sTiles.permutation(F)) == 1:n

            close(F)
            @test !isopen(F)
            @test occursin("closed", sprint(show, F))
            @test_throws sTilesError logdet(F)
            close(F)                                  # idempotent
        end
    end

    @testset "selected inverse" begin
        n = 40
        Q = random_spd(n)
        Qinv = inv(Matrix(Q))
        closing(sTiles.cholesky(Q; inverse = true)) do F
            @test selinv!(F) === F
            @test sTiles.summary(F).selinv_time isa Float64
            d = selinv_diag(F)
            @test length(d) == n
            @test d ≈ diag(Qinv) rtol = 1e-10
            for i in (1, 7, n)
                @test selinv_elm(F, i, i) ≈ Qinv[i, i] rtol = 1e-10
            end
            # every in-pattern entry matches the dense inverse; off-pattern is 0
            nmatch = 0
            for j in 1:n, i in j:n
                z = selinv_elm(F, i, j)
                @test z == selinv_elm(F, j, i)
                if z != 0
                    @test z ≈ Qinv[i, j] rtol = 1e-8
                    nmatch += 1
                end
            end
            @test nmatch >= n
            r = selinv_row(F, 5, [5, 1, 2])
            @test r[1] ≈ Qinv[5, 5] rtol = 1e-10
            @test r[2] == selinv_elm(F, 5, 1)
            @test_throws BoundsError selinv_elm(F, 0, 1)
            @test_throws BoundsError selinv_elm(F, 1, n + 1)
        end
        closing(sTiles.cholesky(Q)) do F              # no inverse storage
            @test_throws sTilesError selinv_diag(F)
        end
    end

    @testset "chol_elm" begin
        n = 30
        Q = band_spd(n)
        closing(sTiles.cholesky(Q)) do F
            @test 2 * sum(log(sTiles.chol_elm(F, i, i)) for i in 1:n) ≈ logdet(F) rtol = 1e-10
            @test sTiles.chol_elm(F, 1, 1) > 0
        end
    end

    @testset "analyze then factorize, reuse pattern" begin
        n = 50
        Q1 = band_spd(n)
        closing(sTiles.analyze(Q1)) do F
            @test !sTiles.isfactored(F)
            @test_throws sTilesError logdet(F)
            @test_throws sTilesError F \ ones(n)
            @test sTiles.summary(F).chol_time === nothing
            @test occursin("not factorized", sprint(show, F))
            sTiles.factorize!(F)
            @test logdet(F) ≈ logdet(Matrix(Q1)) rtol = 1e-12

            Q2 = 2.0 * Q1                             # same pattern
            sTiles.update!(F, Q2)
            @test logdet(F) ≈ logdet(Matrix(Q2)) rtol = 1e-12
            @test F \ ones(n) ≈ Matrix(Q2) \ ones(n) rtol = 1e-12

            Q3 = Q1 + spdiagm(2 => fill(0.1, n - 2), -2 => fill(0.1, n - 2))
            @test_throws sTilesError sTiles.factorize!(F, Q3)   # other pattern
            @test logdet(F) ≈ logdet(Matrix(Q2)) rtol = 1e-12   # still usable
        end
    end

    @testset "inputs" begin
        n = 20
        Q = band_spd(n)
        ld = logdet(Matrix(Q))
        closing(sTiles.cholesky(Matrix(Q))) do F      # dense input
            @test logdet(F) ≈ ld rtol = 1e-12
        end
        U = copy(Q); U[1, 2] = 100.0                  # upper triangle ignored
        closing(sTiles.cholesky(U)) do F
            @test logdet(F) ≈ ld rtol = 1e-12
        end
        for m in (:dense, "semisparse", sTiles.MODE_SPARSE, :auto)
            closing(sTiles.cholesky(Q; mode = m)) do F
                @test logdet(F) ≈ ld rtol = 1e-10
            end
        end
        @test_throws ArgumentError sTiles.cholesky(Q; mode = :bogus)
        @test_throws ArgumentError sTiles.cholesky(sprand(5, 6, 0.5))
        closing(sTiles.cholesky(random_spd(80); cores = 2)) do F
            @test sTiles.summary(F).cores == 2
            @test F \ ones(80) ≈ Matrix(random_spd(80)) \ ones(80) rtol = 1e-10
        end
    end

    @testset "errors and lifecycle" begin
        n = 10
        Q = band_spd(n)
        closing(sTiles.cholesky(Q)) do F
            @test_throws sTilesError sTiles.cholesky(Q)   # one live factor
        end
        closing(sTiles.cholesky(Q)) do F              # clean re-init after quit
            @test logdet(F) ≈ logdet(Matrix(Q)) rtol = 1e-12
        end
        # not positive definite: clean error, nothing left live
        N = spdiagm(0 => [1.0, -1.0, 1.0])
        @test_throws sTilesError sTiles.cholesky(N)
        closing(sTiles.cholesky(Q)) do F
            @test logdet(F) ≈ logdet(Matrix(Q)) rtol = 1e-12
        end
    end

    @testset "estimate_memory" begin
        gb = sTiles.estimate_memory(1000, 5000)
        @test gb isa Float64 && gb >= 0
    end
end
