using FuzzySphereLDiag
using Test

const J = FuzzySphereLDiag.JScheme   # O2Scheme.jl is include()d into JScheme

@testset "FuzzySphereLDiag" begin

    # ---- the two built-in validation suites -------------------------------
    # Both print a "[PASS]/[FAIL]" line per check and return 0 iff every
    # check passed (JScheme.jl:1117 and O2Scheme.jl:1074, both end with
    # `return fails == 0 ? 0 : 1`).  Exit-code 0 is the pass signal.
    @testset "validate() suites" begin
        @test J.validate() == 0        # Ising: orbit tables, dipole exact
                                       # solution, m-scheme dims, matvec vs
                                       # dense, FuzzifiED single-l channels,
                                       # 30 reference energies
        @test J.o2_validate() == 0     # O(2): 6j symmetry, chain scalars,
                                       # literal-H brute force N = 4,5,
                                       # FuzzifiED N = 8 anchors, n_pm_max
                                       # variationality, Hellmann-Feynman,
                                       # set_d!, matvec, charge conjugation
    end

    # ---- cheap direct pins on known values --------------------------------
    @testset "Ising reference energies (h = 3.153, V = (4.75, 1.0))" begin
        # JScheme.REFERENCE, cross-checked against an independent Python ED
        # implementation and against FuzzifiED
        @test J.solve(6, 3.153; L = 0, z2 = +1, k = 2) ≈
              [-10.714531, -2.527597]  atol = 3e-5
        @test J.solve(8, 3.153; L = 2, z2 = +1, k = 2) ≈
              [1.992050, 3.597643]     atol = 3e-5
    end

    @testset "sector dimension == m-scheme count" begin
        @test IsingSector(8, 3.153, 4.75, 1.0, 0, +1).dim ==
              J.mscheme_sector_dim(8, 0, +1) == 22
    end

    @testset "O(2) FuzzifiED anchor (literal D = 2.96, V = (4.0, 1.0))" begin
        # O2Scheme.O2_REFERENCE_N8[0], absolute energies, charge Q = 0
        @test J.o2_union(8, 2.9600, [4.0, 1.0], 0, 0; Lmax = 8)[1:3] ≈
              [23.68226059, 31.11270750, 33.36820456]  atol = 5e-8
        # convention arithmetic: D_FED = D + kappa/2,
        # kappa = [(2N-1) V0 - (2N-3) V1] / N
        let N = 8, V0 = 4.0, V1 = 1.0
            @test kappa(N, V0, V1) ≈ 5.875
            @test 2.9600 + kappa(N, V0, V1) / 2 ≈ 5.8975
        end
    end

    @testset "Wigner caches survive threaded use" begin
        # cg() once used a plain shared Dict with get!; concurrent writes
        # corrupt it during rehash, which aborts any threaded build.
        args = vec([(2a, 2b, 2c, 2d, 2e, 2f) for a in 0:3, b in 0:3, c in 0:3,
                                                 d in 0:3, e in 0:2, f in 0:2])
        ref6 = [J.sixj(x...) for x in args]
        refc = [J.cg(x[1], 0, x[2], 0, x[3], 0) for x in args]
        got6 = fill(NaN, length(args)); gotc = fill(NaN, length(args))
        Threads.@threads :static for i in eachindex(args)
            got6[i] = J.sixj(args[i]...)
            gotc[i] = J.cg(args[i][1], 0, args[i][2], 0, args[i][3], 0)
        end
        @test got6 == ref6
        @test gotc == refc
        J.merge_wigner_caches!()
        @test [J.sixj(x...) for x in args] == ref6
        @test [J.cg(x[1], 0, x[2], 0, x[3], 0) for x in args] == refc
    end

end
