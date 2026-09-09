# =============================================================================
#  FuzzySphereLDiag -- the O(2) model and the anisotropy convention
#
#      julia -t 4 examples/tutorial_o2.jl             (Part 2 dominates the runtime)
#
#  Read examples/tutorial_ising.jl first: it explains what the fuzzy sphere is
#  and why a basis of definite total angular momentum is worth building.
#
#  THE MODEL.  Same lowest Landau level, but each electron now carries three
#  flavours instead of two, which is a spin-1 degree of freedom.  The
#  interaction is a flavour-blind repulsion minus an exchange in the xy plane
#  of that spin, and the tuning knob is a single-ion anisotropy D that pushes
#  the Sz = +-1 flavours up relative to Sz = 0.  Small D lets the exchange
#  order the spins in the plane; large D empties those flavours and leaves a
#  unique disordered state.  Between them sits the O(2), or XY, fixed point.
#
#  THE EXTRA LABEL.  The unbroken O(2) rotation is a conserved charge Q, so
#  sectors carry both L and Q here.  The ground state is (L, Q) = (0, 0).
#
#  THE CONVENTION.  D below is the literal coefficient in the Hamiltonian as
#  written.  FuzzifiED normal-orders its polarization term, so the same physics
#  needs D + kappa/2 there, with kappa supplied by this package.  Get this
#  wrong and the two codes disagree in the first digit.
#
#  Part 1 runs the comparison at the smaller of the two sizes set below, and
#  Part 2 at the larger, where the coupled basis overtakes the m-scheme one.
#  Part 3 builds a reduced-basis emulator over the (V0, D) plane and reads the
#  order parameter along a cut at fixed V0, as in Fig. 5 of the paper.
#
#  Needs both packages:  using Pkg; Pkg.add(["FuzzySphereLDiag", "FuzzifiED"])
# =============================================================================
using FuzzifiED, LinearAlgebra, Printf
using FuzzySphereLDiag
FuzzifiED.SilentStd = true                     # keep FuzzifiED's own log quiet

head(s) = (println(); println("="^76); println("  ", s); println("="^76); println())

runtime_start = time()

# -----------------------------------------------------------------------------
#  Step 1.  Define the couplings, shared by both sizes below.
# -----------------------------------------------------------------------------
V0 = 4.0       # Haldane pseudopotential: energy of the closest pair state
V1 = 1.0       # the next pair state out; this also sets the overall energy scale
D  = 2.96      # single-ion anisotropy, as it appears in the Hamiltonian we mean

N_small = 8    # the size Part 1 runs
N_large = 11   # the size Part 2 runs, chosen to sit past the crossover

# -----------------------------------------------------------------------------
#  Step 2.  One routine that computes the ground state both ways, so that
#           Part 2 runs exactly the same code as Part 1 at a larger N.
#           Returns the two wall times, for the comparison at the end.
# -----------------------------------------------------------------------------
function ground_state_both_ways(N; V0, V1, D)

    # --- the convention: convert our literal D to the one FuzzifiED expects.
    κ     = kappa(N, V0, V1)
    D_fed = D + κ / 2
    @printf("  N = %d electrons.  D = %.2f as written, so FuzzifiED needs\n", N, D)
    @printf("  D + kappa/2 = %.6f    (kappa = %.6f)\n\n", D_fed, κ)

    # ------ route one: name the sector and solve inside it (this package) ----
    println("  Now running the exact-L solver ...")
    L_total = 0                                # total angular momentum
    Q_charge = 0                               # O(2) charge

    time_start = time()
    sector     = O2Sector(N, D, [V0, V1], L_total, Q_charge)
    time_build = time() - time_start

    time_start = time()
    E0_L       = o2_levels(sector; k = 1)[1]   # k = how many levels to return
    time_solve = time() - time_start

    @printf("  exact-L  (this pkg)    E0 = %.12f\n", E0_L)
    @printf("      block %d states,  %.1f s to build + %.1f s to solve,  total %.1f s\n\n",
            sector.dim, time_build, time_solve, time_build + time_solve)

    # ------ route two: build the whole (Lz, Q) block and diagonalize it ------
    println("  Now running FuzzifiED ...")
    # The xy exchange brings in S_y, which is complex, so the element type
    # changes here; the Ising tutorial stayed real throughout.
    FuzzifiED.ElementType = ComplexF64

    # The spin-1 matrices acting on the three-flavour index.
    r        = 1 / sqrt(2)
    spin_x   = ComplexF64[0 r 0; r 0 r; 0 r 0]
    spin_y   = ComplexF64[0 -im*r 0; im*r 0 -im*r; 0 im*r 0]
    spin_z2  = ComplexF64[1 0 0; 0 0 0; 0 0 1]      # S_z^2, what D couples to

    # The three quantum numbers that label this basis.
    qn_number = GetNeQNDiag(3N)                     # 3N spin-orbitals in total
    qn_lz     = GetLz2QNDiag(N, 3)
    qn_charge = GetFlavQNDiag(N, 3, [1, 0, -1])     # the O(2) charge Q

    # Basis, terms and matrix are all construction; time them together so the
    # comparison with the exact-L route above counts the same kind of work.
    time_start = time()

    basis_m = Basis(Confs(3N, [N, 0, 0], [qn_number, qn_lz, qn_charge]))

    # The Hamiltonian: flavour-blind repulsion, minus half the xy exchange in
    # each of the two directions, plus the anisotropy in FuzzifiED's convention.
    repulsion  = GetDenIntTerms(N, 3, [V0, V1])
    exchange_x = GetDenIntTerms(N, 3, [V0, V1], spin_x)
    exchange_y = GetDenIntTerms(N, 3, [V0, V1], spin_y)
    anisotropy = GetPolTerms(N, 3, spin_z2)
    terms_m    = SimplifyTerms(repulsion - 0.5*exchange_x - 0.5*exchange_y
                               + D_fed * anisotropy)
    matrix_m   = OpMat(Operator(basis_m, terms_m))

    time_build_m = time() - time_start

    time_start   = time()
    E0_m         = minimum(real.(GetEigensystem(matrix_m, 2)[1]))
    time_solve_m = time() - time_start

    @printf("  m-scheme (FuzzifiED)   E0 = %.12f\n", E0_m)
    @printf("      block %d states,  %.1f s to construct + %.1f s to solve,  total %.1f s\n",
            basis_m.dim, time_build_m, time_solve_m, time_build_m + time_solve_m)
    @printf("\n  the energies differ by %.1e;  totals %.1f s and %.1f s\n",
            abs(E0_m - E0_L), time_build + time_solve, time_build_m + time_solve_m)

    return (exactL_build  = time_build,   exactL_solve  = time_solve,
            exactL_dim    = sector.dim,
            mscheme_build = time_build_m, mscheme_solve = time_solve_m,
            mscheme_dim   = basis_m.dim)
end

head("PART 1.  The ground state at N = $N_small, and the D convention")

small = ground_state_both_ways(N_small; V0, V1, D)

println()
println("""  Handing FuzzifiED the bare D instead makes the two numbers disagree in the
  first digit.  When comparing two codes, check whether each one normal-orders
  the anisotropy before reading anything into a disagreement.
""")

head("PART 2.  The same run at N = $N_large, where the coupled basis overtakes")

println("""  Only N changes.  At the smaller size the coupled basis is the slower of the
  two, because building its recoupling tables costs more than diagonalizing the
  m-scheme block.  A few electrons later the order is reversed.
""")

large = ground_state_both_ways(N_large; V0, V1, D)

println()
# Everything quoted below is measured by the two runs above, not written in.
total(r, kind) = kind === :L ? r.exactL_build + r.exactL_solve :
                               r.mscheme_build + r.mscheme_solve
steps    = N_large - N_small
growth_m = (large.mscheme_dim / small.mscheme_dim) ^ (1 / steps)
growth_L = (large.exactL_dim  / small.exactL_dim ) ^ (1 / steps)

@printf("""  exact-L divided by FuzzifiED wall time:   %.2f at N = %d,   %.2f at N = %d

  Block size grows by a factor %.1f per electron in the m-scheme against %.1f in
  the exact-L basis, so the gap keeps widening from here.  The balance inside
  the exact-L timing shifts as well: solve divided by build is %.2f at N = %d
  and %.2f at N = %d.

  Things to try: a larger N if you have a few minutes, or k = 5 in o2_levels to
  read off the low scaling dimensions of the O(2) CFT.
""", total(small, :L) / total(small, :m), N_small,
     total(large, :L) / total(large, :m), N_large,
     growth_m, growth_L,
     small.exactL_solve / small.exactL_build, N_small,
     large.exactL_solve / large.exactL_build, N_large)

head("PART 3.  A one-dimensional emulator, and the order parameter")

println("""  Fix V0 and scan the anisotropy alone.  The ordered phase breaks the O(2)
  rotation, so the order parameter carries charge and maps the (L, Q) = (0, 0)
  block onto (0, 1); on a finite sphere its expectation vanishes by that same
  symmetry, so the measurable quantity is m^2 = ||O|GS>||^2.

  This emulator is grown against m^2 rather than the energy: the greedy stops
  once the largest relative change in m^2 along the scan, between consecutive
  snapshots, falls below the tolerance below.
""")

# -----------------------------------------------------------------------------
#  Step 3.  Fix V0, and set the cut and the convergence tolerance.
# -----------------------------------------------------------------------------
V0_cut  = V0                # the fixed V0 for this cut
D_box   = (0.0, 5.2)        # the anisotropy range to cover
m2_tol  = 1e-5              # stop when m^2 changes by less than this
D_watch = D_box[1]:0.01:D_box[2]          # grid the stopping rule watches
D_scan  = range(D_box...; length = 25)    # coarser grid, for printing

# -----------------------------------------------------------------------------
#  Step 4.  Regroup the affine pieces: H(D) = (base + V0_cut * piece_V0) + D * piece_D
# -----------------------------------------------------------------------------
affine_2d, solve_2d = o2_affine(N_small, 0, 0)
base, piece_V0, piece_D = affine_2d.pieces

affine_1d = AffineOperator(Function[x -> base(x) .+ V0_cut .* piece_V0(x),
                                    piece_D], affine_2d.dim)
solve_1d(θ, k, guess = nothing) = solve_2d([V0_cut, θ[1]], k, guess)

# -----------------------------------------------------------------------------
#  Step 5.  Grow it, watching m^2 along the whole cut.
# -----------------------------------------------------------------------------
watch(em) = (G = o2_order_param_gram(em, N_small);
             [reduced_expectation(em, G, (d,)) for d in D_watch])

time_start  = time()
emulator_1d = greedy(affine_1d, solve_1d, [D_box]; k = 1, tol = 0.0,
                     maxsnap = 30, monitor = watch, monitor_tol = m2_tol)
time_1d     = time() - time_start

@printf("  N = %d,  V0 = %.1f fixed,  D in [%.1f, %.1f],  m^2 tolerance %.0e\n",
        N_small, V0_cut, D_box..., m2_tol)
@printf("  block %d states,  %d snapshots,  %.2f s\n\n",
        affine_1d.dim, length(emulator_1d.pts), time_1d)

# -----------------------------------------------------------------------------
#  Step 6.  Read the order parameter along the cut.
# -----------------------------------------------------------------------------
G_1d = o2_order_param_gram(emulator_1d, N_small)
m2   = [reduced_expectation(emulator_1d, G_1d, (d,)) for d in D_scan]

println("        D        m^2")
for (d, value) in zip(D_scan, m2)
    @printf("     %.4f  %8.3f   %s\n", d, value,
            "#"^round(Int, 46 * value / maximum(m2)))
end

D_check                = D_scan[end ÷ 2]
m2_emulated            = reduced_expectation(emulator_1d, G_1d, (D_check,))
energy_exact, m2_exact = o2_order_parameter(N_small, D_check, [V0_cut, V1])
@printf("\n  at D = %.4f:  emulated %.6f,  exact %.6f,  relative %.1e\n\n",
        D_check, m2_emulated, m2_exact, abs(m2_emulated - m2_exact) / m2_exact)

@printf("  m^2 falls by %.0f%% across the cut as the anisotropy empties the\n",
        100 * (m2[1] - m2[end]) / m2[1])
println("""  Sz = +-1 flavours and destroys the planar order.  Fig. 5 of the paper reads
  the transition off curves like this one at several N.
""")

head("PART 4.  A two-dimensional emulator over the (V0, D) plane")

println("""  Both couplings free now, and the energy alone.  The greedy stops on the
  certified residual, which bounds the error of every emulated point without
  touching the full space.
""")

# -----------------------------------------------------------------------------
#  Step 7.  The region to cover, and the residual tolerance.
# -----------------------------------------------------------------------------
box          = [(3.0, 5.0),      # range of V0
                (2.0, 3.8)]      # range of D
residual_tol = 1e-9

time_start  = time()
emulator_2d = greedy(affine_2d, solve_2d, box; k = 1, tol = residual_tol)
time_2d     = time() - time_start

@printf("  N = %d,  box V0 in [%.1f, %.1f] x D in [%.1f, %.1f],  tolerance %.0e\n",
        N_small, box[1]..., box[2]..., residual_tol)
@printf("  block %d states,  %d snapshots,  %.2f s,  certified residual %.1e\n\n",
        affine_2d.dim, length(emulator_2d.pts), time_2d, emulator_2d.res)

# -----------------------------------------------------------------------------
#  Step 8.  A surface over the plane, then a check off the snapshots.
# -----------------------------------------------------------------------------
n_V0, n_D = 30, 30
V0_grid   = range(box[1]...; length = n_V0)
D_grid    = range(box[2]...; length = n_D)

time_start   = time()
surface      = [emulate(emulator_2d, (v, d))[1] for v in V0_grid, d in D_grid]
time_surface = time() - time_start

@printf("  a %dx%d surface (%d points) evaluated in %.3f s\n\n",
        n_V0, n_D, n_V0 * n_D, time_surface)

println("  checked against exact solves at points the emulator never saw:")
for (v, d) in ((3.40, 2.30), (4.70, 3.55))
    emulated = emulate(emulator_2d, (v, d))[1]
    exact    = solve_2d([v, d], 1)[1][1]
    @printf("    (V0=%.2f, D=%.2f):   error %.1e\n", v, d, abs(emulated - exact))
end

@printf("\n  that grid by direct diagonalization would need %d solves; the emulator\n",
        n_V0 * n_D)
@printf("  used %d.\n\n", length(emulator_2d.pts))

println("""  Things to try: a larger N, or k = 5 in o2_levels to read off the low scaling
  dimensions of the O(2) CFT.
""")

@printf("  total runtime %.1f s\n", time() - runtime_start)
