# =============================================================================
#  FuzzySphereLDiag -- 3d Ising CFT application
#
#      julia -t 4 examples/tutorial_ising.jl    
# =============================================================================
using FuzzifiED, LinearAlgebra, Printf
using FuzzySphereLDiag
FuzzifiED.SilentStd = true                     # keep FuzzifiED's own log quiet

head(s) = (println(); println("="^76); println("  ", s); println("="^76); println())

runtime_start = time()

head("PART 1.  FuzzifiED vs Exact L ED")

# -----------------------------------------------------------------------------
#  Step 1.  Define the parameters: particle number and couplings.
# -----------------------------------------------------------------------------
N  = 14        # electrons, and orbitals in the LLL (they are equal here)
V0 = 4.75      # Haldane pseudopotential: energy of the closest pair state
V1 = 1.0       # the next pair state out; this also sets the overall energy scale
h  = 3.16      # transverse field, near the critical value for these V's

@printf("  N = %d electrons,  V0 = %.2f,  V1 = %.1f,  h = %.2f\n\n", N, V0, V1, h)

# -----------------------------------------------------------------------------
#  Step 2.  Route one, the m-scheme: build the whole Lz = 0 block (FuzzifiED).
# -----------------------------------------------------------------------------
println("  Now running FuzzifiED ...")

FuzzifiED.ElementType = Float64                # this Hamiltonian is real

# The flavour structure, as 2x2 matrices acting on the flavour index.
n_up   = Float64[1 0; 0 0]                     # projector onto flavour up
n_down = Float64[0 0; 0 1]                     # projector onto flavour down
flip   = Float64[0 1; 1 0]                     # sigma_x: what the field couples to

# The two quantum numbers that label this basis: particle number and Lz.
qn_number = GetNeQNDiag(2N)                    # 2N spin-orbitals in total
qn_lz     = GetLz2QNDiag(N, 2)

# Everything up to the assembled matrix is "construction"; time it as such.
time_start = time()

# Every configuration of N electrons in those orbitals that has Lz = 0.
basis_m = Basis(Confs(2N, [N, 0], [qn_number, qn_lz]))

# The Hamiltonian, as a list of second-quantized terms.  Passing the two
# projectors separately generates only the (up, down) ordering of each pair,
# so the pseudopotentials are doubled to compensate.
interaction = GetDenIntTerms(N, 2, 2 .* [V0, V1], n_up, n_down)
field       = GetPolTerms(N, 2, flip)
terms_m     = SimplifyTerms(interaction - h * field)
matrix_m    = OpMat(Operator(basis_m, terms_m))

time_build_m = time() - time_start

# Then the eigensolve itself.
time_start   = time()
E0_m         = minimum(real.(GetEigensystem(matrix_m, 2)[1]))
time_solve_m = time() - time_start

@printf("  m-scheme (FuzzifiED)   E0 = %.12f\n", E0_m)
@printf("      block %d states,  %.2f s to construct + %.2f s to solve,  total %.2f s\n\n",
        basis_m.dim, time_build_m, time_solve_m, time_build_m + time_solve_m)

# -----------------------------------------------------------------------------
#  Step 3.  Route two, the exact-L basis: name the sector you want (this pkg).
# -----------------------------------------------------------------------------
println("  Now running the exact-L solver ...")

L_total   = 0        # total angular momentum of the sector
z2_parity = +1       # Ising parity, i.e. the flavour-exchange symmetry

time_start = time()
sector     = IsingSector(N, h, V0, V1, L_total, z2_parity)
time_build_L = time() - time_start

time_start   = time()
E0_L         = ising_levels(sector; k = 1)[1]  # k = how many levels to return
time_solve_L = time() - time_start

@printf("  exact-L  (this pkg)    E0 = %.12f\n", E0_L)
@printf("      block %d states,  %.2f s to construct + %.2f s to solve,  total %.2f s\n",
        sector.dim, time_build_L, time_solve_L, time_build_L + time_solve_L)
@printf("\n  the energies differ by %.1e;  block sizes %d and %d;  totals %.2f s and %.2f s\n\n",
        abs(E0_m - E0_L), basis_m.dim, sector.dim,
        time_build_m + time_solve_m, time_build_L + time_solve_L)

head("PART 2.  The order parameter from a one-dimensional emulator")

println("""  Fix V0 and scan the field alone.  The Hamiltonian is affine in both, so the
  one-parameter operator is the two-parameter one with the V0 piece folded into
  the constant part.  The same machinery serves, grouped differently.

  This emulator is grown against an observable rather than the energy.  The
  order parameter is m^2 = ||O|GS>||^2, with O the Z2-odd magnetization that
  maps the Z2 = +1 block onto Z2 = -1.  Assemble G = (O.Psi)'(O.Psi) from the
  snapshots and m^2 = y' G y at any field costs one reduced eigensolve.  The
  greedy stops once the largest relative change in m^2 along the scan, between
  consecutive snapshots, falls below m2_tol, set in the next step.
""")

# -----------------------------------------------------------------------------
#  Step 4.  Fix V0, and set the cut and the convergence tolerance.
# -----------------------------------------------------------------------------
V0_cut  = 4.75              # the fixed V0 for this cut
h_box   = (0.0, 4.5)        # the field range to cover
m2_tol  = 1e-5              # stop when m^2 changes by less than this
h_watch = h_box[1]:0.01:h_box[2]          # grid the stopping rule watches
h_scan  = range(h_box...; length = 25)    # coarser grid, for printing

# -----------------------------------------------------------------------------
#  Step 5.  Regroup the affine pieces: H(h) = (base + V0_cut * piece_V0) + h * piece_h
# -----------------------------------------------------------------------------
affine_2d, solve_2d = jscheme_ising_affine(N, L_total, z2_parity)
base, piece_V0, piece_h = affine_2d.pieces

affine_1d = AffineOperator(Function[x -> base(x) .+ V0_cut .* piece_V0(x),
                                    piece_h], affine_2d.dim)
solve_1d(θ, k, guess = nothing) = solve_2d([V0_cut, θ[1]], k, guess)

# -----------------------------------------------------------------------------
#  Step 6.  Grow it, watching m^2 along the whole cut.  greedy hands `watch` the
#           emulator built so far; whatever it returns is compared with the
#           previous snapshot's values.
# -----------------------------------------------------------------------------
watch(em) = (G = ising_order_param_gram(em, N);
             [reduced_expectation(em, G, (h,)) for h in h_watch])

time_start   = time()
emulator_1d  = greedy(affine_1d, solve_1d, [h_box]; k = 1, tol = 0.0,
                      maxsnap = 30, monitor = watch, monitor_tol = m2_tol)
time_1d      = time() - time_start

@printf("  V0 = %.2f fixed,  h in [%.1f, %.1f],  m^2 tolerance %.0e\n",
        V0_cut, h_box..., m2_tol)
@printf("  block %d states,  %d snapshots,  %.2f s\n\n",
        affine_1d.dim, length(emulator_1d.pts), time_1d)

# -----------------------------------------------------------------------------
#  Step 7.  Read the order parameter along the cut.
# -----------------------------------------------------------------------------
G_1d = ising_order_param_gram(emulator_1d, N)
m2   = [reduced_expectation(emulator_1d, G_1d, (h,)) for h in h_scan]

println("        h        m^2")
for (h, value) in zip(h_scan, m2)
    @printf("     %.4f  %9.3f   %s\n", h, value,
            "#"^round(Int, 46 * value / maximum(m2)))
end

h_check                = h_scan[end ÷ 2]
m2_emulated            = reduced_expectation(emulator_1d, G_1d, (h_check,))
energy_exact, m2_exact = ising_order_parameter(N, h_check, [V0_cut, V1])
@printf("\n  at h = %.4f:  emulated %.6f,  exact %.6f,  relative %.1e\n\n",
        h_check, m2_emulated, m2_exact, abs(m2_emulated - m2_exact) / m2_exact)

@printf("  m^2 starts at N^2 = %d, where the ground state is fully ordered, and\n", N^2)
println("""  falls as the field disorders it.  The order parameter is several orders of
  magnitude less accurate than the energy: a Ritz value is stationary in the
  error of the state, so its own error is second order in that error, while an
  expectation value is first order.
""")

head("PART 3.  A two-dimensional emulator over the (V0, h) plane")

# -----------------------------------------------------------------------------
#  Step 8.  Define the region of coupling space to cover, at the residual
#           tolerance set below.
# -----------------------------------------------------------------------------
box = [(3.8, 5.8),      # range of V0
       (3.02, 3.32)]    # range of h

residual_tol = 1e-9     # target for the worst certified residual over the box

@printf("  box:  V0 in [%.1f, %.1f]  x  h in [%.2f, %.2f],   tolerance %.0e\n\n",
        box[1]..., box[2]..., residual_tol)

# -----------------------------------------------------------------------------
#  Step 9.  Ask the package for what the emulator needs: the affine pieces of
#           H, and a function that solves exactly at one point.
# -----------------------------------------------------------------------------
affine, solve_exact = jscheme_ising_affine(N, L_total, z2_parity)

# -----------------------------------------------------------------------------
#  Step 10. Build the emulator.  greedy() adds one exact solve at a time, always
#           at the point of worst certified residual.  It stops when that
#           residual drops below residual_tol, or earlier if a new snapshot
#           would be numerically linearly dependent on the ones already held,
#           which means the stored basis already spans the solution manifold.
# -----------------------------------------------------------------------------
time_start = time()
emulator   = greedy(affine, solve_exact, box; k = 1, tol = residual_tol)
time_greedy = time() - time_start

@printf("  block %d states,  %d snapshots,  %.2f s,  certified residual %.1e\n",
        affine.dim, length(emulator.pts), time_greedy, emulator.res)
println(emulator.res < residual_tol ?
        "  it reached the tolerance" :
        "  it stopped short of the tolerance: the snapshots already span the manifold")
println()

# -----------------------------------------------------------------------------
#  Step 11. Draw the box as a character grid, with each snapshot marked by the
#           order the greedy chose it.
# -----------------------------------------------------------------------------
n_cols, n_rows = 40, 12
grid = fill(' ', n_rows, n_cols)

for (i, point) in enumerate(emulator.pts)
    V0_point, h_point = point[1], point[2]

    # map the coupling onto a cell; h runs upwards, so row 1 is the top of box
    frac_V0 = (V0_point   - box[1][1]) / (box[1][2] - box[1][1])
    frac_h  = (box[2][2] - h_point)    / (box[2][2] - box[2][1])
    col = clamp(1 + round(Int, frac_V0 * (n_cols - 1)), 1, n_cols)
    row = clamp(1 + round(Int, frac_h  * (n_rows - 1)), 1, n_rows)

    label = i < 10 ? '0' + i : 'a' + (i - 10)
    grid[row, col] = grid[row, col] == ' ' ? label : '*'   # '*' = two in one cell
end

println("  snapshots, in the order chosen (1-9, then a, b, ...; * = two in one cell):")
println()
@printf("  h = %.2f  +%s+\n", box[2][2], "-"^n_cols)
for row in 1:n_rows
    @printf("            |%s|\n", String(grid[row, :]))
end
@printf("  h = %.2f  +%s+\n", box[2][1], "-"^n_cols)

axis_left  = @sprintf("V0 = %.1f", box[1][1])
axis_right = @sprintf("%.1f", box[1][2])
gap = n_cols + 2 - length(axis_left) - length(axis_right)
@printf("            %s%s%s\n\n", axis_left, " "^max(gap, 1), axis_right)

println("""  The first snapshot lands in the middle of the box and the next four at its
  corners, where the stored basis constrains the surface least.  The rest fill
  in wherever the certified residual is largest.
""")

# -----------------------------------------------------------------------------
#  Step 12. Use it: a 40x40 energy surface, then a check at points the greedy
#           never visited, against fresh exact solves.
# -----------------------------------------------------------------------------
n_V0, n_h = 40, 40
V0_grid = range(box[1]...; length = n_V0)
h_grid  = range(box[2]...; length = n_h)

time_start = time()
surface    = [emulate(emulator, (v, f))[1] for v in V0_grid, f in h_grid]
time_surface = time() - time_start

@printf("  a %dx%d surface (%d points) evaluated in %.3f s\n\n",
        n_V0, n_h, n_V0 * n_h, time_surface)

println("  checked against exact solves at points the emulator never saw:")
for (v, f) in ((4.10, 3.07), (5.50, 3.29))
    emulated = emulate(emulator, (v, f))[1]
    exact    = solve_exact([v, f], 1)[1][1]
    @printf("    (V0=%.2f, h=%.2f):   error %.1e\n", v, f, abs(emulated - exact))
end

println()
@printf("""  That %dx%d grid by direct diagonalization would need %d solves; the emulator
  used %d and reproduces it to the errors above.
""", n_V0, n_h, n_V0 * n_h, length(emulator.pts))

println("  Things to try: widen box, or raise k to emulate excited states as well.")

@printf("  total runtime %.1f s\n", time() - runtime_start)
