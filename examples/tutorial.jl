# =============================================================================
#  FuzzySphereLDiag tutorial
#
#      julia -t 4 examples/tutorial.jl
#
#  Requires FuzzySphereLDiag and FuzzifiED in the active environment:
#      using Pkg; Pkg.add(["FuzzySphereLDiag", "FuzzifiED"])
#
#  1. Ising ground state: FuzzifiED, then the coupled (exact-L) basis.
#  2. O(2)  ground state: FuzzifiED, then the coupled (exact-L) basis.
#  3. Two reduced-basis scans of the same Ising surface, one driven by each ED
#     backend through the SAME model-agnostic greedy.
#
#  The emulator is told only: the parameter box, the affine pieces of H as
#  callables, and a solver.  It never learns which model or which code it is
#  talking to, which is why part 3 is a two-line change between backends.
# =============================================================================
using FuzzifiED, LinearAlgebra, Printf
using FuzzySphereLDiag
FuzzifiED.SilentStd = true

head(s) = (println(); println("="^76); println("  ", s); println("="^76))

# =============================================================================
head("1.  ISING ground state    N=8, V0=4.75, V1=1, h=3.16")
# =============================================================================
const NI, V0I, HI = 8, 4.75, 3.16
FuzzifiED.ElementType = Float64
s1, s2, sx = Float64[1 0; 0 0], Float64[0 0; 0 1], Float64[0 1; 1 0]
bsI = Basis(Confs(2NI, [NI, 0], [GetNeQNDiag(2NI), GetLz2QNDiag(NI, 2)]))

# --- FuzzifiED.  GetDenIntTerms with the flavour projectors passed separately
# --- generates only the (up,down) pair ordering, hence the factor 2.
tmsI = SimplifyTerms(GetDenIntTerms(NI, 2, 2 .* [V0I, 1.0], s1, s2)
                     - HI * GetPolTerms(NI, 2, sx))
t = time(); eF = sort(real.(GetEigensystem(OpMat(Operator(bsI, tmsI)), 2)[1]))[1]
tF = time() - t
@printf("  FuzzifiED   E0 = %.12f     (Lz=0 block, dim %d, %.2f s)\n", eF, bsI.dim, tF)

# --- FuzzySphereLDiag.  Same state, but it arrives already labelled by (L, Z2).
t = time(); eJ = ising_levels(IsingSector(NI, HI, V0I, 1.0, 0, +1); k = 1)[1]
tJ = time() - t
@printf("  exact-L     E0 = %.12f     (L=0,Z2=+1 block, dim %d, %.2f s)\n",
        eJ, IsingSector(NI, HI, V0I, 1.0, 0, +1).dim, tJ)
@printf("  difference     %.2e\n", abs(eF - eJ))
println("  The diagonalised block is 22 states, not 1064, and L labels the block")
println("  rather than being measured afterwards from <L^2>.")

# =============================================================================
head("2.  O(2) ground state     N=8, V0=4, V1=1, D=2.96 (literal)")
# =============================================================================
const NO, V0O, DO = 8, 4.0, 2.96
κ = kappa(NO, V0O, 1.0)
@printf("  kappa = %.6f, so FuzzifiED runs at D_FED = D + kappa/2 = %.6f\n", κ, DO + κ/2)

FuzzifiED.ElementType = ComplexF64                     # S_y is complex
r = 1/sqrt(2)
Sx  = ComplexF64[0 r 0; r 0 r; 0 r 0]
Sy  = ComplexF64[0 -im*r 0; im*r 0 -im*r; 0 im*r 0]
Sz2 = ComplexF64[1 0 0; 0 0 0; 0 0 1]
tmsO = SimplifyTerms(GetDenIntTerms(NO, 3, [V0O, 1.0])
                     - 0.5*GetDenIntTerms(NO, 3, [V0O, 1.0], Sx)
                     - 0.5*GetDenIntTerms(NO, 3, [V0O, 1.0], Sy)
                     + (DO + κ/2) * GetPolTerms(NO, 3, Sz2))
qndO = [GetNeQNDiag(3NO), GetLz2QNDiag(NO, 3), GetFlavQNDiag(NO, 3, [1, 0, -1])]
t = time()
eFO = sort(real.(GetEigensystem(OpMat(Operator(Basis(Confs(3NO, [NO,0,0], qndO)), tmsO)), 2)[1]))[1]
tFO = time() - t
@printf("  FuzzifiED   E0 = %.12f     (Q=0, %.2f s)\n", eFO, tFO)
t = time(); eJO = o2_levels(O2Sector(NO, DO, [V0O, 1.0], 0, 0); k = 1)[1]; tJO = time() - t
@printf("  exact-L     E0 = %.12f     (L=0,Q=0, %.2f s)\n", eJO, tJO)
@printf("  difference     %.2e\n", abs(eFO - eJO))
println("  Comparing at equal D instead of equal D_FED makes these disagree at the")
println("  first digit: FuzzifiED normal-orders; D here is the literal coefficient.")

# =============================================================================
head("3.  TWO reduced-basis scans of the same Ising surface, one per backend")
# =============================================================================
const BOX = [(3.8, 5.8), (3.02, 3.32)]
@printf("  box: V0 in [%.1f, %.1f] x h in [%.2f, %.2f]\n\n", BOX[1]..., BOX[2]...)

# ---- backend A: this package's coupled-basis solver ----------------------
HA, solveA = jscheme_ising_affine(NI, 0, +1)
t = time(); emA = greedy(HA, solveA, BOX; k = 1, tol = 1e-9); tA = time() - t
@printf("  A  exact-L  : dim %5d, %2d snapshots, %5.2f s, certified residual %.1e\n",
        HA.dim, length(emA.pts), tA, emA.res)

# ---- backend B: FuzzifiED.  Same greedy, different affine pieces + solver.
FuzzifiED.ElementType = Float64
den(ps) = Operator(bsI, SimplifyTerms(GetDenIntTerms(NI, 2, 2 .* ps, s1, s2)))
opV1, opV0 = den([0.0, 1.0]), den([1.0, 0.0])
oph = Operator(bsI, SimplifyTerms(-GetPolTerms(NI, 2, sx)))
HB = AffineOperator(Function[x -> opV1*x, x -> opV0*x, x -> oph*x], bsI.dim)
solveB(θ, k) = GetEigensystem(OpMat(Operator(bsI,
    SimplifyTerms(GetDenIntTerms(NI, 2, 2 .* [θ[1], 1.0], s1, s2)
                  - θ[2]*GetPolTerms(NI, 2, sx)))), k)
t = time(); emB = greedy(HB, solveB, BOX; k = 1, tol = 1e-9, maxsnap = 40); tB = time() - t
@printf("  B  FuzzifiED: dim %5d, %2d snapshots, %5.2f s, certified residual %.1e\n",
        HB.dim, length(emB.pts), tB, emB.res)

# ---- the two surfaces, and each against its own backend's exact solves ----
nv = nh = 40
vs, hs = range(BOX[1]...; length = nv), range(BOX[2]...; length = nh)
t = time()
SA = [emulate(emA, (v, h))[1] for v in vs, h in hs]
SB = [emulate(emB, (v, h))[1] for v in vs, h in hs]
@printf("\n  two %dx%d surfaces (%d points each) evaluated in %.2f s total\n",
        nv, nh, nv*nh, time() - t)
@printf("  max |A - B| across the box = %.2e\n", maximum(abs, SA .- SB))
println("\n  off-snapshot check, each emulator against its OWN backend:")
for (v, h) in ((4.10, 3.07), (5.50, 3.29))
    xa = abs(emulate(emA, (v, h))[1] - solveA([v, h], 1)[1][1])
    xb = abs(emulate(emB, (v, h))[1] - sort(real.(solveB([v, h], 1)[1]))[1])
    @printf("    (V0=%.2f, h=%.2f):  A %.1e    B %.1e\n", v, h, xa, xb)
end
@printf("\n  A %dx%d grid by direct ED needs %d solves; the emulators used %d and %d.\n",
        nv, nh, nv*nh, length(emA.pts), length(emB.pts))
println("  greedy() is byte-identical between A and B -- only the AffineOperator")
println("  and the solver closure changed.")
