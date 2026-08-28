"""
    FuzzySphereLDiag

Exact-(L,Q) diagonalisation and certified reduced-basis emulation for
fuzzy-sphere conformal field theories.

Two ingredients:

  * an angular-momentum-COUPLED (J-scheme) exact diagonalisation, in which the
    total angular momentum L and the flavour charge are exact by construction
    rather than assigned afterwards from <L^2>.  The symmetry blocks are one to
    two orders of magnitude smaller than the m-scheme (fixed-L_z) blocks that
    general fuzzy-sphere codes diagonalise.

  * a certified greedy reduced-basis emulator (eigenvector continuation).  Both
    models are exactly affine in their couplings, so a handful of exact solves
    generates a continuous energy surface over a whole coupling box, every
    point carrying a rigorous a-posteriori residual bound.

Models: the two-flavour Ising model and the three-flavour (spin-1) O(2) model.

Conventions that differ from FuzzifiED, and matter:

  * Ising: `ps_pot[l+1] = V_l`.  FuzzifiED's `GetDenIntTerms` takes
    `2 .* ps_pot` when the two flavour projectors are passed separately.
  * O(2): `D` is the anisotropy of the LITERAL Hamiltonian, as in Dey et al.,
    arXiv:2604.18705.
    FuzzifiED normal-orders, and its anisotropy is `D + kappa(N,V0,V1)/2`.
    Omitting the shift changes the spectrum entirely -- see `examples/tutorial.jl`.

Run `examples/tutorial.jl` for a cross-code check against FuzzifiED and a
reduced-basis energy surface.
"""
module FuzzySphereLDiag

using LinearAlgebra, Printf

include("JScheme.jl")            # defines JScheme, which includes O2Scheme.jl
using .JScheme

# ---------------------------------------------------------------- Ising -----
"""
    IsingSector(N, h, ps_pot, L, z2)

Exact-(L, Z2) block of the fuzzy-sphere Ising model: `N` orbitals, transverse
field `h`, Haldane pseudopotentials `ps_pot[l+1] = V_l`, Ising parity
`z2 = (-1)^{n_-}` in the sigma^x basis.
"""
const IsingSector = JScheme.Sector

"`ising_levels(sector; k)` -- the `k` lowest eigenvalues of an `IsingSector`."
ising_levels(s; k = 2, kwargs...) = JScheme.eigenvalues(s; k = k, kwargs...)
"`ising_states(sector; k)` -- eigenvalues and eigenvectors."
ising_states(s; k = 2, kwargs...) = JScheme.eigensystem(s; k = k, kwargs...)

# ------------------------------------------------------------------ O(2) ----
"""
    O2Sector(N, D, ps_pot, L, Q)

Exact-(L, Q) block of the three-flavour (spin-1) O(2) model.  `D` is the
LITERAL single-ion anisotropy (see [`kappa`](@ref)).
"""
const O2Sector = JScheme.O2Sector

"`o2_levels(sector; k)` -- the `k` lowest eigenvalues of an `O2Sector`."
o2_levels(s; k = 2, kwargs...) = JScheme.o2_eigenvalues(s; k = k, kwargs...)

"""
    o2_ground_cparity(sector, parity)

Lowest state of definite charge-conjugation parity inside a `Q = 0` sector.
Needed for the stress tensor: in the unresolved `(L=2, Q=0)` block the C-odd
descendant of the current lies BELOW `T`, and the two cross along the critical
line, so no fixed level index tracks `T`.
"""
const o2_ground_cparity = JScheme.o2_ground_cparity

"""
    kappa(N, V0, V1 = 1.0)

Normal-ordering shift, `kappa = ((2N-1)V0 - (2N-3)V1)/N`.  The literal
Hamiltonian at anisotropy `D` equals the normal-ordered one at `D + kappa/2`,
which is the convention FuzzifiED uses.
"""
kappa(N, V0, V1 = 1.0) = ((2N - 1) * V0 - (2N - 3) * V1) / N

# ---------------------------------------------------- reduced basis ---------
include("rbm.jl")

"""
    jscheme_ising_affine(N, L, z2)

Wrap the coupled-basis Ising sector `(L, z2)` as an [`AffineOperator`](@ref) plus a
solver, ready for [`greedy`](@ref).  Parameters are `θ = (V0, h)` at `V1 = 1`.

Returns `(H, solve)`.  This is the adapter pattern: any backend that can apply
its affine Hamiltonian pieces to a vector and solve at a point plugs in the
same way -- see `examples/tutorial.jl` for the FuzzifiED adapter.
"""
function jscheme_ising_affine(N, L, z2; dense_limit = 64)
    S0 = JScheme.Sector(N, 0.0, 0.0, 1.0, L, z2)    # V1 = 1 only
    Sg = JScheme.Sector(N, 0.0, 1.0, 1.0, L, z2)    # + V0 piece
    Sh = JScheme.Sector(N, 1.0, 0.0, 1.0, L, z2)    # + h piece
    ops = [JScheme.SectorOperator(s, JScheme.workspace(s)...) for s in (S0, Sg, Sh)]
    dim = S0.dim
    base(x) = (y = zeros(dim); mul!(y, ops[1], x); y)
    diff(j) = x -> (y = zeros(dim); mul!(y, ops[1], x);
                    z = zeros(dim); mul!(z, ops[j], x); z .- y)
    H = AffineOperator(Function[base, diff(2), diff(3)], dim)
    solve(θ, k) = JScheme.eigensystem(JScheme.Sector(N, θ[2], θ[1], 1.0, L, z2);
                                      k = k, dense_limit = dense_limit)
    H, solve
end

"""
    ising_emulator(N, L, z2; k, box, tol, maxsnap)

Convenience wrapper: build a certified emulator for the Ising sector `(L, z2)`
over `box = [(V0min, V0max), (hmin, hmax)]` using the coupled-basis solver.
"""
function ising_emulator(N, L, z2; k = 1, box = [(3.8, 5.8), (3.02, 3.32)],
                        tol = 1e-8, maxsnap = 60, dense_limit = 64,
                        verbose = false)
    H, solve = jscheme_ising_affine(N, L, z2; dense_limit = dense_limit)
    greedy(H, solve, box; k = k, tol = tol, maxsnap = maxsnap, verbose = verbose)
end

export IsingSector, O2Sector, ising_levels, ising_states, o2_levels,
       o2_ground_cparity, kappa,
       AffineOperator, Emulator, emulate, greedy, nparams,
       jscheme_ising_affine, ising_emulator, JScheme

end # module
