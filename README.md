# FuzzySphereLDiag

Exact diagonalisation of fuzzy-sphere CFT models directly in the coupled
`(L, Z2)` / `(L, Q)` basis, so total angular momentum and the internal charge
are good quantum numbers by construction rather than by post-hoc projection of
an m-scheme spectrum. Two models ship: the Ising model in the J-scheme
(`JScheme`, basis `|[(n_+, J_+, a_+) x (n_-, J_-, a_-)] L>` with exact
`Z2 = (-1)^{n_-}`), and its three-flavour O(2) extension (`O2Scheme`,
charge-resolved sectors with a charge-conjugation projector). Every two-body
term factorises NuShellX-style into per-orbit reduced matrix elements tied by a
6j, so the Lanczos matvec is a threaded sum of small dense products; the
spectra are pinned against FuzzifiED and an independent Python port.

## Install

Clone the package, in a shell:

```sh
git clone https://github.com/virgilb91/FuzzySphereLDiag.jl
```

then pick whichever of the following suits you. The Julia snippets below are
written for the `julia>` prompt; at the `pkg>` prompt (reached with `]`) the
same commands drop `Pkg.`, the parentheses and the quotes, as noted under each.

**Use its own environment.** `Project.toml` lists every dependency, so nothing
else is needed:

```sh
julia --project=/path/to/FuzzySphereLDiag.jl
```

```julia
julia> using Pkg; Pkg.instantiate()      # first time only
julia> using FuzzySphereLDiag
```

At the `pkg>` prompt the first line is just `instantiate`.

**Track the clone from an environment of your own**, so edits to the source take
effect without reinstalling:

```julia
julia> using Pkg
julia> Pkg.develop(path = "/path/to/FuzzySphereLDiag.jl")
julia> using FuzzySphereLDiag
```

At the `pkg>` prompt: `dev /path/to/FuzzySphereLDiag.jl`.

**Skip the package manager entirely.** The module can be included directly, as
long as Arpack, LinearAlgebra, Printf, Random and SparseArrays are available in
the active environment. Included this way it is a local module, so the names
live under `.FuzzySphereLDiag`:

```julia
julia> include("/path/to/FuzzySphereLDiag.jl/src/FuzzySphereLDiag.jl")
julia> using .FuzzySphereLDiag
```

**Or install straight from the URL**, which copies a fixed commit into the depot
instead of tracking a clone:

```julia
julia> using Pkg
julia> Pkg.add(url = "https://github.com/virgilb91/FuzzySphereLDiag.jl")
```

At the `pkg>` prompt: `add https://github.com/virgilb91/FuzzySphereLDiag.jl`.

The tutorials additionally compare against
[FuzzifiED](https://github.com/FuzzifiED/FuzzifiED.jl), which is not a dependency
of the library itself:

```julia
julia> Pkg.add("FuzzifiED")          # only for the examples/
```

To run the test suite, from an environment where the package is developed:

```julia
julia> Pkg.test("FuzzySphereLDiag")  # a couple of minutes, both validate suites
```

## Quick start

```julia
using FuzzySphereLDiag
const J = FuzzySphereLDiag.JScheme          # full solver API lives here

# Ising: exact-(L, Z2) blocks
ising_levels(IsingSector(12, 3.153, [4.75, 1.0], 2, +1); k = 3)
J.solve(16, 3.153; L = 0, z2 = +1, k = 2)          # shorthand, V = (4.75, 1.0)

# O(2): exact-(L, Q), literal anisotropy D
o2_levels(O2Sector(8, 2.9600, [4.0, 1.0], 0, 0); k = 3)
o2_ground_cparity(O2Sector(8, 2.8747, [4.0, 1.0], 2, 0), +1)   # C-even lowest (T)

kappa(8, 4.0, 1.0)                                  # 5.875, see Conventions
```

`IsingSector(N, h, ps_pot, L, z2)` takes `ps_pot[l+1] = V_l`; the form
`IsingSector(N, h, V0, V1, L, z2)` is shorthand for `ps_pot = [V0, V1]`.

## Reduced-basis emulation

The emulator is independent of the model and of the ED backend. It takes the
parameter box, the **affine pieces** of the Hamiltonian as callables
`H_j(x) -> H_j*x`, and a **solver** returning the lowest exact eigenpairs at a
point. Nothing else about the model or the ED code enters.

```julia
H = AffineOperator([x -> H1*x, x -> H2*x, x -> H3*x], dim)   # H(θ) = H1 + θ1 H2 + θ2 H3
em = greedy(H, solve, [(3.8, 5.8), (3.02, 3.32)]; k = 1, tol = 1e-9)
emulate(em, (4.75, 3.16))            # anywhere in the box, O(r^3)
em.res                               # certified residual bound
```

Any number of parameters is supported; `box` carries one `(lo, hi)` per
parameter. Adapters ship for both coupled-basis solvers — `jscheme_ising_affine`
with `θ = (V0, h)`, `o2_affine` with `θ = (V0, D)`, plus the `ising_emulator`
convenience wrapper — and the two tutorials drive `greedy` with them. Any other
backend that can apply its affine pieces to a vector and solve at a point plugs
into the same `greedy` unchanged.

Why the greedy can work at all: both Hamiltonians are exactly affine, so for any
θ the **true** residual norm is available in `O(r^2)` without touching the full
space,

```
|| (H(θ) - E) Q y ||^2  =  y' B(θ) y - E^2        (Q'Q = I, |y| = 1)
```

which is a rigorous a-posteriori bound rather than an estimate. Each new
snapshot is placed where that bound is largest.

### Observables

The energy is a Ritz value of the projected matrices alone. An observable is a
norm in the full space, so it needs the snapshot basis, which the emulator
carries in `em.basis`. For a symmetry-odd operator `O` the quantity that locates
a transition is the second moment `m^2 = ||O|GS>||^2`; assemble
`G = (O Ψ)'(O Ψ)` once, and it is free at every coupling afterwards.

```julia
G = ising_order_param_gram(em, N)          # r applications of O, once
reduced_expectation(em, G, (4.75, 3.16))   # m^2 anywhere in the box
emulate_state(em, θ)                       # values and reduced vectors
```

`o2_order_param_gram` is the O(2) counterpart, where `O` raises the charge and
maps `(L, Q)` onto `(L, Q+1)`. For checking, `ising_order_parameter` and
`o2_order_parameter` give `m^2` from a direct diagonalisation.

An emulator can also be grown against an observable instead of the residual.
`greedy(...; monitor, monitor_tol)` hands `monitor` the emulator built so far
and stops when the largest relative change in what it returns, between
consecutive snapshots, falls below `monitor_tol`. That is the rule used for the
order-parameter figures in the papers. The energy
typically agrees to `1e-13` and an observable to `1e-9`, because a Ritz value is
stationary in the error of the state and its own error is second order in it,
while an expectation value is first order.

## Tutorials

One per model, each self-contained:

```
julia -t 4 examples/tutorial_ising.jl   # well under a minute
julia -t 4 examples/tutorial_o2.jl      # its middle part dominates the runtime
```

Both need FuzzySphereLDiag and FuzzifiED in the active environment. Each is a
sequence of numbered steps that announces which solver is running, reports the
construction and solve time of each, and prints a total runtime at the end.
Every figure quoted in the text is computed by the run.

`tutorial_ising.jl`

1. One ground-state energy from FuzzifiED and from the coupled basis: the same
   number to `1e-13`, from a block orders of magnitude smaller and already
   labelled `(L, Z2)`.
2. A one-parameter emulator at fixed `V0`, grown until the order parameter stops
   changing, then `m^2` along the field cut. At zero field it returns `N^2`
   exactly, the fully ordered value.
3. A two-parameter emulator over the `(V0, h)` plane, with the snapshot
   locations drawn as a character grid in the order the greedy chose them, then
   a surface from a handful of exact solves, checked off the snapshots.

`tutorial_o2.jl`

1. The O(2) ground state both ways, demonstrating the `kappa/2` shift. Omitting
   it makes the two codes disagree in the first digit.
2. The same run at a larger `N`, where the coupled basis overtakes the m-scheme
   one; the growth rates and the crossover are computed from the two runs.
3. A one-parameter emulator at fixed `V0`, grown against `m^2`, giving the
   anisotropy cut of Fig. 5 of the O(2) paper.
4. A two-parameter emulator over the `(V0, D)` plane, on the energy alone.

## References

The models and the methods this package implements:

- W. Zhu, C. Han, E. Huffman, J. S. Hofmann, Y.-C. He,
  *Uncovering conformal symmetry in the 3D Ising transition*,
  Phys. Rev. X **13**, 021009 (2023) — the fuzzy-sphere construction.
- A. Dey, L. Herviou, C. Mudry, S. Rychkov, A. M. Läuchli,
  *Conformal data for the O(2) Wilson–Fisher CFT ... on the fuzzy sphere*,
  arXiv:2604.18705 — the three-flavour O(2) model used here.
- K. J. Wiese, *Locating the Ising conformal field theory via the ground-state
  energy on the fuzzy sphere*, Phys. Rev. B **113**, 085106 (2026).
- Z. Zhou, *FuzzifiED: A Julia package for numerics on the fuzzy sphere*,
  arXiv:2503.00100 — the reference implementation compared against here.
- D. Frame *et al.*, *Eigenvector continuation with subspace learning*,
  Phys. Rev. Lett. **121**, 032501 (2018); T. Duguet, A. Ekström,
  R. J. Furnstahl, S. König, D. Lee, *Colloquium: Eigenvector continuation and
  projection-based emulators*, Rev. Mod. Phys. **96**, 031002 (2024).
- E. Caurier *et al.*, Rev. Mod. Phys. **77**, 427 (2005); B. A. Brown and
  W. D. M. Rae, Nucl. Data Sheets **120**, 115 (2014) — the J-coupled
  (NuShellX/NATHAN) factorisation the solver is built on.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Virgil V. Baran.

FuzzifiED, compared against in the tutorials, is separately MIT-licensed
(Copyright (c) 2024–2025 Zheng Zhou and contributors) and is a normal package
dependency of the examples only, not of the library.

