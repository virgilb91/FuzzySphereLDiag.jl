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

```julia
using Pkg
Pkg.develop(path = "/path/to/FuzzySphereLDiag")   # this directory
Pkg.test("FuzzySphereLDiag")                      # ~4.5 min, runs both validate suites
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

## Reduced-basis emulation — model and backend independent

The emulator knows only three things: the parameter box, the **affine pieces**
of the Hamiltonian as callables `H_j(x) -> H_j*x`, and a **solver** returning the
lowest exact eigenpairs at a point. It never learns which model or which ED code
it is talking to.

```julia
H = AffineOperator([x -> H1*x, x -> H2*x, x -> H3*x], dim)   # H(θ) = H1 + θ1 H2 + θ2 H3
em = greedy(H, solve, [(3.8, 5.8), (3.02, 3.32)]; k = 1, tol = 1e-9)
emulate(em, (4.75, 3.16))            # anywhere in the box, O(r^3)
em.res                               # certified residual bound
```

Any number of parameters is supported; `box` carries one `(lo, hi)` per
parameter. Adapters ship for the coupled-basis solver (`jscheme_ising_affine`,
`ising_emulator`); `examples/tutorial.jl` builds the FuzzifiED adapter in six
lines and drives the *same* `greedy` with it.

Why the greedy can work at all: both Hamiltonians are exactly affine, so for any
θ the **true** residual norm is available in `O(r^2)` without touching the full
space,

```
|| (H(θ) - E) Q y ||^2  =  y' B(θ) y - E^2        (Q'Q = I, |y| = 1)
```

a rigorous a-posteriori bound, not an estimate. Snapshots go where it is worst.

## Tutorial

```
julia -t 4 examples/tutorial.jl      # DEFAULT env, so FuzzifiED resolves
```

1. Ising ground state from FuzzifiED, then from the coupled basis — same number to
   `1e-14`, but from a 22-state block instead of 1064, already labelled `(L, Z2)`.
2. O(2) ground state both ways, `1e-13`, demonstrating the `kappa/2` shift.
3. Two reduced-basis scans of the same Ising surface, one driven by each ED
   backend through the identical `greedy`. Both pick 8 snapshots and reach the
   same certified residual; the two surfaces agree to `6e-14`, and a 40x40 grid
   costs 8 exact solves instead of 1600.

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

FuzzifiED, compared against in the tutorial, is separately MIT-licensed
(Copyright (c) 2024–2025 Zheng Zhou and contributors) and is a normal package
dependency of the tutorial only, not of the library.

